"""Smart Claims -- Databricks App (Streamlit).

P5 SCOPE: prove the plumbing, nothing more. One number from each of the two backends, plus the
signed-in user. No claim submission, no admin screens -- those come in later steps, once this is
deployed and known to work.

WHY TWO BACKENDS. The app reads the same lakehouse two different ways, on purpose:

  * SQL warehouse -> whole-table analytics over ~13k gold rows. Seconds, and fine for that.
  * Lakebase      -> single-policy point lookups on the claim submission path, and the app's own
                     transactional writes. Sub-second, which the warehouse cannot do.

The transcript's reasoning for the second one: "if we go always through the SQL warehouse to
access our data... you would always have some kind of latency because the SQL warehouse is more
of an analytical compute option".
"""

import io
import os
import uuid
from pathlib import Path

import pandas as pd
import psycopg2
import streamlit as st
from openai import OpenAI
from databricks import sql
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

# MUST be the first Streamlit call in the file. Anything above it -- even st.write in a debug
# line -- makes this raise.
st.set_page_config(page_title="Smart Claims", page_icon="🚗", layout="wide")

# Fully-qualified name of the gold table, handed over already resolved by the `gold-claims-table`
# app resource. Assembling it here from a catalog and a schema would mean the app knowing which
# target it is running in, which it has no good way to find out.
CLAIMS_TABLE = os.getenv("CLAIMS_TABLE", "")

# The Postgres copy always lives at public.<same table name>, so the last segment is enough.
PG_CLAIMS_TABLE = f"public.{CLAIMS_TABLE.split('.')[-1]}" if CLAIMS_TABLE else ""

LAKEBASE_ENDPOINT = os.getenv("LAKEBASE_ENDPOINT", "")

# Model id arrives as config (see src/app/app.yaml), so switching models is a redeploy of that
# file rather than a code change. Verified against the key: chat.completions and the newer
# responses API both accept gpt-5.5; chat.completions is used below because its `messages` list
# maps straight onto conversation history when that arrives.
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")


# --- identity ---------------------------------------------------------------------------------
def current_user() -> str:
    """Email of the signed-in viewer, or a placeholder when running locally.

    Data access does NOT use this -- every query runs as the app's service principal. It exists
    so a submitted claim can record WHO filed it, which is real attribution rather than the
    transcript's customer/admin mode toggle.

    The forwarded headers only exist in a DEPLOYED app, so this is always the fallback locally.
    The exact header name is worth confirming from the deployed app rather than trusting this
    list -- `x-forwarded-access-token` is the only one the docs name explicitly.
    """
    try:
        headers = st.context.headers or {}
    except Exception:
        return "local-dev"
    for key in ("x-forwarded-email", "x-forwarded-preferred-username", "x-forwarded-user"):
        value = headers.get(key)
        if value:
            return value
    return "unknown"


# --- backend 1: SQL warehouse ------------------------------------------------------------------
@st.cache_resource
def warehouse_connection():
    """Connection to the SQL warehouse, as the app's service principal.

    `Config()` finds DATABRICKS_CLIENT_ID / DATABRICKS_CLIENT_SECRET, which the platform injects
    into a deployed app; locally it falls back to the CLI profile. Same code either way, which is
    the point of using Config() rather than reading env vars by hand.
    """
    cfg = Config()
    return sql.connect(
        server_hostname=cfg.host,
        http_path=f"/sql/1.0/warehouses/{os.environ['DATABRICKS_WAREHOUSE_ID']}",
        credentials_provider=lambda: cfg.authenticate,
    )


# --- backend 2: Lakebase Postgres --------------------------------------------------------------
# The TTL is load bearing. A Lakebase credential is an OAuth token that expires after 60 minutes
# (measured, not assumed), and a cached connection holds a token that will die with it. 40 minutes
# leaves a 20 minute margin, so a connection handed out just before expiry is still usable.
#
# This is the one piece of machinery the AppKit (TypeScript) client would have provided for free:
# its lakebase plugin refreshes tokens with a 2-minute buffer. On the Python side it is ours.
@st.cache_resource(ttl=2400)
def lakebase_connection():
    if not LAKEBASE_ENDPOINT:
        raise RuntimeError("LAKEBASE_ENDPOINT is unset -- is the postgres resource attached?")
    w = WorkspaceClient()
    credential = w.postgres.generate_database_credential(LAKEBASE_ENDPOINT)
    # Resolved from the API rather than read from the injected PGHOST: the generated hostname
    # changes whenever the endpoint is recreated (three different values in one day of building
    # this), so asking is more reliable than trusting a value captured earlier.
    endpoint = w.postgres.get_endpoint(LAKEBASE_ENDPOINT)
    connection = psycopg2.connect(
        host=endpoint.status.hosts.host,
        dbname=os.getenv("PGDATABASE", "databricks_postgres"),
        user=os.getenv("PGUSER") or Config().client_id,
        password=credential.token,
        port=int(os.getenv("PGPORT", "5432")),
        sslmode="require",
        connect_timeout=30,
        # TCP keepalives so a socket the far end has dropped is DETECTED rather than waited on.
        # Without these, a query against a connection whose endpoint suspended underneath it
        # blocks until the OS gives up, which can be minutes -- the page simply hangs.
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=3,
        # A hard ceiling on any single statement. Every query here is a point lookup or a small
        # aggregate; anything running longer than 30s is stuck, and failing is more useful than
        # a spinner that never resolves.
        options="-c statement_timeout=30000",
    )
    # psycopg2 defaults autocommit to False, which means every SELECT opens a transaction that is
    # never closed -- the connection sits "idle in transaction", holding resources for the whole
    # 40 minutes it is cached. Reads want autocommit; the one place that needs a real transaction
    # turns it off deliberately around the write.
    connection.autocommit = True
    return connection


@st.cache_data(ttl=300)
def warehouse_claim_count() -> int:
    if not CLAIMS_TABLE:
        raise RuntimeError("CLAIMS_TABLE is unset -- is the gold-claims-table resource attached?")
    with warehouse_connection().cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {CLAIMS_TABLE}")
        return cur.fetchone()[0]


@st.cache_data(ttl=300)
def portfolio_kpis() -> dict:
    """Every headline number in ONE query.

    Four separate `SELECT count(...)` calls would mean four round trips to the warehouse and four
    full scans of the same 13k rows. One pass computing six aggregates costs the same as computing
    one, and the whole row arrives together so the metrics can never disagree with each other --
    which they could if they were fetched seconds apart while the table was being written.

    The 5 minute TTL matches the data: the gold table is rebuilt by an hourly job, so anything
    shorter re-queries for numbers that cannot have moved.
    """
    if not CLAIMS_TABLE:
        raise RuntimeError("CLAIMS_TABLE is unset -- is the gold-claims-table resource attached?")
    with warehouse_connection().cursor() as cur:
        cur.execute(f"""
            SELECT
                count(*)                                                        AS total_claims,
                round(sum(claim_amount), 2)                                     AS total_exposure,
                round(avg(claim_amount), 2)                                     AS avg_claim,
                sum(CASE WHEN incident_within_coverage THEN 0 ELSE 1 END)       AS outside_coverage,
                sum(CASE WHEN claim_to_sum_insured_ratio > 1 THEN 1 ELSE 0 END) AS over_insured,
                max(incident_date)                                              AS latest_incident
            FROM {CLAIMS_TABLE}
        """)
        row = cur.fetchone()
    return {
        "total_claims": int(row[0] or 0),
        "total_exposure": float(row[1] or 0),
        "avg_claim": float(row[2] or 0),
        "outside_coverage": int(row[3] or 0),
        "over_insured": int(row[4] or 0),
        "latest_incident": row[5],
    }


# The app's OWN tables, in its OWN schema. Nothing here touches public.* -- that is the synced
# copy of the gold table, maintained by the sync pipeline and read-only to this app by grant.
# Reference data flows down from the lakehouse; transactional data is born here.
CLAIMS_SCHEMA_DDL = (
    "CREATE SCHEMA IF NOT EXISTS claims",
    """
    CREATE TABLE IF NOT EXISTS claims.submitted_claim (
        claim_number            TEXT PRIMARY KEY,
        policy_number           TEXT        NOT NULL,
        customer_name           TEXT,
        incident_date           DATE        NOT NULL,
        incident_type           TEXT,
        accident_location       TEXT,
        claim_amount            NUMERIC(12,2) NOT NULL,
        self_assessed_severity  TEXT        NOT NULL,
        predicted_severity      TEXT,
        vehicles_involved       INTEGER,
        notes                   TEXT,
        image_path              TEXT,
        submitted_by            TEXT,
        status                  TEXT        NOT NULL,
        submitted_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    # One row per check per claim, rather than four boolean columns. The admin screen shows each
    # check with its own verdict and an explanation, and a row-per-check renders without the UI
    # knowing the list up front -- adding a fifth check later becomes a backend change only.
    """
    CREATE TABLE IF NOT EXISTS claims.claim_check (
        id           SERIAL PRIMARY KEY,
        claim_number TEXT    NOT NULL REFERENCES claims.submitted_claim(claim_number)
                             ON DELETE CASCADE,
        check_name   TEXT    NOT NULL,
        passed       BOOLEAN NOT NULL,
        detail       TEXT    NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS claim_check_claim_number_idx ON claims.claim_check (claim_number)",
)


@st.cache_resource
def ensure_claims_schema() -> bool:
    """Create the app's schema and tables if they are not there. Runs once per container.

    WHO RUNS THIS FIRST DECIDES WHO OWNS IT. In Postgres the creator of a schema owns it, and the
    app's service principal only holds CAN_CONNECT_AND_CREATE -- enough to create its own objects,
    not to touch somebody else's. So if a developer runs this app locally before it has ever been
    deployed, their personal credentials own `claims`, the service principal is locked out, and
    the documented remedy is dropping the schema and losing whatever is in it.

    Deploy first. Then local runs are harmless, because the schema already exists and IF NOT
    EXISTS makes this a no-op.

    @st.cache_resource, not cache_data: this is an effect executed once, not a value to reuse.
    """
    with live_lakebase_connection().cursor() as cur:
        for statement in CLAIMS_SCHEMA_DDL:
            cur.execute(statement)
    return True


@st.cache_data(ttl=30)
def list_submitted_claims(status: str | None = None) -> "pd.DataFrame":
    """Claims this app has taken in, newest first, with a count of failed checks.

    Returns an EMPTY frame when claims.submitted_claim does not exist. That is not an error
    state: the schema is created by the first submission, so an admin opening this screen before
    anyone has filed a claim is the normal first-run experience, and an "undefined table"
    traceback would be a poor way to say "no claims yet".

    Catching that cleanly depends on the connection being in autocommit -- a failed statement
    inside an open transaction would poison the connection for every subsequent query on it.

    Short TTL because this is operational rather than analytical: a handler wants to see what
    arrived a minute ago, not a five-minute-old snapshot.
    """
    from psycopg2 import errors

    sql = """
        SELECT c.claim_number, c.policy_number, c.customer_name, c.incident_date,
               c.incident_type, c.claim_amount, c.self_assessed_severity, c.status,
               c.submitted_by, c.submitted_at,
               count(*) FILTER (WHERE NOT k.passed) AS failed_checks
        FROM claims.submitted_claim c
        LEFT JOIN claims.claim_check k ON k.claim_number = c.claim_number
        WHERE (%s::text IS NULL OR c.status = %s)
        GROUP BY c.claim_number
        ORDER BY c.submitted_at DESC
        LIMIT 200
    """
    columns = ["claim_number", "policy_number", "customer_name", "incident_date", "incident_type",
               "claim_amount", "self_assessed_severity", "status", "submitted_by", "submitted_at",
               "failed_checks"]
    try:
        with live_lakebase_connection().cursor() as cur:
            cur.execute(sql, (status, status))
            rows = cur.fetchall()
    except errors.UndefinedTable:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


CLAIM_DETAIL_COLUMNS = (
    "claim_number", "policy_number", "customer_name", "incident_date", "incident_type",
    "accident_location", "claim_amount", "self_assessed_severity", "predicted_severity",
    "vehicles_involved", "notes", "image_path", "submitted_by", "status", "submitted_at",
)


@st.cache_data(ttl=30)
def fetch_claim(claim_number: str) -> dict | None:
    """One submitted claim, whole -- every column, not the queue's subset.

    The queue deliberately selects a narrow set of columns for a table that has to stay readable
    at a glance. This is the other half: accident_location, vehicles_involved, notes, image_path
    and predicted_severity, none of which belong in a list view but all of which a handler needs
    before deciding anything.

    Returns None when the claim number does not exist, which is reachable in normal use: the
    queue is cached for 30s, so a row can be selected slightly after someone else deleted it.
    """
    sql = f"SELECT {', '.join(CLAIM_DETAIL_COLUMNS)} FROM claims.submitted_claim WHERE claim_number = %s"
    with live_lakebase_connection().cursor() as cur:
        cur.execute(sql, (claim_number,))
        row = cur.fetchone()
    if row is None:
        return None
    claim = dict(zip(CLAIM_DETAIL_COLUMNS, row))
    # NUMERIC arrives as Decimal, which formats badly and compares badly against the floats the
    # rest of the app uses. lookup_policy does the same conversion for the same reason.
    claim["claim_amount"] = float(claim["claim_amount"]) if claim["claim_amount"] is not None else None
    return claim


@st.cache_data(ttl=30)
def fetch_claim_checks(claim_number: str) -> list[dict]:
    """The verdicts for one claim, in the order the checks were run.

    ORDER BY id, not by name: id is a SERIAL and submit_claim inserts the checks in the order
    run_claim_checks produced them, so the sequence the transcript presents survives into the
    admin screen for free. Sorting by check_name would scramble it alphabetically.

    A list of dicts rather than a DataFrame because the caller renders one block per check rather
    than a table -- the detail text is a sentence, not a cell.

    Returns [] for an unknown claim, which is indistinguishable here from a claim whose checks
    were never written. The caller has fetch_claim to tell those apart.
    """
    sql = """
        SELECT check_name, passed, detail
        FROM claims.claim_check
        WHERE claim_number = %s
        ORDER BY id
    """
    with live_lakebase_connection().cursor() as cur:
        cur.execute(sql, (claim_number,))
        rows = cur.fetchall()
    return [{"check_name": name, "passed": passed, "detail": detail} for name, passed, detail in rows]


SPEED_LIMIT_KPH = 130


def run_claim_checks(policy: dict, claim: dict) -> list[dict]:
    """The four automated checks, in the order the transcript presents them.

    A pure function -- policy in, claim in, verdicts out. No database, no Streamlit. That is
    deliberate: this is the only part of the submission with real business logic, and keeping it
    free of I/O is what makes it readable and testable.

    Every check returns a `detail` even when it passes, because the admin screen has to answer
    "why was this auto-approved" as often as "why was this held".
    """
    checks: list[dict] = []

    # 1. The damage model against the customer's own assessment. Not wired up yet, so there is
    #    nothing to disagree with -- reported as passed-with-caveat rather than failed. Failing a
    #    claim for a check that cannot run would hold every claim for a reason nobody can fix.
    predicted = claim.get("predicted_severity")
    if not predicted:
        checks.append({
            "check_name": "Severity match",
            "passed": True,
            "detail": (
                f"No model prediction available; customer assessed "
                f"\"{claim['self_assessed_severity']}\"."
            ),
        })
    else:
        agrees = predicted == claim["self_assessed_severity"]
        checks.append({
            "check_name": "Severity match",
            "passed": agrees,
            "detail": (
                f"Model and customer agree on \"{predicted}\"." if agrees
                else f"Customer assessed \"{claim['self_assessed_severity']}\" but the model "
                     f"predicts \"{predicted}\"."
            ),
        })

    # 2. Claimed amount against the policy ceiling.
    sum_insured = policy.get("sum_insured")
    if sum_insured is None:
        checks.append({"check_name": "Policy amount", "passed": False,
                       "detail": "The policy has no sum insured recorded."})
    else:
        within = claim["claim_amount"] <= sum_insured
        checks.append({
            "check_name": "Policy amount",
            "passed": within,
            "detail": (
                f"${claim['claim_amount']:,.2f} is within the ${sum_insured:,.2f} sum insured."
                if within else
                f"${claim['claim_amount']:,.2f} exceeds the ${sum_insured:,.2f} sum insured."
            ),
        })

    # 3. Was the policy actually in force on the day.
    effective, expiry = policy.get("effective"), policy.get("expiry")
    if not effective or not expiry:
        checks.append({"check_name": "Policy validity", "passed": False,
                       "detail": "The policy's coverage dates are incomplete."})
    else:
        in_force = effective <= claim["incident_date"] <= expiry
        checks.append({
            "check_name": "Policy validity",
            "passed": in_force,
            "detail": (
                f"The incident falls inside the cover period ({effective} to {expiry})."
                if in_force else
                f"The incident on {claim['incident_date']} falls outside the cover period "
                f"({effective} to {expiry})."
            ),
        })

    # 4. Telematics. Only 8 of ~13,000 policies carry a device, so "no data" is overwhelmingly the
    #    normal case and must not fail the claim -- otherwise almost every claim is held for a
    #    reason the customer can do nothing about.
    if not policy.get("has_telematics") or policy.get("max_speed") is None:
        checks.append({"check_name": "Speed check", "passed": True,
                       "detail": "No telematics device on this vehicle; speed was not checked."})
    else:
        within = policy["max_speed"] <= SPEED_LIMIT_KPH
        checks.append({
            "check_name": "Speed check",
            "passed": within,
            "detail": (
                f"Peak recorded speed {policy['max_speed']:.0f} km/h is within the "
                f"{SPEED_LIMIT_KPH} km/h threshold." if within else
                f"Peak recorded speed {policy['max_speed']:.0f} km/h exceeds the "
                f"{SPEED_LIMIT_KPH} km/h threshold."
            ),
        })

    return checks


def submit_claim(policy: dict, claim: dict) -> tuple[str, str, list[dict]]:
    """Run the checks, write the claim and its verdicts, return (claim_number, status, checks).

    The claim and its checks go in ONE transaction. A claim stored without its checks would show
    in the review queue with no reason attached, and there would be no way to tell that from a
    claim nobody has assessed yet.
    """
    from datetime import datetime

    ensure_claims_schema()
    checks = run_claim_checks(policy, claim)
    # Any failed check sends it to a human -- the transcript's "claims that do not pass all of
    # those out of the box checks" become reviewable.
    status = "Approved" if all(c["passed"] for c in checks) else "Under Review"
    claim_number = f"WEB-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

    connection = live_lakebase_connection()
    connection.autocommit = False  # one transaction for the claim and its checks
    try:
        with connection.cursor() as cur:
            cur.execute(
                """
                INSERT INTO claims.submitted_claim (
                    claim_number, policy_number, customer_name, incident_date, incident_type,
                    accident_location, claim_amount, self_assessed_severity, predicted_severity,
                    vehicles_involved, notes, image_path, submitted_by, status
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    claim_number, claim["policy_number"], claim["customer_name"],
                    claim["incident_date"], claim["incident_type"], claim["accident_location"],
                    claim["claim_amount"], claim["self_assessed_severity"],
                    claim.get("predicted_severity"), claim["vehicles_involved"],
                    claim["notes"], claim.get("image_path"), claim["submitted_by"], status,
                ),
            )
            for check in checks:
                cur.execute(
                    """INSERT INTO claims.claim_check (claim_number, check_name, passed, detail)
                       VALUES (%s,%s,%s,%s)""",
                    (claim_number, check["check_name"], check["passed"], check["detail"]),
                )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.autocommit = True

    return claim_number, status, checks


def md_escape(text: str) -> str:
    """Escape markdown that would otherwise be interpreted rather than shown.

    A pair of dollar signs in one string makes Streamlit render everything between them as LaTeX,
    so "$10,000.00 is within the $85,718.46 sum insured" came out as maths italics. The check
    details are generated with currency in them and are stored in Postgres, so escaping happens
    at DISPLAY time -- the data stays clean for anything else that reads it.
    """
    return text.replace("$", r"\$")


def claims_volume_path() -> str:
    """The volume as a /Volumes/... path, whichever form CLAIMS_VOLUME arrives in.

    A `uc_securable` valueFrom hands over the securable's full name, and for a VOLUME that is the
    dotted three-part name -- not a filesystem path. The Files API wants a path. Rather than
    assume which form turns up, this accepts either: a leading slash is taken as already a path,
    anything else is treated as catalog.schema.volume and converted.
    """
    raw = os.getenv("CLAIMS_VOLUME", "").strip()
    if not raw:
        raise RuntimeError("CLAIMS_VOLUME is unset -- is the claims-volume resource attached?")
    if raw.startswith("/"):
        return raw.rstrip("/")
    return "/Volumes/" + raw.replace(".", "/")


def upload_claim_photo(uploaded, policy_number: str) -> str:
    """Put the customer's photo in the volume and return the path it landed at.

    Uploaded here rather than held until submission, for the reason the transcript's flow implies:
    the damage model scores the image as soon as it arrives, and the customer sees the verdict
    before filling in the rest. The cost is that abandoning the form leaves an orphaned file --
    acceptable, and cheaper than holding megabytes in session state across every re-run.

    The name carries the policy and a timestamp: two claims on one policy must not collide, and a
    filename taken from the upload would let a customer choose where they write.
    """
    from datetime import datetime

    suffix = Path(uploaded.name).suffix.lower() or ".png"
    if suffix not in {".png", ".jpg", ".jpeg"}:
        raise ValueError(f"Unsupported image type '{suffix}'. Use PNG or JPEG.")
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    path = f"{claims_volume_path()}/images/{policy_number}_{stamp}{suffix}"
    WorkspaceClient().files.upload(path, io.BytesIO(uploaded.getvalue()), overwrite=True)
    return path


# The categories the gold table actually uses, so a submitted claim is comparable with the
# portfolio rather than introducing values nothing else knows about. Hardcoded rather than queried:
# six values that do not change are not worth a warehouse round trip on every form render -- but
# they DO have to stay in step with the source, so they are named here where that is visible.
INCIDENT_TYPES = ["Collision", "Fire", "Flood", "Hail", "Theft", "Vandalism"]
SEVERITIES = ["Minor Damage", "Major Damage", "Total Loss"]


# One row per policy, newest claim first. The synced table holds one row per CLAIM, so a policy
# with several historic claims appears several times; DISTINCT ON collapses that. The policy terms
# are identical across those rows, so any one of them answers "what does this policy cover".
#
# `%s` IS A PLACEHOLDER, NOT STRING FORMATTING. The policy number comes from a text box, so it is
# the one value in this app that an outsider controls. psycopg2 sends it separately from the SQL,
# which is what makes `POL-1'; DROP TABLE ...` a policy number that simply does not match rather
# than a statement. The TABLE name is interpolated because it comes from bundle config, not a user.
POLICY_LOOKUP_SQL = """
    SELECT DISTINCT ON (policy_number)
           policy_number, customer_name, sum_insured, deductible,
           policy_effective_date, policy_expiry_date, has_telematics, max_speed
    FROM public.customer_claim_policy_telematics
    WHERE policy_number = %s
    ORDER BY policy_number, incident_date DESC
"""


@st.cache_data(ttl=120)
def lookup_policy(policy_number: str) -> dict | None:
    """Fetch one policy from Lakebase, or None if there is no such policy.

    This is the reason Lakebase exists in this app. The same question could be asked of the gold
    table through the SQL warehouse, but that is analytical compute: seconds, while a customer
    waits mid-form. Here it is an indexed point lookup in Postgres -- the index added for exactly
    this query, since the table's primary key is claim_number and this searches by policy_number.

    Cached per policy number, briefly: a customer filling in a form re-triggers a script re-run on
    every keystroke elsewhere on the page, and the lookup should not be repeated each time.
    """
    with live_lakebase_connection().cursor() as cur:
        cur.execute(POLICY_LOOKUP_SQL, (policy_number.strip(),))
        row = cur.fetchone()
    if row is None:
        return None
    return {
        "policy_number": row[0],
        "customer_name": row[1],
        "sum_insured": float(row[2]) if row[2] is not None else None,
        "deductible": float(row[3]) if row[3] is not None else None,
        "effective": row[4],
        "expiry": row[5],
        "has_telematics": bool(row[6]),
        "max_speed": float(row[7]) if row[7] is not None else None,
    }


def live_lakebase_connection():
    """The cached Lakebase connection, verified usable -- rebuilt if it is not.

    The connection is cached for 40 minutes, but the endpoint underneath it can suspend, restart
    or scale to zero in that window, leaving a socket that is open locally and dead remotely. A
    query on one of those does not raise; it BLOCKS. The Lakebase documentation says to implement
    retry logic for exactly this, and this is that.

    A `SELECT 1` costs a round trip and turns a hang into a reconnect.
    """
    connection = lakebase_connection()
    try:
        with connection.cursor() as cur:
            cur.execute("SELECT 1")
        return connection
    except Exception:  # noqa: BLE001 -- any failure here means "this connection is no good"
        # Drop the cached object so the next call builds a fresh one, with a fresh OAuth token.
        lakebase_connection.clear()
        try:
            connection.close()
        except Exception:  # noqa: BLE001 -- already broken; closing is best effort
            pass
        return lakebase_connection()


@st.cache_data(ttl=300)
def monthly_exposure(months: int = 24) -> "pd.DataFrame":
    """Claim exposure per month, EXCLUDING the month the data ends in.

    That exclusion is the whole point. The source stops on 2025-10-01, so the final month holds a
    single day -- 249,828 against a ~7,000,000 run rate. Plotted, it reads as a 97% collapse in
    the book rather than as "the data stops here", and a reader has no way to tell the difference
    from the chart alone. Dropping the incomplete period is more honest than drawing it.
    """
    if not CLAIMS_TABLE:
        raise RuntimeError("CLAIMS_TABLE is unset -- is the gold-claims-table resource attached?")
    with warehouse_connection().cursor() as cur:
        cur.execute(f"""
            WITH m AS (
                SELECT date_trunc('MONTH', incident_date) AS month,
                       sum(claim_amount)                  AS exposure,
                       count(*)                           AS claims
                FROM {CLAIMS_TABLE}
                GROUP BY 1
            )
            SELECT date_format(month, 'yyyy-MM') AS month, exposure, claims
            FROM m
            WHERE month < (SELECT max(month) FROM m)
            ORDER BY month DESC
            LIMIT {int(months)}
        """)
        rows = cur.fetchall()
    frame = pd.DataFrame(rows, columns=["month", "exposure", "claims"])
    # Fetched newest-first so LIMIT takes the RECENT months, then flipped for the chart -- a time
    # axis has to run left to right.
    return frame.iloc[::-1].reset_index(drop=True)


@st.cache_data(ttl=300)
def breakdown(column: str) -> "pd.DataFrame":
    """Claim count and exposure grouped by one categorical column.

    `column` is interpolated into the SQL, so it must never come from user input. The callers
    below pass literals; if a filter ever feeds this, it needs an allow-list first.
    """
    with warehouse_connection().cursor() as cur:
        cur.execute(f"""
            SELECT {column} AS category, count(*) AS claims, sum(claim_amount) AS exposure
            FROM {CLAIMS_TABLE}
            GROUP BY {column}
            ORDER BY claims DESC
        """)
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["category", "claims", "exposure"])


@st.cache_data(ttl=60)
def lakebase_claim_count() -> int:
    # A fresh cursor per call, but the CONNECTION is cached -- opening a Postgres connection per
    # query is what exhausts the pool under any real traffic.
    with live_lakebase_connection().cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {PG_CLAIMS_TABLE}")
        return cur.fetchone()[0]

# --- assistant ----------------------------------------------------------------------------------
@st.cache_resource
def openai_client() -> OpenAI:
    """One client for the process. Cached for the same reason the database connections are --
    Streamlit re-runs this whole file on every interaction, and rebuilding an HTTP client per
    keystroke is waste. No ttl: unlike the Lakebase token, an API key does not expire on a timer.
    """
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is unset -- is the openai-key resource attached?")
    return OpenAI(api_key=key)


# Constrains the model rather than trusting it. The app sits on real claims data that the model
# CANNOT see, so the failure mode to design against is a confident invented claim number. Telling
# it to refuse and point at Genie is cheaper than discovering a hallucinated figure in a demo.
SYSTEM_PROMPT = (
    "You are an assistant inside a motor insurance claims application. "
    "Help with general insurance questions, explaining terms, and drafting claim descriptions. "
    "You have NO access to this system's claims, policies or customers. "
    "If asked about specific claims, numbers, totals or customers, say you cannot see the data "
    "and suggest the Genie space or the admin screens instead. Never invent claim data."
)

# How many prior turns to resend. Every turn is re-sent in full on each request -- the API is
# stateless, so "memory" is just the transcript travelling with the question. That means cost
# grows with conversation length, and an unbounded history eventually hits the context limit
# mid-conversation. Trimming to the last few exchanges keeps both bounded. The system prompt is
# always prepended and never counted here.
MAX_HISTORY_TURNS = 12


# --- screens ------------------------------------------------------------------------------------
# One function per tab. Streamlit re-runs this whole file on every interaction, so these are called
# fresh each time -- they hold no state beyond st.session_state. Splitting them out now, while
# there are only three, keeps the file navigable once the claim submission flow and the admin
# screens land.
def render_customer() -> None:
    st.subheader("Submit a claim")
    st.caption("Start by finding your policy.")

    policy_number = st.text_input(
        "Policy number",
        placeholder="POL-000000",
        help="Printed on your policy documents.",
    ).strip()

    if not policy_number:
        st.info("Enter a policy number to begin.")
        return

    try:
        policy = lookup_policy(policy_number)
    except Exception as err:  # noqa: BLE001 -- the page is the only place errors are visible
        st.error(f"Could not look up the policy: {err}")
        return

    if policy is None:
        # Deliberately says the policy was not found and nothing else. Confirming which part of an
        # identifier was wrong would help someone guessing at policy numbers more than it helps a
        # customer holding their documents.
        st.warning(f"No policy found for **{policy_number}**. Check the number and try again.")
        return

    st.success(f"Policy **{policy['policy_number']}** — {policy['customer_name']}")

    a, b, c = st.columns(3)
    a.metric("Sum insured", f"${policy['sum_insured']:,.0f}" if policy["sum_insured"] else "—")
    b.metric("Deductible", f"${policy['deductible']:,.0f}" if policy["deductible"] else "—")
    c.metric("Cover ends", str(policy["expiry"]) if policy["expiry"] else "—")

    st.caption(
        f"In force {policy['effective']} to {policy['expiry']} · "
        + (
            f"telematics fitted, peak recorded speed {policy['max_speed']:.0f} km/h"
            if policy["has_telematics"] and policy["max_speed"] is not None
            else "no telematics device on this vehicle"
        )
    )

    # Stated now rather than sprung at submission. The dataset's incidents run to October 2025 and
    # most policies have lapsed against today's date, so a customer whose cover has ended should
    # see that before filling in a form that will be held for review because of it.
    from datetime import date

    if policy["expiry"] and policy["expiry"] < date.today():
        st.warning(
            f"This policy expired on {policy['expiry']}. A claim can still be submitted, but the "
            "coverage check will fail and it will be held for review."
        )

    st.divider()
    _render_photo_upload(policy)
    st.divider()
    _render_claim_form(policy)


def _render_photo_upload(policy: dict) -> None:
    """Accident photo, uploaded to the Unity Catalog volume."""
    st.markdown("**Photo of the damage**")

    uploaded = st.file_uploader(
        "Upload a photo", type=["png", "jpg", "jpeg"], label_visibility="collapsed",
        help="A clear photo of the damage. PNG or JPEG.",
    )

    if uploaded is None:
        # Not a blocker. Three of the four checks need no image, so a claim without a photo is
        # held for review rather than refused.
        st.caption("Optional, but a claim without a photo cannot have its severity verified.")
        st.session_state.pop("claim_photo_path", None)
        return

    left, right = st.columns([1, 2])
    with left:
        # NOT use_container_width: st.image only accepts that from Streamlit 1.39, and the Databricks
        # Apps runtime ships 1.38.0. width= has been valid throughout.
        st.image(uploaded, caption=uploaded.name, width=280)

    with right:
        # Re-uploading the same file on every script re-run would mean a fresh copy in the volume
        # per keystroke elsewhere on the page. Keyed on name+size so choosing a DIFFERENT file
        # does upload again.
        marker = f"{uploaded.name}:{uploaded.size}"
        if st.session_state.get("claim_photo_marker") != marker:
            try:
                with st.spinner("Saving the photo..."):
                    path = upload_claim_photo(uploaded, policy["policy_number"])
                st.session_state.claim_photo_path = path
                st.session_state.claim_photo_marker = marker
            except Exception as err:  # noqa: BLE001
                st.error(f"Could not save the photo: {err}")
                st.session_state.pop("claim_photo_path", None)
                return

        st.success("Photo saved.")
        st.caption(f"`{st.session_state.claim_photo_path}`")
        st.caption(f"{uploaded.size / 1024:,.0f} KB")
        st.info(
            "The damage model is not wired up yet. Once it is, the severity it predicts from this "
            "photo is compared against your own assessment below."
        )


def _render_claim_form(policy: dict) -> None:
    """The claim details and the submission, in ONE step.

    An earlier version split this into "Review claim" then "Submit claim". That needed the
    reviewed claim to survive between two separate script runs in st.session_state -- and when a
    session was lost (a dropped websocket, a restarted container), the summary and the submit
    button disappeared and the click went nowhere. The transcript has no review step either:
    "we can put some additional notes and now I can basically submit".

    One button, one script run: validate, run the checks, write, show the verdict. Nothing has to
    survive a re-run, so nothing can be lost between them.
    """
    from datetime import date

    st.markdown("**Claim details**")

    with st.form("claim_details"):
        left, right = st.columns(2)
        with left:
            incident_date = st.date_input(
                "Date of the incident", value=date.today(),
                # A claim cannot be made for something that has not happened. The widget refusing
                # is clearer than a validation message after the fact.
                max_value=date.today(),
            )
            incident_type = st.selectbox("What happened?", INCIDENT_TYPES)
            claim_amount = st.number_input(
                "Amount claimed (USD)", min_value=1.0, max_value=1_000_000.0,
                value=10_000.0, step=500.0, format="%.2f",
            )
        with right:
            severity = st.selectbox(
                "How bad is the damage?", SEVERITIES,
                help="Your own assessment. The damage model's opinion is compared against it.",
            )
            location = st.text_input("Where did it happen?", placeholder="City or address")
            vehicles = st.number_input("Vehicles involved", min_value=1, max_value=20, value=1)

        notes = st.text_area("Anything else we should know?", placeholder="Optional")
        st.caption(
            "Submitting runs four checks: your severity assessment against the damage model, the "
            "amount against your sum insured, the incident date against your coverage window, and "
            "recorded speed where a telematics device is fitted."
        )
        submitted = st.form_submit_button("Submit claim", type="primary")

    if not submitted:
        return

    # Validation the widgets cannot express. The amount is bounded for sanity only -- whether the
    # policy actually covers it is one of the four checks below, and is deliberately not
    # pre-judged here.
    problems = []
    if not location.strip():
        problems.append("Tell us where the incident happened.")
    if incident_date < policy["effective"]:
        problems.append(f"The incident date is before this policy began ({policy['effective']}).")
    if problems:
        for problem in problems:
            st.error(problem)
        return

    claim = {
        "policy_number": policy["policy_number"],
        "customer_name": policy["customer_name"],
        "incident_date": incident_date,
        "incident_type": incident_type,
        "claim_amount": float(claim_amount),
        "self_assessed_severity": severity,
        "accident_location": location.strip(),
        "vehicles_involved": int(vehicles),
        "notes": notes.strip() or None,
        "submitted_by": current_user(),
        "image_path": st.session_state.get("claim_photo_path"),
        "predicted_severity": None,
    }

    try:
        with st.spinner("Running checks and submitting..."):
            claim_number, status, checks = submit_claim(policy, claim)
    except Exception as err:  # noqa: BLE001 -- the page is the only place errors are visible
        st.error(f"Could not submit the claim: {err}")
        return

    st.session_state.pop("claim_photo_path", None)
    st.session_state.pop("claim_photo_marker", None)

    if status == "Approved":
        st.success(f"Claim **{claim_number}** approved.")
        st.balloons()
        st.write("You will receive your settlement within 3 to 5 business days.")
    else:
        # Not phrased as a rejection. A held claim is one a person will look at, and the checks
        # below say exactly why -- a customer who reads "declined" when they mean "queued" will
        # call, which helps nobody.
        st.warning(f"Claim **{claim_number}** has been submitted and is under review.")
        st.write("One or more checks did not pass. A claims handler will look at it.")

    st.markdown("**Check results**")
    for check in checks:
        icon = "✅" if check["passed"] else "⚠️"
        st.markdown(f"{icon} **{check['check_name']}** — {md_escape(check['detail'])}")


def render_admin() -> None:
    st.subheader("Portfolio overview")

    # A cold serverless warehouse takes ~17 seconds to answer its first query. Without a spinner
    # the page just sits there and looks broken.
    try:
        with st.spinner("Querying the gold layer..."):
            kpis = portfolio_kpis()
    except Exception as err:  # noqa: BLE001 -- the page is the only place errors are visible
        st.error(f"Could not load portfolio figures: {err}")
        return

    if kpis["total_claims"] == 0:
        st.warning("No claims in the gold table. Has the transformations pipeline run?")
        return

    total = kpis["total_claims"]
    a, b, c, d = st.columns(4)
    a.metric("Claims", f"{total:,}")
    b.metric("Exposure", f"${kpis['total_exposure']:,.0f}")
    # Percentages alongside the counts: "10,556 claims" means little on its own, "81% of the book"
    # is the number someone would act on.
    c.metric(
        "Outside coverage",
        f"{kpis['outside_coverage'] / total:.0%}",
        delta=f"{kpis['outside_coverage']:,} claims",
        delta_color="off",
    )
    d.metric(
        "Over sum insured",
        f"{kpis['over_insured'] / total:.0%}",
        delta=f"{kpis['over_insured']:,} claims",
        delta_color="off",
    )

    # Freshness belongs next to the numbers. A KPI with no as-of date invites the assumption that
    # it is current, and this table is rebuilt hourly from a source that stops in October 2025.
    st.caption(
        f"Average claim ${kpis['avg_claim']:,.0f} · most recent incident "
        f"{kpis['latest_incident']} · figures cached for 5 minutes"
    )

    st.divider()

    # A CHART ONLY WHERE THERE IS SOMETHING TO SEE. Exposure moves month to month (6.8M-8.0M), so
    # a trend line earns its space. The categorical splits do NOT: severity lands 4,346/4,330/4,320
    # and status within 2.4% across five values, because the generator assigns them uniformly.
    # Three bar charts of identical bars would imply a pattern that is not there. Those go in a
    # table, where "these are all the same" is legible instead of disguised.
    st.markdown("**Monthly exposure**")
    try:
        trend = monthly_exposure()
        if trend.empty:
            st.info("No complete months of history yet.")
        else:
            st.line_chart(trend, x="month", y="exposure", height=260)
            st.caption(
                f"{len(trend)} complete months to {trend['month'].iloc[-1]}. The month the data "
                "ends in is excluded -- it holds a single day and would read as a collapse."
            )
    except Exception as err:  # noqa: BLE001
        st.error(f"Could not load the trend: {err}")

    st.divider()
    st.markdown("**Breakdowns**")
    left, right = st.columns(2)
    for col, (column, label) in zip(
        (left, right), (("incident_severity", "By severity"), ("claim_status", "By status"))
    ):
        with col:
            st.markdown(f"*{label}*")
            try:
                frame = breakdown(column)
                st.dataframe(
                    frame,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "category": st.column_config.TextColumn(label.replace("By ", "").title()),
                        "claims": st.column_config.NumberColumn("Claims", format="%d"),
                        "exposure": st.column_config.NumberColumn("Exposure", format="$%.0f"),
                    },
                )
            except Exception as err:  # noqa: BLE001
                st.error(f"Could not load {label.lower()}: {err}")

    st.caption(
        "These categories are near-uniform in this dataset -- the generator assigns them evenly, "
        "so the flatness is a property of the synthetic data, not a finding about claims."
    )

    st.divider()
    _render_review_queue()


def _render_review_queue() -> None:
    """Claims submitted through this app -- NOT the 12,996 above.

    Worth keeping straight: everything higher on this screen aggregates the gold table, which is
    the historical book from the lakehouse. This reads claims.submitted_claim in Lakebase, which
    holds only what this app has taken in. They are different populations and will never agree.
    """
    st.markdown("**Review queue**")
    st.caption(
        "Claims submitted through this app. Separate from the portfolio figures above, which "
        "come from the gold layer."
    )

    left, right = st.columns([3, 1])
    with left:
        choice = st.radio(
            "Show", ["All", "Under Review", "Approved"],
            horizontal=True, label_visibility="collapsed",
        )
    with right:
        if st.button("Refresh", use_container_width=True):
            # The 30s cache is right for normal use and wrong when a handler has just asked
            # someone to resubmit and wants to see it land.
            list_submitted_claims.clear()
            st.rerun()

    try:
        frame = list_submitted_claims(None if choice == "All" else choice)
    except Exception as err:  # noqa: BLE001
        st.error(f"Could not load the review queue: {err}")
        return

    if frame.empty:
        st.info(
            "No claims submitted yet. The customer tab files one, and the `claims` schema is "
            "created by that first submission."
            if choice == "All" else f"No claims with status “{choice}”."
        )
        return

    held = int((frame["status"] == "Under Review").sum())
    a, b, c = st.columns(3)
    a.metric("Submitted", f"{len(frame):,}")
    b.metric("Held for review", f"{held:,}")
    c.metric("Auto-approved", f"{len(frame) - held:,}")

    st.dataframe(
        frame,
        hide_index=True,
        use_container_width=True,
        column_config={
            "claim_number": st.column_config.TextColumn("Claim"),
            "policy_number": st.column_config.TextColumn("Policy"),
            "customer_name": st.column_config.TextColumn("Policy holder"),
            "incident_date": st.column_config.DateColumn("Incident"),
            "incident_type": st.column_config.TextColumn("Type"),
            "claim_amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
            "self_assessed_severity": st.column_config.TextColumn("Self-assessed"),
            "status": st.column_config.TextColumn("Status"),
            "submitted_by": st.column_config.TextColumn("Filed by"),
            "submitted_at": st.column_config.DatetimeColumn("Submitted"),
            "failed_checks": st.column_config.NumberColumn("Failed checks", format="%d"),
        },
    )
    st.caption("Per-claim detail -- the four verdicts and the photo -- is the next step.")


def render_assistant() -> None:
    st.caption(f"Model `{OPENAI_MODEL}` · no access to claims data — see the note below")


    # st.session_state survives the script re-running, which ordinary variables do not: Streamlit
    # executes this file top to bottom on every interaction, so a plain list would be empty again by
    # the time the next message arrived. It is per browser session -- two people using the app have
    # separate conversations, and a refresh starts over.
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Replay the conversation. Necessary, not decorative: the re-run wipes the rendered page, so
    # without this only the newest message would be visible.
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input("Ask about insurance terms, or draft a claim description...")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            try:
                stream = openai_client().chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=(
                        [{"role": "system", "content": SYSTEM_PROMPT}]
                        + st.session_state.chat_history[-MAX_HISTORY_TURNS:]
                    ),
                    stream=True,
                    # Without this, a streamed response carries NO usage at all and the token count
                    # below is always None -- verified against the API, not assumed.
                    stream_options={"include_usage": True},
                )

                usage = {}

                def tokens(chunks) -> str:
                    """Yield text as it arrives; capture usage from the final chunk on the way past.

                    st.write_stream renders each yielded string immediately, which is what makes the
                    answer appear progressively instead of after a long pause. Usage arrives in a
                    LAST chunk that carries no content, hence collecting it as a side effect here
                    rather than returning it.
                    """
                    for chunk in chunks:
                        if getattr(chunk, "usage", None):
                            usage["total"] = chunk.usage.total_tokens
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content

                answer = st.write_stream(tokens(stream))
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                if usage.get("total"):
                    st.caption(f"{usage['total']} tokens")
            except Exception as err:  # noqa: BLE001 -- the page is the only place errors are visible
                st.error(f"Chat failed: {err}")
                # Drop the unanswered question, or it is re-sent with every later message and the
                # model keeps seeing a turn it never replied to.
                st.session_state.chat_history.pop()

    if st.session_state.chat_history and st.button("Clear conversation"):
        st.session_state.chat_history = []
        st.rerun()

    st.info(
        "This assistant answers from general knowledge only. It is deliberately cut off from the "
        "claims data: a language model asked for a total would produce a plausible number rather "
        "than a true one. For real figures use the Genie space or the admin views."
    )


def render_diagnostics() -> None:
    """The P5 connectivity check, kept but demoted.

    It proves both backends are reachable, which was the whole app a few steps ago. A claims
    handler never needs it, so it moves out of the main flow rather than being deleted -- when a
    screen misbehaves, "can the app reach its databases at all" is the first question worth
    answering.

    NOTE: no st.expander in here. This function is called from inside one, and Streamlit raises
    StreamlitAPIException on a nested expander.
    """
    st.write(
        "Both backends should report the same number. They are the same data: the Lakebase row "
        "is a snapshot copy of the gold table, synced once. A mismatch means the snapshot has "
        "gone stale relative to gold -- expected behaviour for SNAPSHOT sync, not a bug."
    )
    left, right = st.columns(2)
    with left:
        st.markdown("**SQL warehouse** — `gold` (Delta)")
        try:
            st.metric("Claims", f"{warehouse_claim_count():,}")
        except Exception as err:  # noqa: BLE001 -- surface the real cause on the page
            st.error(f"Warehouse query failed: {err}")
    with right:
        st.markdown("**Lakebase** — `public` (Postgres)")
        try:
            st.metric("Claims", f"{lakebase_claim_count():,}")
        except Exception as err:  # noqa: BLE001
            st.error(f"Lakebase query failed: {err}")

    st.divider()
    render_runtime_environment()


def render_runtime_environment() -> None:
    """Just the environment panel -- no queries, safe to render on every run."""
    st.markdown("**Runtime environment**")
    st.caption(
        "What the container actually received. These come from `valueFrom` bindings in "
        "src/app/app.yaml; an `(unset)` means a binding did not resolve, which is otherwise "
        "invisible until something tries to use it."
    )
    st.write(
        {
            "CLAIMS_TABLE": CLAIMS_TABLE or "(unset)",
            "PG_CLAIMS_TABLE": PG_CLAIMS_TABLE or "(unset)",
            "DATABRICKS_WAREHOUSE_ID": os.getenv("DATABRICKS_WAREHOUSE_ID", "(unset)"),
            "LAKEBASE_ENDPOINT": LAKEBASE_ENDPOINT or "(unset)",
            "CLAIMS_VOLUME": os.getenv("CLAIMS_VOLUME", "(unset)"),
            "PGUSER": os.getenv("PGUSER", "(unset -- falling back to SP client id)"),
            "PGHOST_injected": os.getenv("PGHOST", "(unset)"),
            "streamlit": st.__version__,
            "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "(unset)"),
            # PRESENCE AND LENGTH ONLY -- never the value. This expander renders on a page any
            # app viewer can open, so printing the key here would leak it to everyone with access.
            # Length is enough to tell "the binding worked" from "the binding is empty".
            "OPENAI_API_KEY": (
                f"set ({len(os.environ['OPENAI_API_KEY'])} chars)"
                if os.environ.get("OPENAI_API_KEY") else "(unset)"
            ),
        }
    )
    # Printed so the real forwarded-header names can be confirmed from the deployed app instead
    # of guessed. Remove once current_user() is known to pick the right one.
    try:
        st.write({"forwarded_headers": {k: v for k, v in (st.context.headers or {}).items()
                                        if k.lower().startswith("x-forwarded")}})
    except Exception:
        st.write("headers unavailable (local run)")


# --- page ---------------------------------------------------------------------------------------
st.title("🚗 Smart Claims")
st.caption(f"Signed in as **{current_user()}** · queries run as the app's service principal")

# The transcript's two modes: "it has the customer mode which we are in currently as well as the
# admin mode". A workflow switch, NOT a security boundary -- every query runs as the app's service
# principal whichever screen is open, and nothing here checks who you are.
#
# A RADIO, NOT st.tabs, AND THE REASON MATTERS. st.tabs is client-side: Streamlit executes the
# content of EVERY tab on EVERY script run, then the browser shows one. With three tabs that meant
# each keystroke-triggered re-run fired the admin KPI query, the trend query, two breakdowns and
# the review queue -- while the user was on the customer screen. Cold-start a suspended warehouse
# in the middle of that and the page blocks for ~17 seconds and looks like it has lost its
# connection. A radio picks one branch, so only that screen's queries run.
mode = st.radio(
    "Mode", ["Customer", "Admin", "Assistant"],
    horizontal=True, label_visibility="collapsed",
)

if mode == "Customer":
    render_customer()
elif mode == "Admin":
    render_admin()
else:
    render_assistant()

# Collapsed is not the same as not executed -- everything inside an expander runs on every script
# run. These are two database round trips, so they sit behind an explicit button instead.
with st.expander("Diagnostics"):
    if st.button("Run connectivity check"):
        render_diagnostics()
    else:
        st.caption(
            "Queries both backends to confirm they are reachable. Not run automatically: it costs "
            "a warehouse query and a Lakebase query every time the page re-runs."
        )
        render_runtime_environment()
