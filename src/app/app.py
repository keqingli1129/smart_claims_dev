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

import os

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
    return psycopg2.connect(
        host=endpoint.status.hosts.host,
        dbname=os.getenv("PGDATABASE", "databricks_postgres"),
        user=os.getenv("PGUSER") or Config().client_id,
        password=credential.token,
        port=int(os.getenv("PGPORT", "5432")),
        sslmode="require",
        connect_timeout=30,
    )


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
    with lakebase_connection().cursor() as cur:
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
    with lakebase_connection().cursor() as cur:
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
    _render_claim_form(policy)


def _render_claim_form(policy: dict) -> None:
    """The claim details, gated behind a successful policy lookup.

    st.form batches the inputs: without it Streamlit re-runs this whole file on every keystroke,
    which would re-trigger the policy lookup on each character typed into the notes box. Inside a
    form, nothing happens until the submit button is pressed.
    """
    from datetime import date

    st.markdown("**Claim details**")

    with st.form("claim_details"):
        left, right = st.columns(2)
        with left:
            incident_date = st.date_input(
                "Date of the incident",
                value=date.today(),
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
        submitted = st.form_submit_button("Review claim", type="primary")

    if not submitted:
        return

    # Validation the widgets cannot express. The amount bound is a sanity check, not the coverage
    # check -- whether the policy actually covers it is one of the four automated checks at
    # submission, and is deliberately NOT pre-judged here.
    problems = []
    if not location.strip():
        problems.append("Tell us where the incident happened.")
    if incident_date < policy["effective"]:
        problems.append(
            f"The incident date is before this policy began ({policy['effective']})."
        )
    if problems:
        for problem in problems:
            st.error(problem)
        return

    # Held in session state, not written anywhere. The write, and the four checks that decide
    # whether the claim clears automatically, are the next step.
    st.session_state.pending_claim = {
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
    }

    st.success("Ready to submit.")
    pending = st.session_state.pending_claim
    st.dataframe(
        pd.DataFrame(
            [
                ("Policy", pending["policy_number"]),
                ("Policy holder", pending["customer_name"]),
                ("Incident", f"{pending['incident_type']} on {pending['incident_date']}"),
                ("Location", pending["accident_location"]),
                ("Amount claimed", f"${pending['claim_amount']:,.2f}"),
                ("Your assessment", pending["self_assessed_severity"]),
                ("Vehicles involved", str(pending["vehicles_involved"])),
                ("Filed by", pending["submitted_by"]),
            ],
            columns=["Field", "Value"],
        ),
        hide_index=True,
        use_container_width=True,
    )

    # Named now so the shape of what follows is visible, and so a reader can see that the amount
    # is not being silently judged against the policy at this stage.
    st.info(
        "Nothing has been submitted yet. Submitting will run four checks -- your severity "
        "assessment against the damage model, the amount against your sum insured, the incident "
        "date against your coverage window, and recorded speed where a telematics device is "
        "fitted. That step is not built yet."
    )


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

    st.info(
        "Review queue and per-claim detail are not built yet -- they read the claims this app "
        "takes in, which begins with the customer submission flow."
    )


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
# admin mode". Worth being explicit about what this is -- a workflow switch, NOT a security
# boundary. Every query runs as the app's service principal whichever tab is open, and nothing
# here checks who you are. Real separation would mean per-user authorization, a different design.
tab_customer, tab_admin, tab_assistant = st.tabs(["Customer", "Admin", "Assistant"])

with tab_customer:
    render_customer()

with tab_admin:
    render_admin()

with tab_assistant:
    render_assistant()

# Outside the tabs and collapsed by default: reachable from any screen, in the way of none.
with st.expander("Diagnostics"):
    render_diagnostics()
