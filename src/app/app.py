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

import psycopg2
import streamlit as st
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


@st.cache_data(ttl=60)
def lakebase_claim_count() -> int:
    # A fresh cursor per call, but the CONNECTION is cached -- opening a Postgres connection per
    # query is what exhausts the pool under any real traffic.
    with lakebase_connection().cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {PG_CLAIMS_TABLE}")
        return cur.fetchone()[0]


# --- page ---------------------------------------------------------------------------------------
st.title("🚗 Smart Claims")
st.caption(f"Signed in as **{current_user()}** · queries run as the app's service principal")

st.subheader("Connectivity check")
st.write(
    "Both backends should report the same number. They are the same data: the Lakebase row is a "
    "snapshot copy of the gold table, synced once. A mismatch means the snapshot has gone stale "
    "relative to gold -- expected behaviour for SNAPSHOT sync, not a bug."
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

with st.expander("Runtime environment"):
    st.write(
        {
            "CLAIMS_TABLE": CLAIMS_TABLE or "(unset)",
            "PG_CLAIMS_TABLE": PG_CLAIMS_TABLE or "(unset)",
            "DATABRICKS_WAREHOUSE_ID": os.getenv("DATABRICKS_WAREHOUSE_ID", "(unset)"),
            "LAKEBASE_ENDPOINT": LAKEBASE_ENDPOINT or "(unset)",
            "CLAIMS_VOLUME": os.getenv("CLAIMS_VOLUME", "(unset)"),
            "PGUSER": os.getenv("PGUSER", "(unset -- falling back to SP client id)"),
            "PGHOST_injected": os.getenv("PGHOST", "(unset)"),
        }
    )
    # Printed so the real forwarded-header names can be confirmed from the deployed app instead
    # of guessed. Remove once current_user() is known to pick the right one.
    try:
        st.write({"forwarded_headers": {k: v for k, v in (st.context.headers or {}).items()
                                        if k.lower().startswith("x-forwarded")}})
    except Exception:
        st.write("headers unavailable (local run)")
