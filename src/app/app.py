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
    st.info(
        "Not built yet. This is where a customer uploads a photo of the damage, the model "
        "classifies its severity, and the claim is submitted through four automated checks."
    )


def render_admin() -> None:
    st.subheader("Claims review")
    st.info(
        "Not built yet. This is where the portfolio overview and the review queue will live -- "
        "aggregates from the gold table, and the claims this app has taken in."
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
