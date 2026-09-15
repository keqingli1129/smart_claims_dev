# Databricks notebook source
# MAGIC %md
# MAGIC # Simulated SQL Server source database
# MAGIC
# MAGIC Stands in for the RDS SQL Server that Part 2 ingests with Lakeflow Connect. That path
# MAGIC needs a real database and an ingestion gateway on classic compute; this needs neither.
# MAGIC
# MAGIC Three Delta tables with `delta.enableChangeDataFeed = true` -- the direct analogue of
# MAGIC the transcript's `sys.sp_cdc_enable_table`. Mutating them with ordinary SQL produces a
# MAGIC real change feed, so nothing about the CDC downstream is faked; only the origin of the
# MAGIC feed differs.
# MAGIC
# MAGIC * **`mode=seed`** -- create the tables empty, enable the feed, then load. Run once.
# MAGIC * **`mode=mutate`** -- the transcript's insert / update / delete, each its own commit.

# COMMAND ----------

import random
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

from pyspark.sql.types import (
    DateType, DecimalType, StringType, StructField, StructType, TimestampType,
)

dbutils.widgets.dropdown("mode", "seed", ["seed", "mutate"], "Mode")
dbutils.widgets.text("source_schema", "", "Fully qualified source schema")
dbutils.widgets.text("random_seed", "42", "Random seed")
dbutils.widgets.dropdown("force", "false", ["false", "true"], "Force reseed")
dbutils.widgets.text("churn", "0", "Extra random claim updates (mutate mode)")

MODE = dbutils.widgets.get("mode")
SOURCE_SCHEMA = dbutils.widgets.get("source_schema").strip()
RANDOM_SEED = int(dbutils.widgets.get("random_seed"))
FORCE = dbutils.widgets.get("force") == "true"
CHURN = int(dbutils.widgets.get("churn"))

if not SOURCE_SCHEMA:
    raise ValueError("source_schema is required, e.g. smart_claims_dev.dev_me_source")

N_CUSTOMERS = 7_000
N_POLICIES = 12_000
N_CLAIMS = 13_000

print(f"mode={MODE} schema={SOURCE_SCHEMA} seed={RANDOM_SEED}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Schemas
# MAGIC
# MAGIC Shaped like tables a claims system would actually have. `policy.chassis_number` is the
# MAGIC join back to Part 1 -- the telematics generator was written against a ten-vehicle fleet
# MAGIC "so the same chassis recurs and will join against policy data in a later part".

# COMMAND ----------

CUSTOMER_SCHEMA = StructType([
    StructField("customer_id", StringType(), False),
    StructField("first_name", StringType()),
    StructField("last_name", StringType()),
    StructField("date_of_birth", DateType()),
    StructField("email", StringType()),
    StructField("phone", StringType()),
    StructField("address", StringType()),
    StructField("city", StringType()),
    StructField("state", StringType()),
    StructField("zip_code", StringType()),
    StructField("created_at", TimestampType()),
    StructField("updated_at", TimestampType()),
])

POLICY_SCHEMA = StructType([
    StructField("policy_number", StringType(), False),
    StructField("customer_id", StringType()),
    StructField("chassis_number", StringType()),
    StructField("policy_type", StringType()),
    StructField("effective_date", DateType()),
    StructField("expiry_date", DateType()),
    StructField("premium", DecimalType(10, 2)),
    StructField("sum_insured", DecimalType(12, 2)),
    StructField("deductible", DecimalType(10, 2)),
    StructField("created_at", TimestampType()),
    StructField("updated_at", TimestampType()),
])

CLAIM_SCHEMA = StructType([
    StructField("claim_number", StringType(), False),
    StructField("policy_number", StringType()),
    StructField("incident_date", DateType()),
    StructField("incident_type", StringType()),
    StructField("incident_severity", StringType()),
    StructField("claim_amount", DecimalType(12, 2)),
    StructField("claim_date", DateType()),
    StructField("claim_status", StringType()),
    StructField("created_at", TimestampType()),
    StructField("updated_at", TimestampType()),
])

TABLES = {"customer": CUSTOMER_SCHEMA, "policy": POLICY_SCHEMA, "claim": CLAIM_SCHEMA}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generators
# MAGIC
# MAGIC Small fixed vocabularies and a seeded RNG: reseeding reproduces the same rows, and no
# MAGIC new dependency is needed -- pyproject.toml's dependency list stays empty.
# MAGIC `incident_severity` uses the transcript's own wording, so its demo transfers verbatim.

# COMMAND ----------

FIRST_NAMES = ["Maria", "James", "Aisha", "Chen", "Sofia", "Omar", "Elena", "Noah",
               "Priya", "Lucas", "Fatima", "Henry", "Yuki", "Diego", "Anna", "Ravi"]
LAST_NAMES = ["Garcia", "Smith", "Okafor", "Wang", "Rossi", "Haddad", "Novak", "Johnson",
              "Patel", "Silva", "Ahmed", "Muller", "Tanaka", "Lopez", "Nielsen", "Kumar"]
CITIES = [("Houston", "TX", "770"), ("Katy", "TX", "774"), ("Sugar Land", "TX", "774"),
          ("Richmond", "TX", "774"), ("Pearland", "TX", "775"), ("Missouri City", "TX", "774")]
STREETS = ["Oak", "Maple", "Cedar", "Elm", "Pine", "Willow", "Birch", "Ash"]

POLICY_TYPES = ["Comprehensive", "Third Party", "Collision", "Liability"]
INCIDENT_TYPES = ["Collision", "Theft", "Fire", "Hail", "Vandalism", "Flood"]
# The transcript flips a claim from Total Loss to Minor Damage. Same vocabulary here.
SEVERITIES = ["Total Loss", "Major Damage", "Minor Damage"]
CLAIM_STATUSES = ["Open", "Under Review", "Approved", "Denied", "Settled"]

EPOCH = date(2020, 1, 1)


def _money(rng, low, high):
    return Decimal(str(round(rng.uniform(low, high), 2)))


def make_customers(rng, n):
    now = datetime.now(timezone.utc)
    rows = []
    for i in range(1, n + 1):
        first, last = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
        city, state, zip_prefix = rng.choice(CITIES)
        rows.append((
            f"CUST-{i:05d}", first, last,
            EPOCH - timedelta(days=rng.randint(6570, 25550)),
            f"{first.lower()}.{last.lower()}{i}@example.com",
            f"713-{rng.randint(200, 999)}-{rng.randint(1000, 9999)}",
            f"{rng.randint(100, 9999)} {rng.choice(STREETS)} St",
            city, state, f"{zip_prefix}{rng.randint(10, 99)}",
            now, now,
        ))
    return rows


def make_policies(rng, n, n_customers):
    now = datetime.now(timezone.utc)
    rows = []
    for i in range(1, n + 1):
        effective = EPOCH + timedelta(days=rng.randint(0, 1800))
        rows.append((
            f"POL-{i:06d}",
            f"CUST-{rng.randint(1, n_customers):05d}",
            # Policy i owns chassis i, so Part 1's CH-00001..CH-00010 fleet is the first ten
            # policies. Most policies carry no telematics device, which is also reality.
            f"CH-{i:05d}",
            rng.choice(POLICY_TYPES),
            effective, effective + timedelta(days=365),
            _money(rng, 300, 2500), _money(rng, 5000, 90000), _money(rng, 250, 2000),
            now, now,
        ))
    return rows


def make_claims(rng, n, n_policies):
    now = datetime.now(timezone.utc)
    rows = []
    for i in range(1, n + 1):
        incident = EPOCH + timedelta(days=rng.randint(400, 2100))
        rows.append((
            f"CLM-{i:06d}",
            f"POL-{rng.randint(1, n_policies):06d}",
            incident,
            rng.choice(INCIDENT_TYPES),
            rng.choice(SEVERITIES),
            _money(rng, 500, 60000),
            incident + timedelta(days=rng.randint(0, 21)),
            rng.choice(CLAIM_STATUSES),
            now, now,
        ))
    return rows

# COMMAND ----------

# MAGIC %md
# MAGIC ## Seed
# MAGIC
# MAGIC Three commits per table, deliberately:
# MAGIC
# MAGIC * **v0** create the table EMPTY
# MAGIC * **v1** `ALTER TABLE ... SET TBLPROPERTIES (delta.enableChangeDataFeed = true)`
# MAGIC * **v2** insert the seed rows
# MAGIC
# MAGIC A `CREATE TABLE ... AS SELECT` would collapse v1 and v2 into one version. The change
# MAGIC feed is guaranteed only for commits made *after* the property is set, so the seed could
# MAGIC be missing from the feed entirely -- and the pipeline would then succeed while producing
# MAGIC three empty bronze tables. A silent wrong answer is worse than a crash.

# COMMAND ----------

def table_exists(name):
    return spark.catalog.tableExists(f"{SOURCE_SCHEMA}.{name}")


def create_empty(name, schema):
    spark.sql(f"DROP TABLE IF EXISTS {SOURCE_SCHEMA}.{name}")
    spark.createDataFrame([], schema).write.format("delta").saveAsTable(f"{SOURCE_SCHEMA}.{name}")

    # An explicit ALTER, not a writer option: unambiguously a table property, and
    # unambiguously its own commit.
    spark.sql(
        f"ALTER TABLE {SOURCE_SCHEMA}.{name} "
        "SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
    )

    props = {r["key"]: r["value"] for r in
             spark.sql(f"SHOW TBLPROPERTIES {SOURCE_SCHEMA}.{name}").collect()}
    if props.get("delta.enableChangeDataFeed") != "true":
        raise RuntimeError(
            f"{name}: change data feed not enabled "
            f"(got {props.get('delta.enableChangeDataFeed')!r})"
        )
    print(f"created {SOURCE_SCHEMA}.{name} (empty, CDF on)")


if MODE == "seed":
    existing = [t for t in TABLES if table_exists(t)]
    if existing and not FORCE:
        raise RuntimeError(
            f"{', '.join(existing)} already exist. Re-seeding recreates the tables, which "
            "restarts their change feed at version 0 and invalidates the source_cdc pipeline's "
            "checkpoint. Re-run with force=true AND full-refresh that pipeline afterwards."
        )

    rng = random.Random(RANDOM_SEED)
    seeded = {
        "customer": (make_customers(rng, N_CUSTOMERS), CUSTOMER_SCHEMA),
        "policy": (make_policies(rng, N_POLICIES, N_CUSTOMERS), POLICY_SCHEMA),
        "claim": (make_claims(rng, N_CLAIMS, N_POLICIES), CLAIM_SCHEMA),
    }

    for name, schema in TABLES.items():
        create_empty(name, schema)

    for name, (rows, schema) in seeded.items():
        spark.createDataFrame(rows, schema).write.mode("append").saveAsTable(
            f"{SOURCE_SCHEMA}.{name}"
        )
        print(f"seeded {SOURCE_SCHEMA}.{name}: "
              f"{spark.table(f'{SOURCE_SCHEMA}.{name}').count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Mutate
# MAGIC
# MAGIC The transcript's three operations, each a separate statement and therefore a separate
# MAGIC Delta commit -- so `_commit_version` orders them unambiguously downstream.

# COMMAND ----------

if MODE == "mutate":
    missing = [t for t in TABLES if not table_exists(t)]
    if missing:
        raise RuntimeError(f"{', '.join(missing)} do not exist. Run mode=seed first.")

    # 1. INSERT a policy that does not exist yet.
    spark.sql(f"""
        INSERT INTO {SOURCE_SCHEMA}.policy VALUES (
            'POL-999001', 'CUST-00042', 'CH-99001', 'Comprehensive',
            DATE'2026-01-01', DATE'2027-01-01',
            1499.00, 45000.00, 750.00,
            current_timestamp(), current_timestamp()
        )
    """)
    print("inserted policy POL-999001")

    # 2. UPDATE one claim's severity -- the transcript's total loss -> minor damage.
    spark.sql(f"""
        UPDATE {SOURCE_SCHEMA}.claim
        SET incident_severity = 'Minor Damage', updated_at = current_timestamp()
        WHERE claim_number = 'CLM-000001'
    """)
    print("updated claim CLM-000001 severity -> Minor Damage")

    # 3. DELETE one customer.
    spark.sql(f"DELETE FROM {SOURCE_SCHEMA}.customer WHERE customer_id = 'CUST-00001'")
    print("deleted customer CUST-00001")

    # Optional bulk churn, for exercising the flow at volume rather than at three rows. Each
    # UPDATE is its own statement and therefore its own commit, so no commit ever carries two
    # changes to one key. Claims start at 2 so CLM-000001 -- the one the verification query
    # checks -- is never disturbed.
    if CHURN:
        churn_rng = random.Random(RANDOM_SEED + 1)
        for _ in range(CHURN):
            claim_no = f"CLM-{churn_rng.randint(2, N_CLAIMS):06d}"
            spark.sql(f"""
                UPDATE {SOURCE_SCHEMA}.claim
                SET incident_severity = '{churn_rng.choice(SEVERITIES)}',
                    claim_status = '{churn_rng.choice(CLAIM_STATUSES)}',
                    updated_at = current_timestamp()
                WHERE claim_number = '{claim_no}'
            """)
        print(f"churn: {CHURN} additional claim updates committed")

    print(f"{3 + CHURN} changes committed. Run the source_cdc pipeline to propagate them.")
