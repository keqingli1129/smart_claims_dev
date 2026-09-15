# Simulated SQL Server CDC Ingestion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ingest three simulated SQL Server tables into bronze with CDC semantics, on serverless compute, behind a swap boundary that lets Lakeflow Connect replace the simulation later.

**Architecture:** A `source` schema holds three Delta tables with `delta.enableChangeDataFeed = true`, standing in for the SQL Server. A job seeds them and mutates them with ordinary INSERT/UPDATE/DELETE. A separate pipeline reads each table's Delta change feed through a temporary view and applies it to a bronze streaming table with `create_auto_cdc_flow` (SCD Type 1).

**Tech Stack:** Declarative Automation Bundles, Lakeflow Spark Declarative Pipelines (`pyspark.pipelines`), Delta change data feed, serverless compute only.

**Spec:** `docs/001-source-cdc-simulation/spec.md`

## Global Constraints

- Everything runs on **serverless compute**. No classic cluster is created. (SC-006)
- **No automated tests are required.** Decided at specification time; reviewers must not raise missing-test findings. Verification is by the commands in each task. (spec §Testing)
- Bronze tables MUST NOT carry `_change_type`, `_commit_version` or `_commit_timestamp`. (FR-011, SC-004)
- All flows use `stored_as_scd_type=1`. SCD Type 2 is User Story 3, explicitly out of scope.
- Source tables MUST be created empty, then populated in a separate commit. Never `CREATE TABLE AS SELECT`. (FR-002)
- Each mutation is its own commit; no commit contains two changes to one primary key. (FR-005)
- Generated data MUST be deterministic under a seed parameter. (FR-006)
- No new project dependencies. `pyproject.toml` has an empty `dependencies` list and stays that way — use `random` with an explicit seed, not Faker.
- Profile is chosen by the operator, never auto-selected. Commands below show `--profile DEFAULT`; substitute as needed.
- Row counts: customer 7,000 / policy 12,000 / claim 13,000. (spec §Key Entities)

## File Structure

| File | Responsibility |
|---|---|
| `databricks.yml` | *modify* — declare the `source_schema` variable |
| `resources/catalog.yml` | *modify* — declare the `source` schema resource |
| `src/jobs/simulate_source_database.py` | *create* — seed and mutate the source tables |
| `resources/source_database.job.yml` | *create* — the job resource |
| `src/smart_claims_dev_etl/utilities/cdc_source.py` | *create* — shared change-feed read + AUTO CDC flow |
| `src/smart_claims_dev_etl/transformations_dev/customer.py` | *create* — bronze.customer |
| `src/smart_claims_dev_etl/transformations_dev/policy.py` | *create* — bronze.policy |
| `src/smart_claims_dev_etl/transformations_dev/claim.py` | *create* — bronze.claim |
| `resources/source_cdc.pipeline.yml` | *create* — the CDC pipeline resource |
| `README.md` | *modify* — run order, swap path, why the simulation exists |

**Dependency chain — tasks must run in order:**

```
Task 1  source schema exists
   └── Task 2  tables seeded, change feed has content
         └── Task 2  mutations produce further change records
               └── Task 3  one table proves the CDC pattern
                     └── Task 4  the other two follow it
                           └── Task 5  convergence demo + docs
```

**Part 1 is not modified at all.** The CDC datasets live in `transformations_dev/`, a sibling
of `transformations/` rather than a subfolder of it. Part 1's glob says
`transformations/**`, which requires that exact directory name followed by a separator, so it
never reaches `transformations_dev/`. No file moves, no edit to
`smart_claims_dev_etl.pipeline.yml`.

Both pipelines keep the same `root_path`, which is what puts `utilities/` on the import path
for both. A second `root_path` would mean a second copy of the shared code.

Note that `**` is not stylistic: the pipelines API rejects a single asterisk outright —
*"Single asterisk glob pattern is not supported ... Use a double asterisk '**'"* — so
`transformations/*.py` is not an alternative way to achieve this.

---

### Task 1: Source schema

**Files:**
- Modify: `databricks.yml` (variables block)
- Modify: `resources/catalog.yml` (schemas block)

**Interfaces:**
- Consumes: nothing
- Produces: `${var.source_schema}` (default `source`), and the resource
  `${resources.schemas.source.name}` which resolves to `dev_keqingli1129_source` in the dev target.

- [ ] **Step 1: Declare the variable**

In `databricks.yml`, directly after the `landing_volume` variable:

```yaml
  source_schema:
    description: >-
      Stands in for the SQL Server database Part 2 ingests from. Holds three Delta tables with
      the change data feed enabled; the CDC pipeline reads that feed the way Lakeflow Connect's
      gateway would read a transaction log. Not a medallion layer -- it represents a system
      outside the lakehouse, and disappears when the managed connector replaces it.
    default: source
```

- [ ] **Step 2: Declare the schema resource**

In `resources/catalog.yml`, inside `schemas:`, after `landing:`:

```yaml
    source:
      catalog_name: ${var.catalog}
      name: ${var.source_schema}
      comment: >-
        Stand-in for the external SQL Server database of Part 2. Three tables with the Delta
        change data feed enabled, mutated by the source_database job. Deleted when Lakeflow
        Connect becomes available.
```

- [ ] **Step 3: Validate the resolution**

Run:
```bash
databricks bundle validate -t dev --profile DEFAULT -o json \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['resources']['schemas']['source'])"
```
Expected: a dict whose `name` is `dev_<your_username>_source`.

- [ ] **Step 4: Deploy and confirm the schema exists**

Run:
```bash
databricks bundle deploy -t dev --profile DEFAULT
databricks schemas list smart_claims_dev --profile DEFAULT -o json \
  | python3 -c "import json,sys; print([s['name'] for s in json.load(sys.stdin)])"
```
Expected: the list includes a `dev_<your_username>_source` entry.

- [ ] **Step 5: Commit**

```bash
git add databricks.yml resources/catalog.yml
git commit -m "Add source schema standing in for the Part 2 SQL Server database"
```

---

### Task 2: Seed the source database

**Files:**
- Create: `src/jobs/simulate_source_database.py`
- Create: `resources/source_database.job.yml`

**Interfaces:**
- Consumes: `${resources.schemas.source.name}` from Task 1.
- Produces: `<catalog>.<source_schema>.customer` (7,000 rows), `.policy` (12,000),
  `.claim` (13,000), each with `delta.enableChangeDataFeed = true`.
  `policy.chassis_number` runs `CH-00001`–`CH-12000`; `policy` row *i* gets `CH-{i:05d}`,
  so Part 1's ten-vehicle fleet is the first ten policies.

- [ ] **Step 1: Write the notebook**

Create `src/jobs/simulate_source_database.py`:

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # Simulated SQL Server source database
# MAGIC
# MAGIC Stands in for the RDS SQL Server that Part 2 ingests with Lakeflow Connect. That path
# MAGIC needs a real database and a classic-compute ingestion gateway; this needs neither.
# MAGIC
# MAGIC Three Delta tables with `delta.enableChangeDataFeed = true` -- the direct analogue of
# MAGIC the transcript's `sys.sp_cdc_enable_table`. Mutating them with ordinary SQL produces a
# MAGIC real change feed, so nothing about the CDC downstream is faked.
# MAGIC
# MAGIC * **`mode=seed`** -- create the tables empty, then bulk load. Run once.
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
# MAGIC join back to Part 1's telematics data.

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
# MAGIC Small fixed vocabularies and a seeded RNG, so a reseed reproduces the same rows and no
# MAGIC new dependency is needed. `incident_severity` uses the transcript's own wording.

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
# The transcript's vocabulary. Its demo flips a claim from Total Loss to Minor Damage.
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
# MAGIC Tables are created EMPTY with the change feed enabled, then loaded in a second commit.
# MAGIC A `CREATE TABLE AS SELECT` would make the enabling commit and the data commit the same
# MAGIC version, and the change feed is only guaranteed for commits made after the property is
# MAGIC set -- the pipeline's first run could then see nothing at all.

# COMMAND ----------

def table_exists(name):
    return spark.catalog.tableExists(f"{SOURCE_SCHEMA}.{name}")


def create_empty(name, schema):
    spark.sql(f"DROP TABLE IF EXISTS {SOURCE_SCHEMA}.{name}")
    spark.createDataFrame([], schema).write.format("delta") \
        .saveAsTable(f"{SOURCE_SCHEMA}.{name}")
    # Enabled as an explicit ALTER rather than a writer option, so it is unambiguously a table
    # property and unambiguously its own commit. Table history is then: v0 create, v1 enable,
    # v2 seed load -- and the seed is safely after the enabling.
    spark.sql(
        f"ALTER TABLE {SOURCE_SCHEMA}.{name} "
        "SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
    )
    # Belt and braces: assert the property actually took.
    props = spark.sql(f"SHOW TBLPROPERTIES {SOURCE_SCHEMA}.{name}").collect()
    enabled = {r["key"]: r["value"] for r in props}.get("delta.enableChangeDataFeed")
    if enabled != "true":
        raise RuntimeError(f"{name}: change data feed not enabled (got {enabled!r})")
    print(f"created {SOURCE_SCHEMA}.{name} (empty, CDF on)")


if MODE == "seed":
    existing = [t for t in TABLES if table_exists(t)]
    if existing and not FORCE:
        raise RuntimeError(
            f"{', '.join(existing)} already exist. Re-seeding recreates the tables, which "
            "restarts their change feed at version 0 and invalidates the source_cdc pipeline's "
            "checkpoint. Re-run with force=true AND full-refresh the pipeline afterwards."
        )

    rng = random.Random(RANDOM_SEED)
    customers = make_customers(rng, N_CUSTOMERS)
    policies = make_policies(rng, N_POLICIES, N_CUSTOMERS)
    claims = make_claims(rng, N_CLAIMS, N_POLICIES)

    for name, schema in TABLES.items():
        create_empty(name, schema)

    for name, rows, schema in [
        ("customer", customers, CUSTOMER_SCHEMA),
        ("policy", policies, POLICY_SCHEMA),
        ("claim", claims, CLAIM_SCHEMA),
    ]:
        spark.createDataFrame(rows, schema).write.mode("append") \
            .saveAsTable(f"{SOURCE_SCHEMA}.{name}")
        n = spark.table(f"{SOURCE_SCHEMA}.{name}").count()
        print(f"seeded {SOURCE_SCHEMA}.{name}: {n} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Mutate
# MAGIC
# MAGIC The transcript's three operations, each a separate statement and therefore a separate
# MAGIC Delta commit -- so `_commit_version` orders them unambiguously downstream.

# COMMAND ----------

if MODE == "mutate":
    for name in TABLES:
        if not table_exists(name):
            raise RuntimeError(f"{SOURCE_SCHEMA}.{name} does not exist. Run mode=seed first.")

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

    # 2. UPDATE one claim's severity -- the transcript changes total loss to minor damage.
    spark.sql(f"""
        UPDATE {SOURCE_SCHEMA}.claim
        SET incident_severity = 'Minor Damage', updated_at = current_timestamp()
        WHERE claim_number = 'CLM-000001'
    """)
    print("updated claim CLM-000001 severity -> Minor Damage")

    # 3. DELETE one customer.
    spark.sql(f"DELETE FROM {SOURCE_SCHEMA}.customer WHERE customer_id = 'CUST-00001'")
    print("deleted customer CUST-00001")

    # Optional bulk churn, for exercising the flow at volume rather than at three rows.
    # Each UPDATE is its own statement and therefore its own commit, so FR-005 holds even if
    # the same claim is drawn twice. Claims start at 2 so CLM-000001 -- the one the
    # verification query checks -- is never disturbed.
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
```

- [ ] **Step 2: Write the job resource**

Create `resources/source_database.job.yml`:

```yaml
# Stands in for Part 2's SQL Server. Seeds three Delta tables with the change data feed
# enabled, then mutates them the way the transcript mutates its RDS database through DataGrip.
#
# This job disappears entirely if Lakeflow Connect becomes available -- see the swap path in
# docs/001-source-cdc-simulation/spec.md.

resources:
  jobs:
    source_database:
      name: source_database
      description: Seeds and mutates the simulated SQL Server source tables.

      tasks:
        - task_key: simulate
          # No cluster spec -> serverless job compute.
          notebook_task:
            notebook_path: ../src/jobs/simulate_source_database.py
            base_parameters:
              mode: seed
              source_schema: ${var.catalog}.${resources.schemas.source.name}
              random_seed: "42"
              force: "false"
```

- [ ] **Step 3: Deploy and seed**

```bash
databricks bundle deploy -t dev --profile DEFAULT
databricks bundle run source_database -t dev --profile DEFAULT --notebook-params mode=seed
```
Expected output includes `seeded ...customer: 7000 rows`, `...policy: 12000 rows`,
`...claim: 13000 rows`.

- [ ] **Step 4: Confirm the change feed is on and populated**

```bash
databricks experimental aitools tools query \
  "SHOW TBLPROPERTIES smart_claims_dev.dev_keqingli1129_source.claim" --profile DEFAULT
```
Expected: `delta.enableChangeDataFeed` = `true`.

```bash
databricks experimental aitools tools query \
  "SELECT _change_type, count(*) AS n
   FROM table_changes('smart_claims_dev.dev_keqingli1129_source.claim', 0)
   GROUP BY _change_type" --profile DEFAULT
```
Expected: one row, `insert` = 13000. If this returns zero rows, FR-002's create-empty-first
rule was not followed and the seed landed in the same commit as the enabling.

- [ ] **Step 5: Confirm the reseed guard**

```bash
databricks bundle run source_database -t dev --profile DEFAULT --notebook-params mode=seed
```
Expected: the task FAILS with the message naming `force=true` and the full-refresh
consequence. (This is the desired outcome, not a problem.)

- [ ] **Step 6: Commit**

```bash
git add src/jobs/simulate_source_database.py resources/source_database.job.yml
git commit -m "Add simulated SQL Server source database with change data feed"
```

---

### Task 3: Mutations

**Files:**
- Modify: none — `mode=mutate` was written in Task 2. This task verifies it.

**Interfaces:**
- Consumes: the seeded tables from Task 2.
- Produces: exactly three further change records — one `insert` on `policy`, one
  `update_postimage` (plus its `update_preimage`) on `claim`, one `delete` on `customer`.

- [ ] **Step 1: Run the mutation**

```bash
databricks bundle run source_database -t dev --profile DEFAULT --notebook-params mode=mutate
```
Expected: three printed lines — inserted, updated, deleted.

- [ ] **Step 2: Confirm each change reached the feed**

```bash
databricks experimental aitools tools query \
  "SELECT 'policy' AS t, _change_type, count(*) AS n
     FROM table_changes('smart_claims_dev.dev_keqingli1129_source.policy', 1) GROUP BY 2
   UNION ALL SELECT 'claim', _change_type, count(*)
     FROM table_changes('smart_claims_dev.dev_keqingli1129_source.claim', 1) GROUP BY 2
   UNION ALL SELECT 'customer', _change_type, count(*)
     FROM table_changes('smart_claims_dev.dev_keqingli1129_source.customer', 1) GROUP BY 2" \
  --profile DEFAULT
```
Expected: `policy`/`insert` = 1; `claim`/`update_preimage` = 1 and `claim`/`update_postimage`
= 1; `customer`/`delete` = 1.

Note the starting version is `1`, not `0` — version 0 is the empty table creation and
version 1 is the seed load, so starting at 1 excludes neither but keeps the output small.
If the counts are larger, `mode=mutate` was run more than once; that is harmless.

- [ ] **Step 3: Optionally exercise it at volume**

```bash
databricks bundle run source_database -t dev --profile DEFAULT \
  --notebook-params mode=mutate,churn=50
```
Expected: `churn: 50 additional claim updates committed`. Each is a separate commit, so the
claim feed gains 50 `update_postimage` records. Skip this if three changes are enough.

- [ ] **Step 4: No commit**

Nothing changed on disk. Proceed to Task 4.

---

### Task 4: Prove the CDC pattern on one table

**Files:**
- Create: `src/smart_claims_dev_etl/utilities/cdc_source.py`
- Create: `src/smart_claims_dev_etl/transformations_dev/customer.py`
- Create: `resources/source_cdc.pipeline.yml`

**Interfaces:**
- Consumes: `spark.conf` key `source_schema`, set by the pipeline resource to
  `<catalog>.<resolved source schema>`.
- Produces, for Task 5 to reuse verbatim:
  - `read_change_feed(spark, table: str) -> DataFrame` — streaming CDF read, preimages filtered
  - `apply_cdc(target: str, source: str, keys: list[str], comment: str) -> None` — creates the
    streaming table and its AUTO CDC flow

**Why one table first:** if the CDF-plus-AUTO-CDC pattern is wrong, this surfaces it with one
dataset rather than three, and one pipeline run rather than a triplicated mistake.

- [ ] **Step 1: Write the shared helper**

Create `src/smart_claims_dev_etl/utilities/cdc_source.py`:

```python
"""Delta change feed -> bronze, the serverless stand-in for Lakeflow Connect.

Part 2 of the transcript ingests SQL Server through Lakeflow Connect: an ingestion gateway
reads the transaction log into a volume, and a managed ingestion pipeline upserts from there
into bronze. Both halves need classic compute and a real database.

What survives the substitution is the part that matters -- CDC semantics. A Delta table with
`delta.enableChangeDataFeed = true` produces a genuine change feed, so nothing here is faked;
only the source of the feed differs.

This module lives in utilities/ rather than transformations_dev/ because a pipeline's glob
executes every file it matches, and this file defines no datasets. utilities/ sits outside
both pipelines' globs, which is also what lets both import from it.
"""

from pyspark import pipelines as dp
from pyspark.sql import functions as F

# Change-feed bookkeeping. Dropped from the target so bronze matches what Lakeflow Connect
# would produce -- this is what keeps the swap boundary honest rather than approximate.
CDF_METADATA_COLUMNS = ["_change_type", "_commit_version", "_commit_timestamp"]


def read_change_feed(spark, table: str):
    """Stream one source table's change feed, after-images only.

    `update_preimage` is the row as it was before an update. AUTO CDC wants the new value, so
    the before-image is dropped here rather than downstream.
    """
    schema = spark.conf.get("source_schema")
    return (
        spark.readStream.option("readChangeFeed", "true")
        .table(f"{schema}.{table}")
        .filter(F.col("_change_type") != "update_preimage")
    )


def apply_cdc(target: str, source: str, keys: list[str], comment: str) -> None:
    """Create a bronze streaming table and the AUTO CDC flow that maintains it.

    `source` must be the NAME of a view or table -- create_auto_cdc_flow rejects a DataFrame --
    which is why each dataset file declares its own @dp.temporary_view first.

    `_commit_version` is the Delta analogue of a log sequence number: it orders changes across
    commits, which is why the source job commits each mutation separately.
    """
    dp.create_streaming_table(
        name=target,
        comment=comment,
        table_properties={"quality": "bronze"},
    )

    dp.create_auto_cdc_flow(
        target=target,
        source=source,
        keys=keys,
        sequence_by="_commit_version",
        apply_as_deletes=F.expr("_change_type = 'delete'"),
        except_column_list=CDF_METADATA_COLUMNS,
        # SCD Type 1: current state only. A delete removes the row, which is what makes the
        # transcript's "no records returned" check the correct assertion. SCD Type 2 is a
        # later story.
        stored_as_scd_type=1,
    )
```

- [ ] **Step 2: Write the first dataset**

Create `src/smart_claims_dev_etl/transformations_dev/customer.py`:

```python
"""Transcript section: "Lakeflow Connect Setup" -- the customer table.

One of the three tables Part 2 ingests from SQL Server. Here the change feed comes from a
Delta table instead of a transaction log; everything downstream of that is identical.
"""

from pyspark import pipelines as dp

from utilities.cdc_source import apply_cdc, read_change_feed


@dp.temporary_view()
def customer_changes():
    # Named explicitly because apply_cdc references this view by name, not by value.
    return read_change_feed(spark, "customer")


apply_cdc(
    target="customer",
    source="customer_changes",
    keys=["customer_id"],
    comment="Customers from the simulated SQL Server source, CDC-applied (SCD Type 1).",
)
```

- [ ] **Step 3: Write the pipeline resource**

Create `resources/source_cdc.pipeline.yml`:

```yaml
# The serverless stand-in for Lakeflow Connect's ingestion pipeline.
#
# Deliberately SEPARATE from smart_claims_dev_etl. When Lakeflow Connect becomes available,
# this file's body is replaced by a gateway_definition + ingestion_definition pair targeting
# the same bronze schema, and nothing downstream notices. Had these datasets lived inside the
# telematics pipeline, that swap would mean surgery on a working pipeline.

resources:
  pipelines:
    source_cdc:
      name: source_cdc
      catalog: ${var.catalog}
      schema: ${var.schema}
      serverless: true
      root_path: "../src/smart_claims_dev_etl"

      libraries:
        - glob:
            include: ../src/smart_claims_dev_etl/transformations_dev/**

      configuration:
        # Fully qualified, so a two-part table reference resolves regardless of the pipeline's
        # current catalog. Resolves through the resource, so development mode's dev_keqingli1129_
        # renaming is picked up automatically.
        source_schema: ${var.catalog}.${resources.schemas.source.name}

      environment:
        dependencies:
          - --editable ${workspace.file_path}
```

- [ ] **Step 4: Deploy and run**

```bash
databricks bundle deploy -t dev --profile DEFAULT
databricks bundle run source_cdc -t dev --profile DEFAULT
```
Expected: the update completes. `bronze.customer` is created.

- [ ] **Step 5: Verify the contract**

```bash
databricks experimental aitools tools query \
  "SELECT (SELECT count(*) FROM smart_claims_dev.dev_keqingli1129_source.customer)   AS src,
          (SELECT count(*) FROM smart_claims_dev.dev_keqingli1129_bronze.customer)   AS bronze" \
  --profile DEFAULT
```
Expected: `src` and `bronze` are equal. If `mode=mutate` ran in Task 3, both are 6999 —
the deleted customer is absent from both, which is the whole point.

```bash
databricks experimental aitools tools query \
  "DESCRIBE smart_claims_dev.dev_keqingli1129_bronze.customer" --profile DEFAULT
```
Expected: twelve columns, and **none** of `_change_type`, `_commit_version`,
`_commit_timestamp`. If they appear, `except_column_list` is not taking effect — stop and fix
before Task 5 triplicates the mistake.

```bash
databricks experimental aitools tools query \
  "SELECT count(*) AS should_be_zero FROM smart_claims_dev.dev_keqingli1129_bronze.customer
   WHERE customer_id = 'CUST-00001'" --profile DEFAULT
```
Expected: 0 — the deleted customer did not survive into bronze.

- [ ] **Step 6: Commit**

```bash
git add src/smart_claims_dev_etl/utilities/cdc_source.py \
        src/smart_claims_dev_etl/transformations_dev/customer.py \
        resources/source_cdc.pipeline.yml
git commit -m "Add CDC ingestion from simulated source, proven on customer"
```

---

### Task 5: The remaining two tables

**Files:**
- Create: `src/smart_claims_dev_etl/transformations_dev/policy.py`
- Create: `src/smart_claims_dev_etl/transformations_dev/claim.py`

**Interfaces:**
- Consumes: `read_change_feed` and `apply_cdc` from Task 4, unchanged.
- Produces: `bronze.policy` (keyed `policy_number`), `bronze.claim` (keyed `claim_number`).

- [ ] **Step 1: Write policy.py**

Create `src/smart_claims_dev_etl/transformations_dev/policy.py`:

```python
"""Transcript section: "Lakeflow Connect Setup" -- the policy table.

`chassis_number` is the join back to Part 1: the telematics generator was written against a
ten-vehicle fleet "so the same chassis recurs and will join against policy data in a later
part". This is that later part.
"""

from pyspark import pipelines as dp

from utilities.cdc_source import apply_cdc, read_change_feed


@dp.temporary_view()
def policy_changes():
    return read_change_feed(spark, "policy")


apply_cdc(
    target="policy",
    source="policy_changes",
    keys=["policy_number"],
    comment="Policies from the simulated SQL Server source, CDC-applied (SCD Type 1).",
)
```

- [ ] **Step 2: Write claim.py**

Create `src/smart_claims_dev_etl/transformations_dev/claim.py`:

```python
"""Transcript section: "Test Incremental Load" -- the claim table.

The transcript's update demo lands here: one claim's incident_severity is changed from
Total Loss to Minor Damage in the source, and the CDC flow carries it through.
"""

from pyspark import pipelines as dp

from utilities.cdc_source import apply_cdc, read_change_feed


@dp.temporary_view()
def claim_changes():
    return read_change_feed(spark, "claim")


apply_cdc(
    target="claim",
    source="claim_changes",
    keys=["claim_number"],
    comment="Claims from the simulated SQL Server source, CDC-applied (SCD Type 1).",
)
```

- [ ] **Step 3: Deploy and run**

```bash
databricks bundle deploy -t dev --profile DEFAULT
databricks bundle run source_cdc -t dev --profile DEFAULT
```
Expected: three datasets in the update, all completing.

- [ ] **Step 4: Verify all three converge**

```bash
databricks experimental aitools tools query \
  "SELECT 'customer' AS t,
          (SELECT count(*) FROM smart_claims_dev.dev_keqingli1129_source.customer) AS src,
          (SELECT count(*) FROM smart_claims_dev.dev_keqingli1129_bronze.customer) AS bronze
   UNION ALL SELECT 'policy',
          (SELECT count(*) FROM smart_claims_dev.dev_keqingli1129_source.policy),
          (SELECT count(*) FROM smart_claims_dev.dev_keqingli1129_bronze.policy)
   UNION ALL SELECT 'claim',
          (SELECT count(*) FROM smart_claims_dev.dev_keqingli1129_source.claim),
          (SELECT count(*) FROM smart_claims_dev.dev_keqingli1129_bronze.claim)" \
  --profile DEFAULT
```
Expected: `src` equals `bronze` on every row. (SC-002)

- [ ] **Step 5: Verify the Part 1 join works**

```bash
databricks experimental aitools tools query \
  "SELECT count(*) AS joined
   FROM smart_claims_dev.dev_keqingli1129_bronze.policy p
   JOIN (SELECT DISTINCT chassis_number FROM smart_claims_dev.dev_keqingli1129_bronze.telematics) t
     ON p.chassis_number = t.chassis_number" --profile DEFAULT
```
Expected: 10 — Part 1's ten-vehicle fleet, each matching exactly one policy. (US2 scenario 5)

- [ ] **Step 6: Commit**

```bash
git add src/smart_claims_dev_etl/transformations_dev/
git commit -m "Add policy and claim CDC datasets"
```

---

### Task 6: Convergence demo and documentation

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: everything above.
- Produces: nothing code depends on.

- [ ] **Step 1: Run the transcript's incremental test end to end**

```bash
databricks bundle run source_database -t dev --profile DEFAULT --notebook-params mode=mutate
databricks bundle run source_cdc -t dev --profile DEFAULT
```

- [ ] **Step 2: Run the transcript's three verification queries**

```bash
databricks experimental aitools tools query \
  "SELECT 'inserted policy' AS check, count(*) AS n
     FROM smart_claims_dev.dev_keqingli1129_bronze.policy WHERE policy_number = 'POL-999001'
   UNION ALL
   SELECT 'updated severity', count(*)
     FROM smart_claims_dev.dev_keqingli1129_bronze.claim
     WHERE claim_number = 'CLM-000001' AND incident_severity = 'Minor Damage'
   UNION ALL
   SELECT 'deleted customer', count(*)
     FROM smart_claims_dev.dev_keqingli1129_bronze.customer WHERE customer_id = 'CUST-00001'" \
  --profile DEFAULT
```
Expected: `inserted policy` = 1, `updated severity` = 1, `deleted customer` = 0. (SC-003)

- [ ] **Step 3: Add a README section**

Insert a `## The simulated source database` section after `## The two bronze tables`,
covering: why the simulation exists (no SQL Server, gateway is classic compute), the run
order below, and the swap path to Lakeflow Connect. Add the three new tables to the
`## Unity Catalog layout` tree and a `source` line to the schema list.

```bash
databricks bundle run source_database -t dev --notebook-params mode=seed     # once
databricks bundle run source_cdc      -t dev                                  # snapshot
databricks bundle run source_database -t dev --notebook-params mode=mutate   # 3 changes
databricks bundle run source_cdc      -t dev                                  # converge
```

- [ ] **Step 4: Update the Status section**

Move SQL Server ingestion out of "Not done" and into "Done", worded as the simulation it is,
with a pointer to `docs/001-source-cdc-simulation/spec.md` for the swap path.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "Document the simulated source database and CDC run order"
```

---

## Verification checklist

Run after Task 6. Every line maps to a success criterion in the spec.

| # | Check | Criterion |
|---|---|---|
| 1 | seed → run → mutate → run needs only the four documented commands | SC-001 |
| 2 | every bronze table matches its source on count and primary keys | SC-002 |
| 3 | the transcript's three queries return 1 / 1 / 0 | SC-003 |
| 4 | no bronze table has a `_change_type` / `_commit_version` / `_commit_timestamp` column | SC-004 |
| 5 | `smart_claims_dev_etl.pipeline.yml` is unchanged from Part 1 (`git diff main -- ` is empty) | SC-005 |
| 6 | `databricks clusters list` returns zero clusters | SC-006 |
