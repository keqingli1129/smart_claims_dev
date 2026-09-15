# Feature Specification: Simulated SQL Server CDC Ingestion

**Created**: 2026-09-14
**Status**: Draft
**Source material**: `docs/transcript_2.txt` (Part 2 — SQL Server via Lakeflow Connect)

## Why this is a simulation

Part 2 of the tutorial ingests three tables from a SQL Server on AWS RDS using Lakeflow
Connect. That path is unavailable here for two independent reasons:

1. **No SQL Server.** Lakeflow Connect reads a real transaction log; unlike Part 1's Kinesis
   gap, there is nothing to simulate at the connector boundary.
2. **The ingestion gateway is classic compute.** The transcript says so explicitly ("this is
   currently not available yet in serverless, so this will be always a classic compute VM").
   This workspace has zero clusters. If it is Free Edition, the gateway cannot run at all,
   regardless of where the database lives.

This feature therefore reproduces the *transferable* half of Part 2 — change-data-capture
semantics — on serverless compute, and is built so the managed connector can replace it later
without disturbing anything downstream.

## Swap boundary

The replaceable unit is the **bronze destination contract**: table names, column schemas, and
CDC semantics. It is NOT the staging volume — Lakeflow Connect's ingestion pipeline is also
managed, so no code of ours would ever read the staged change events, and their format is
internal and undocumented.

When Lakeflow Connect becomes available, the migration is:

- delete `source_database` (job), `source_cdc` (pipeline), and the `source` schema
- add a `gateway_definition` pipeline and an `ingestion_definition` pipeline targeting the
  same bronze schema

`bronze.customer`, `bronze.policy` and `bronze.claim` keep their names and schemas. Nothing
downstream of bronze changes.

---

## User Scenarios

### User Story 1 — A source database that behaves like SQL Server (Priority: P1)

A `source` schema holds three Delta tables standing in for the SQL Server database, with the
Delta change data feed enabled — the direct analogue of the transcript's
`sys.sp_cdc_enable_table`. A job seeds them and later mutates them with ordinary
INSERT / UPDATE / DELETE, exactly as the transcript does through DataGrip.

**Why P1**: nothing can be ingested until there is something to ingest, and the mutations are
what make the CDC claim testable at all.

**Acceptance scenarios**:

1. Given no source tables, when the job runs with `mode=seed`, then `source.customer`,
   `source.policy` and `source.claim` exist with 7,000 / 12,000 / 13,000 rows and
   `delta.enableChangeDataFeed = true`.
2. Given seeded tables, when the job runs with `mode=seed` again, then it refuses and exits
   non-zero, naming `--force` and the full-refresh consequence.
3. Given seeded tables, when the job runs with `mode=mutate`, then one policy is inserted, one
   claim's `incident_severity` becomes `Minor Damage`, and one customer is deleted — each
   committed separately.
4. Given a mutation has run, when the source change feed is queried, then exactly three change
   records are visible beyond the seed.

### User Story 2 — Changes converge into bronze (Priority: P1)

A pipeline reads each source table's change feed and applies it to a bronze streaming table
with `create_auto_cdc_flow`, SCD Type 1. First run replays the seed as a snapshot; later runs
apply only what changed.

**Why P1**: this is the feature. US1 without it produces no lakehouse data.

**Acceptance scenarios**:

1. Given seeded source tables and no bronze tables, when the pipeline runs, then
   `bronze.customer`, `bronze.policy` and `bronze.claim` hold 7,000 / 12,000 / 13,000 rows.
2. Given the pipeline has run, when its output schema is inspected, then no `_change_type`,
   `_commit_version` or `_commit_timestamp` column is present.
3. Given `mode=mutate` has run, when the pipeline runs again, then the inserted policy appears,
   the updated claim shows `Minor Damage`, and the deleted customer returns zero rows.
4. Given any state, when each bronze table is compared to its source table, then row counts and
   primary keys match exactly.
5. Given the telematics tables from Part 1, when `bronze.policy` is joined to
   `bronze.telematics` on `chassis_number`, then the join returns rows.

### User Story 3 — SCD Type 2 history (Priority: P3, deferred)

The same feed applied with `stored_as_scd_type=2`, retaining history via `__START_AT` /
`__END_AT`. **Explicitly out of scope for this iteration.** Recorded so the SCD Type 1 choice
reads as a decision rather than an oversight.

---

## Requirements

### Source database (US1)

- **FR-001**: A `source` schema MUST be declared in `resources/catalog.yml` alongside the
  medallion layers, and MUST be subject to development-mode renaming like the others.
- **FR-002**: The three source tables MUST be created **empty** with
  `TBLPROPERTIES (delta.enableChangeDataFeed = true)`, and the seed rows inserted as a
  separate commit afterwards. A `CREATE TABLE ... AS SELECT` MUST NOT be used: the change feed
  is guaranteed only for commits made *after* the property is set, and a CTAS makes the data
  commit and the enabling commit the same version. Creating empty first removes the ambiguity,
  so the pipeline's first run reliably sees the full seed as inserts.
- **FR-003**: `mode=seed` MUST refuse to run against existing tables unless `--force` is
  given, and MUST state that a forced reseed requires a full refresh of the CDC pipeline.
- **FR-004**: `mode=mutate` MUST default to the transcript's three operations (insert policy,
  update claim severity, delete customer) and MUST accept an option for bulk random churn.
- **FR-005**: Each mutation MUST be committed separately, so `_commit_version` orders changes
  unambiguously. A single commit MUST NOT contain two changes to the same primary key.
- **FR-006**: Generated data MUST be deterministic under a seed parameter, so a reseed
  reproduces the same rows.

### CDC ingestion (US2)

- **FR-007**: `source_cdc` MUST be a separate pipeline resource from `smart_claims_dev_etl`,
  so that replacing it with Lakeflow Connect is a deletion rather than surgery.
- **FR-008**: Each dataset MUST read its source via
  `spark.readStream.option("readChangeFeed", "true")` and MUST exclude
  `_change_type = 'update_preimage'` rows.
- **FR-009**: The CDF read MUST be exposed as a `@dp.temporary_view`, because
  `create_auto_cdc_flow`'s `source` accepts an identifier string and rejects a DataFrame.
- **FR-010**: Each flow MUST use `sequence_by="_commit_version"`,
  `apply_as_deletes=expr("_change_type = 'delete'")` and `stored_as_scd_type=1`.
- **FR-011**: Each flow MUST pass `except_column_list` covering `_change_type`,
  `_commit_version` and `_commit_timestamp`, so bronze matches what Lakeflow Connect would
  produce.
- **FR-012**: The resolved `source` schema name MUST reach pipeline code through pipeline
  `configuration`, following the `landing_volume_path` precedent, so development-mode renaming
  is handled.
- **FR-013**: The pipeline MUST write into the same bronze schema as the telematics pipeline.

### Repository structure

- **FR-014**: The CDC datasets MUST live in `src/smart_claims_dev_etl/transformations_dev/`,
  a sibling of `transformations/`. Part 1's glob (`transformations/**`) requires that exact
  directory name followed by a separator and therefore never matches the sibling, so the two
  pipelines stay disjoint without any change to Part 1.
- **FR-015**: `resources/smart_claims_dev_etl.pipeline.yml` and the files under
  `transformations/` MUST NOT be modified. Verified by `git diff main` reporting no change to
  either. (Supersedes an earlier design that moved Part 1's files to make the layout
  symmetric; the sibling folder achieves the same isolation without touching working code.)
- **FR-017**: Both pipelines MUST keep the same `root_path`, so `utilities/` remains importable
  from each. A separate `root_path` would require duplicating the shared module.
- **FR-016**: README MUST document the source-database run order, the swap path to Lakeflow
  Connect, and why the simulation exists.

## Key Entities

| Entity | Rows | Primary key | Links |
|---|---|---|---|
| `customer` | 7,000 | `customer_id` | — |
| `policy` | 12,000 | `policy_number` | `customer_id` |
| `claim` | 13,000 | `claim_number` | `policy_number` |

Row counts match the transcript's reported upserts (13,000 claims, ~7,000 customers, 12,000
policies).

`policy.chassis_number` is the join back to Part 1's telematics data, spanning
`CH-00001`–`CH-12000`. Part 1's ten-vehicle fleet is a subset: most policies carry no
telematics device, which is both realistic and leaves Part 1 untouched.

`claim.incident_severity` uses the transcript's vocabulary — `Total Loss`, `Major Damage`,
`Minor Damage` — so its demonstration (a claim flipped from total loss to minor damage)
transfers verbatim.

## Success Criteria

- **SC-001**: From a clean deploy, seed → run → mutate → run completes with no manual step
  beyond the four documented commands.
- **SC-002**: After any run, every bronze table matches its source table on row count and
  primary-key set.
- **SC-003**: The transcript's three verification queries return the same answers they do in
  the video.
- **SC-004**: Bronze table schemas contain no change-feed metadata columns.
- **SC-005**: Part 1 is untouched — `git diff main` shows no change to
  `resources/smart_claims_dev_etl.pipeline.yml` or anything under `transformations/`.
- **SC-006**: Everything runs on serverless compute. No classic cluster is created.

## Testing

**No automated tests are required for this feature.** Decided explicitly at specification
time. Part 1 established no test coverage for pipeline code, and the transformations cannot run
outside a pipeline context. Verification is by the acceptance scenarios above, executed by hand
against the deployed workspace.

Reviewers MUST NOT raise missing-test findings against this feature.

## Assumptions

- The workspace remains serverless-only. If classic compute becomes available, this feature is
  superseded rather than extended — see the swap boundary above.
- `docs/transcript_2.txt` is the authority for source table names, row counts and severity
  values.
- Delta change-feed retention (30 days, tied to table history) is sufficient; longer gaps
  require a reseed.

## Out of scope

- SCD Type 2 (User Story 3)
- Silver and gold layers
- Any real SQL Server, connection object, or ingestion gateway
- Any modification to Part 1 whatsoever
