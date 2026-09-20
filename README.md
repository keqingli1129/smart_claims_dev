# smart_claims_dev

A motor-insurance claims lakehouse, built incrementally from a Databricks end-to-end tutorial
(transcripts in `docs/`). The finished system is meant to take a claim, pull in
telematics from the vehicle, policy data from SQL Server and damage photos from object storage,
classify the damage with an ML model, and surface the result through dashboards, Genie, and a
Databricks App over Lakebase.

**What exists today is parts 1 and 2: five bronze tables.** Telematics arrives as a stream;
customer, policy and claim arrive as change data from a stand-in for the SQL Server the tutorial
uses, because the real connector needs compute this workspace does not have. Silver and gold are
empty. See [Status](#status) for the precise line.

The whole thing is a Declarative Automation Bundle — catalog, schemas, volume, pipelines and
jobs are all declared in `resources/` and deployed with `databricks bundle deploy`. A clean checkout
reproduces the entire structure; nothing was clicked into existence.


## Layout

```
resources/            Everything deployable, one file per concern
  catalog.yml           Schemas + the landing volume (NOT the catalog -- see below)
  smart_claims_dev_etl.pipeline.yml   Telematics streaming ingestion    (part 1)
  telematics_generator.job.yml        Simulated Kinesis producer        (part 1)
  source_cdc.pipeline.yml             CDC ingestion into bronze         (part 2)
  source_database.job.yml             Simulated SQL Server, seed+mutate (part 2)
  train_damage_classifier.job.yml     ResNet fine-tune + UC register     (part 5)
  sample_job.job.yml    Template leftover, see Status
src/
  jobs/
    generate_telematics.py           Kinesis-shaped records into the landing volume
    simulate_source_database.py      Seeds and mutates the simulated SQL Server
    train_damage_classifier.py       Part 5: fine-tune, track, register, batch score
  smart_claims_dev_etl/
    utilities/
      telematics_source.py           One reader, four sources, one column contract
      cdc_source.py                  Change-feed read + AUTO CDC flow
    transformations/                 Globbed by the TELEMATICS pipeline
    transformations_dev/             Globbed by the CDC pipeline
  smart_claims_dev/                  Shared package, installed as a wheel
docs/
  transcript_1.txt      Source material for part 1
  transcript_2.txt      Source material for part 2
  001-source-cdc-simulation/    Spec, plan, and the Lakeflow Connect swap file
tests/                  Unit tests for the shared package
```

**Two pipelines, two folders.** A pipeline executes every file its glob matches, so the two
pipelines' datasets must not share a directory -- otherwise both would declare the same tables,
and whichever pipeline runs second fails because another already owns them. `transformations/`
and `transformations_dev/` are siblings rather than nested, because `transformations/**`
requires that exact directory name followed by a separator and so never reaches the sibling.
That is what let part 2 be added without editing part 1 at all. (`**` is not optional: the
pipelines API rejects a single asterisk outright, so `transformations/*.py` is not a route to
the same isolation.)

`utilities/` sits outside both globs deliberately -- it defines no datasets, and being outside
is also what lets both pipelines import from it. The two pipelines share one `root_path`, which
is what puts it on their import path; separate roots would mean a second copy.


## Unity Catalog layout

```
smart_claims_dev
├── landing   volume `telematics_raw`  -- files, no tables
├── source    customer, policy, claim  -- the simulated SQL Server
├── bronze    telematics, telematics_test, customer, policy, claim
├── silver    empty until a later part
└── gold      empty until a later part
```

`source` is not a medallion layer. It stands for a system *outside* the lakehouse -- the SQL
Server part 2 ingests from -- and it disappears the day Lakeflow Connect replaces it.

`landing` holds files that have not become Delta yet; the medallion layers hold tables. In the
`dev` target every schema name is rewritten to `dev_<your_username>_<layer>`, so two people can
deploy into the same catalog without collision.

**One manual prerequisite.** This account has Unity Catalog Default Storage enabled, and catalog
creation through the REST API is refused (`400 INVALID_STATE`). The same statement succeeds
through SQL, so create the catalog once before the first deploy:

```sql
CREATE CATALOG IF NOT EXISTS smart_claims_dev;
```

Consequence: `bundle destroy` removes the schemas and volume but leaves the catalog behind. The
full reasoning is in `resources/catalog.yml`.


## Run order

Deploy, produce, then ingest — in that order, because the pipeline reads what the generator
writes.

```bash
# 1. Deploy the bundle (dev is the default target)
databricks bundle deploy -t dev --profile <PROFILE>

# 2. Produce records into the landing volume
databricks bundle run telematics_generator -t dev --profile <PROFILE>

# 3. Ingest them
databricks bundle run smart_claims_dev_etl -t dev --profile <PROFILE>
```

Step 2's defaults write 40 records (5 batches x 8), matching the 40 already sitting in the
transcript's stream. Three parameters control it:

| Parameter | Default | Effect |
|---|---|---|
| `num_batches` | 5 | Number of files, i.e. number of Auto Loader micro-batches |
| `events_per_batch` | 8 | Records per file |
| `sleep_seconds` | 3 | Gap between files -- paces the *producer*, not the pipeline |

Records written = `num_batches x events_per_batch`. Wall clock is roughly
`(num_batches - 1) x sleep_seconds`, running a few percent long because `sleep` is a floor.

`sleep_seconds` matters more than it looks. Set it to 0 and every file lands at once, Auto Loader
swallows the lot in one or two micro-batches, and the row count jumps in a single step — same
data, no visible streaming.

### Part 2 — source database, then CDC

Same shape: produce, then ingest. Steps 1 and 2 are once; 3 and 4 are the loop.

```bash
# 1. Seed the source tables (7,000 customers / 12,000 policies / 13,000 claims)
databricks bundle run source_database -t dev --profile <PROFILE> --notebook-params mode=seed

# 2. Ingest the snapshot
databricks bundle run source_cdc -t dev --profile <PROFILE>

# 3. Change the source -- insert a policy, update a claim, delete a customer
databricks bundle run source_database -t dev --profile <PROFILE> --notebook-params mode=mutate

# 4. Propagate the changes
databricks bundle run source_cdc -t dev --profile <PROFILE>
```

Add `churn=50` to step 3 for bulk change volume rather than three rows.

Step 1 refuses if the tables already exist. That is deliberate: reseeding drops and recreates
them, which restarts their change feed at version 0 and invalidates the pipeline's checkpoint.
Override with `force=true` only alongside a full refresh of `source_cdc`.

### Part 5 — labelled images, then train

Part 5 needs the training images to carry their label in the filename
(`train_0007_minor_damage.png`), which images generated before part 5 do not. So steps 1 and 2
are only needed once, on a volume seeded earlier:

```bash
# 1. Re-seed the training images, this time with labels in their names
databricks bundle run object_storage_generator -t dev --profile <PROFILE> --notebook-params mode=seed

# 2. Ingest them
databricks bundle run object_storage -t dev --profile <PROFILE>

# 3. Fine-tune, register to UC, and batch score
databricks bundle run train_damage_classifier -t dev --profile <PROFILE>
```

**The old images do not need deleting.** They stay in the volume and in bronze, which is what
bronze is for. Silver's `labelled` expectation drops them — an unlabelled filename extracts to
`""` and fails the check — so `silver.training_images` ends up holding only the labelled set,
and step 3's join to bronze sees only those. Nothing has to be cleaned up by hand.

Step 3 is the one job in this bundle on a **classic** cluster rather than serverless (a
single-node `16.4.x-cpu-ml-scala2.12`, pinned), because the ML runtime pre-installs torch and
scikit-learn. Budget roughly five minutes of cluster start plus ten of training. It also needs
outbound network access to download the `microsoft/resnet-50` checkpoint from Hugging Face.

**About the accuracy:** expect roughly chance level, and that is correct. The generator draws
each image's colour with `rng.randint`, independently of the label it writes into the filename,
so the pixels carry no signal for any model to find. The pipeline is real — real fine-tune, real
MLflow tracking, real UC registration, real batch scoring — and the data is not. Point it at
real labelled photographs and the same code produces a real model. The notebook prints this
next to the confusion matrix so nobody reads the matrix as a result.


## The two telematics tables

Both read the same stream and hold the same number of rows. The difference is the point.

| Table | `data` column | Defined in |
|---|---|---|
| `telematics_test` | raw bytes, unreadable | `transformations/telematics_simple.py` |
| `telematics` | decoded into typed columns | `transformations/telematics_parsed.py` |

Keeping both is a deliberate departure from the transcript, which deletes `telematics_test` once
it has made its point. Side by side they are the before-and-after of the parsing lesson.

Everything in `telematics` is typed `STRING` on purpose — real typing belongs in silver. The
payload is parsed as `MAP<STRING, STRING>` so the producer can add fields without the schema
knowing them up front, and `raw_json` is kept so nothing is lost if it does.


## The simulated source database

Part 2 of the tutorial ingests three SQL Server tables with **Lakeflow Connect**: an ingestion
gateway reads the database's transaction log into a volume, and a managed ingestion pipeline
upserts from there into bronze. Neither half is available here.

- There is no SQL Server. Unlike part 1's missing Kinesis stream, there is nothing to simulate
  at the connector boundary -- Lakeflow Connect's whole job is reading a real transaction log.
- The gateway runs on **classic compute**, which the transcript states outright: *"currently
  not available yet in serverless, so this will be always a classic compute VM."* This
  workspace has none.

So this project reproduces the part that transfers -- **CDC semantics** -- on serverless:

```
source_database job              source schema (stands in for SQL Server)
  mode=seed    → create+load     customer 7,000 · policy 12,000 · claim 13,000
  mode=mutate  → INSERT/UPDATE/  delta.enableChangeDataFeed = true
                 DELETE                        │
                                               │  Delta change feed
                                               ▼
                                 source_cdc pipeline
                                   temp view reads the feed
                                   create_auto_cdc_flow, SCD Type 1
                                               │
                                               ▼
                             bronze.customer · bronze.policy · bronze.claim
```

Enabling the change feed on a Delta table is the exact analogue of the transcript's
`sys.sp_cdc_enable_table` on SQL Server. The feed is real, so nothing downstream is faked --
only its origin differs.

**How the pieces earn their keep**

`_commit_version` is Delta's answer to a log sequence number, and it is what `sequence_by`
orders changes on. That is why the mutate job issues each change as its own statement: one
commit per change, unambiguously ordered.

`create_auto_cdc_flow` takes the *name* of a view, not a DataFrame, so every dataset declares a
`@dp.temporary_view` over the change feed first and passes its name.

`except_column_list` drops `_change_type`, `_commit_version` and `_commit_timestamp`, so bronze
carries exactly the source's columns and nothing else.

Tables are built in three commits -- create empty, enable the feed, then load -- never a
`CREATE TABLE AS SELECT`. Delta guarantees the feed only for commits made *after* the property
is set, and a CTAS collapses those into one version. Get it wrong and the pipeline succeeds
while producing three empty tables, which is worse than a crash.

**SCD Type 1**, so a delete removes the row. That is what makes the transcript's check —
querying the deleted customer and getting nothing — the correct assertion. History belongs in
silver.

**The swap.** `docs/001-source-cdc-simulation/lakeflow-connect-swap.yml` holds the real
Lakeflow Connect resources, parked in `docs/` so the bundle's `resources/*.yml` include never
picks them up. When a SQL Server and classic compute exist, move it into `resources/`, delete
`source_database.job.yml` and `source_cdc.pipeline.yml`, and drop the `source` schema. The
bronze tables keep their names and columns, so nothing downstream of bronze notices. The full
reasoning is in `docs/001-source-cdc-simulation/spec.md`.


## Source modes

The transcript reads an AWS Kinesis stream. All four sources live behind one function in
`utilities/telematics_source.py` and return the **same columns in the Kinesis vocabulary** —
`partitionKey`, `data` (BINARY), `stream`, `shardId`, `sequenceNumber`,
`approximateArrivalTimestamp` — so nothing downstream knows which one ran. Switching is a
variable, not an edit.

| `source_mode` | Needs | Notes |
|---|---|---|
| `volume` *(default)* | nothing | Simulated stream. Auto Loader over the landing volume. |
| `kinesis` | AWS stream + UC service credential | The transcript's source, code unchanged |
| `kafka` | broker list + topic | Secrets from a Databricks secret scope |
| `eventhubs` | namespace + connection string | Kafka protocol on port 9093 |

```bash
databricks bundle deploy -t dev --var="source_mode=kafka" --profile <PROFILE>
```

`volume` is the only mode that uses Auto Loader — it is what makes a directory of files behave
like a message bus, tracking which files it has already seen. Kinesis and Kafka have offsets
natively and need no such thing.

**Credentials never go in pipeline configuration.** Anyone who can view a pipeline can read its
config values, so the `kafka` and `eventhubs` modes take a secret scope name and resolve the
actual secret at runtime.


## Triggered vs continuous

By default the pipeline is **triggered**: an update processes whatever has arrived and stops.
A **continuous** pipeline never finishes — records appear in the tables as they land, and
serverless compute is held until you stop it.

`mode: development` silently strips `continuous: true` out of a pipeline resource, the same
transform that pauses job schedules — an idle laptop should not leave a bill running. Set it in
`*.pipeline.yml` and it is ignored in `dev` and honoured in `prod`, with no warning either way.

To run the continuous demo against the dev pipeline, flip the deployed pipeline itself. The
procedure is written out in full in `resources/smart_claims_dev_etl.pipeline.yml`, next to where
`continuous:` would otherwise go. **The stop at the end is not optional** — until it runs, the
pipeline holds serverless compute whether or not anything is producing.


## Status

Done:

- Unity Catalog structure: catalog, four schemas, landing volume
- Simulated Kinesis producer writing into the landing volume
- Streaming ingestion into two telematics bronze tables, four interchangeable sources
- Continuous-pipeline demo, verified at 1720 rows
- **Part 2, simulated:** CDC from a stand-in source database into three more bronze tables.
  Verified — bronze matches source row-for-row, the transcript's insert/update/delete test
  passes, no change-feed metadata leaks into bronze, and `policy.chassis_number` joins part 1's
  ten-vehicle telematics fleet
- The real Lakeflow Connect resources, written and parked in `docs/`, ready to swap in
- **Part 5:** damage-severity classifier — labels carried in the training filenames and
  extracted into `silver.training_images`, a ResNet fine-tune tracked in MLflow, registered to
  `gold.claims_damage_level@prod`, and batch-scored into `gold.damage_predictions`. Written and
  validated, not yet run end to end. Its accuracy will be chance level by construction — see
  Part 5 under Run order for why that is the honest outcome and not a bug

Not done — later parts of the tutorial:

- Lakeflow Connect against a real SQL Server (blocked: no database, no classic compute)
- SCD Type 2 history on the CDC tables
- Real-time model serving endpoint (the model is registered and aliased, so this is one call —
  left out because an endpoint bills for as long as it exists)
- AI/BI dashboards, Genie, Lakebase, Databricks Apps

This Status section still under-reports parts 3 and 4, which are committed but not described
above.

Template leftovers, kept only because they still deploy cleanly and cost nothing —
delete them when they start getting in the way:

- `sample_job.job.yml`, `src/sample_notebook.ipynb`, `src/smart_claims_dev/taxis.py`
- `transformations/sample_trips_*.py` and `sample_zones_*.py`, which read `samples.nyctaxi`


## Local development on this machine

On a fresh clone elsewhere, run `uv sync --all-groups` first. On this machine it has already
been run, so `.venv/` exists with Databricks Connect, pytest and ruff. Point your IDE at
`./.venv/bin/python` and run/debug/test work directly.

From a terminal there is one wrinkle. This machine's shell exports
`PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages` from the ROS 2 setup script. pytest
scans `PYTHONPATH` for plugin entry points, finds ROS's `launch_testing`, imports it, and dies
on `ModuleNotFoundError: No module named 'yaml'` -- taking every test run with it, Databricks
or not.

`.env` fixes this, along with telling Databricks Connect which compute to use. It is
gitignored, so a fresh clone starts from the tracked template:

```
$ cp .env.sample .env
```

The three settings that matter:

```
PYTHONPATH=
DATABRICKS_SERVERLESS_COMPUTE_ID=auto
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
```

Note that `PYTHONPATH=` only helps inside VS Code. `uv --env-file` follows dotenv semantics, so
a variable your shell already exports wins over the file -- and your shell exports this one.
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is the line that actually works in a terminal, because
nothing has set it. It also targets the real problem more precisely: ROS on `PYTHONPATH` is
harmless right up until pytest scans it for plugins. This project uses no pytest plugins.

To make uv read `.env` on every invocation, export this once in your shell:

```
$ export UV_ENV_FILE=.env
```

Then `uv run pytest` works unadorned. Without it, pass the file explicitly each time:

```
$ uv run --env-file .env pytest
$ uv run --env-file .env python -m smart_claims_dev.main --catalog samples --schema nyctaxi
```

Inside VS Code none of this is needed -- `.vscode/settings.json` sets `python.envFile` and
blanks `PYTHONPATH` for the integrated terminal.

One thing that is *not* runnable locally, by design: the files under
`src/smart_claims_dev_etl/transformations/`. Their `@dp.table` decorators register datasets with
a pipeline that only exists during an update, so running one directly raises `NameError` rather
than building a table. Run those on Databricks instead:

```
$ databricks bundle run smart_claims_dev_etl --refresh <dataset_name>
```


## Using the CLI

Authenticate once, then everything goes through the bundle:

```bash
databricks auth login --host <WORKSPACE_URL> --profile <PROFILE>

databricks bundle validate -t dev --profile <PROFILE>    # resolve + check, no deploy
databricks bundle deploy   -t dev --profile <PROFILE>    # dev is the default target
databricks bundle run <job_or_pipeline> -t dev --profile <PROFILE>
databricks bundle summary  -t dev --profile <PROFILE>    # deployed ids and URLs
```

`bundle validate -o json` is the quickest way to see what a variable actually resolved to —
worth reaching for before assuming a setting took effect.

The `prod` target deploys unprefixed resources and unpauses `sample_job`'s daily schedule. It is
not used yet.

A development copy can be removed with `databricks bundle destroy -t dev --profile <PROFILE>`.
That drops the schemas, the volume and the data in it, but not the catalog.
