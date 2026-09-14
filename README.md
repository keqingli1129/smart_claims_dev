# smart_claims_dev

A motor-insurance claims lakehouse, built incrementally from a Databricks end-to-end tutorial
(transcript in `docs/transcript.txt`). The finished system is meant to take a claim, pull in
telematics from the vehicle, policy data from SQL Server and damage photos from object storage,
classify the damage with an ML model, and surface the result through dashboards, Genie, and a
Databricks App over Lakebase.

**What exists today is Part 1: streaming telematics ingestion into bronze.** Everything else is
scaffolding or empty. See [Status](#status) for the precise line.

The whole thing is a Declarative Automation Bundle — catalog, schemas, volume, pipeline and jobs
are all declared in `resources/` and deployed with `databricks bundle deploy`. A clean checkout
reproduces the entire structure; nothing was clicked into existence.


## Layout

```
resources/            Everything deployable, one file per concern
  catalog.yml           Schemas + the landing volume (NOT the catalog -- see below)
  *.pipeline.yml        The telematics pipeline
  telematics_generator.job.yml   Simulated producer
  sample_job.job.yml    Template leftover, see Status
src/
  jobs/generate_telematics.py        Writes Kinesis-shaped records into the landing volume
  smart_claims_dev_etl/
    utilities/telematics_source.py   One reader, four sources, one column contract
    transformations/                 One file per dataset; the pipeline globs this folder
  smart_claims_dev/                  Shared package, installed as a wheel
docs/transcript.txt   Source material for Part 1
tests/                Unit tests for the shared package
```

`utilities/` sits outside `transformations/` deliberately: the pipeline's glob executes
everything under `transformations/**`, and a module that defines no datasets must not be in
there.


## Unity Catalog layout

```
smart_claims_dev
├── landing   volume `telematics_raw`  -- files, no tables
├── bronze    telematics, telematics_test
├── silver    empty until a later part
└── gold      empty until a later part
```

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


## The two bronze tables

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
- Streaming ingestion into two bronze tables, four interchangeable sources
- Continuous-pipeline demo, verified at 1720 rows

Not done — later parts of the tutorial:

- SQL Server ingestion via Lakeflow Connect
- Damage photos from object storage via Auto Loader
- Damage-classification model, AI/BI dashboards, Genie, Lakebase, Databricks Apps
- `silver` and `gold` are empty

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
