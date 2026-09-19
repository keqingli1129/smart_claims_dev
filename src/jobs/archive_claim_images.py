# Databricks notebook source
# MAGIC %md
# MAGIC # Archive claim images with cleanSource
# MAGIC
# MAGIC Transcript section: *"Archiving with cleanSource"*.
# MAGIC
# MAGIC Reads the accident photos customers uploaded, writes them to `bronze.claim_images`, and
# MAGIC has Auto Loader move each file into `archive/` once it is done with it. The transcript's
# MAGIC reason for wanting that: *"this could be just a place where we archive everything and
# MAGIC this could be another storage tier that is cheaper on the cloud level"* -- or a holding
# MAGIC pen before deletion, so the same bytes are not paid for twice.
# MAGIC
# MAGIC ## Why this is not a pipeline
# MAGIC
# MAGIC *"In this case we will not use the declarative pipelines anymore but we will use plain
# MAGIC pispar code for this."*
# MAGIC
# MAGIC That is the lesson, not an inconvenience. Compare this file with
# MAGIC `transformations_files/training_images.py`: same Auto Loader read, same `binaryFile`
# MAGIC format, same volume-shaped source. That one is eleven lines because the pipeline owns
# MAGIC the dataset and supplies its state. Here nothing owns anything, so every piece has to be
# MAGIC named:
# MAGIC
# MAGIC | Supplied by a pipeline | Named by hand here |
# MAGIC |---|---|
# MAGIC | schema location | `cloudFiles.schemaLocation` |
# MAGIC | checkpoint location | `checkpointLocation` |
# MAGIC | the target table | `.toTable(...)` |
# MAGIC | when the stream stops | `.trigger(availableNow=True)` + `awaitTermination()` |
# MAGIC
# MAGIC ## Why one run is not enough
# MAGIC
# MAGIC The transcript hits this and shrugs it off as timing -- *"sometimes you need to
# MAGIC re-execute this pipeline ... if this is taking less than a minute, then it would probably
# MAGIC not move the files"*. It is more structural than that. Two rules compound:
# MAGIC
# MAGIC 1. **Clean source only runs when there is a batch to process.** It is not a background
# MAGIC    sweeper. A run that finds no new files does no cleanup either, however long you wait.
# MAGIC 2. **A file is eligible only once its commit time is recorded and the retention duration
# MAGIC    has elapsed since** -- so the earliest a file can move is two runs after the one that
# MAGIC    ingested it.
# MAGIC
# MAGIC Together those mean the archive stays empty until a *later run that also has new files to
# MAGIC ingest*. Hence `generate_object_storage_files.py`'s `more_images` mode: it exists to give
# MAGIC run N+2 a batch, so cleanup has a reason to fire. The run order is in the README.
# MAGIC
# MAGIC `cloudFiles.cleanSource.waitForCompletion` keeps the stream alive until the moves finish
# MAGIC rather than letting `availableNow` exit the moment the last row is written. Without it a
# MAGIC run can end with eligible files still sitting in place.

# COMMAND ----------

import os

from pyspark.sql import functions as F

dbutils.widgets.text("source_path", "", "Source: claim images folder")
dbutils.widgets.text("archive_path", "", "Destination for processed files")
dbutils.widgets.text("checkpoint_path", "", "Schema + checkpoint location")
dbutils.widgets.text("target_table", "", "Fully qualified bronze table")
dbutils.widgets.dropdown("clean_source", "MOVE", ["MOVE", "DELETE", "OFF"], "cleanSource mode")
dbutils.widgets.text("retention_duration", "1 minute", "Retention before cleanup")

SOURCE_PATH = dbutils.widgets.get("source_path").rstrip("/")
ARCHIVE_PATH = dbutils.widgets.get("archive_path").rstrip("/")
CHECKPOINT_PATH = dbutils.widgets.get("checkpoint_path").rstrip("/")
TARGET_TABLE = dbutils.widgets.get("target_table").strip()
CLEAN_SOURCE = dbutils.widgets.get("clean_source")
RETENTION_DURATION = dbutils.widgets.get("retention_duration").strip()

if not all([SOURCE_PATH, ARCHIVE_PATH, CHECKPOINT_PATH, TARGET_TABLE]):
    raise ValueError("source_path, archive_path, checkpoint_path and target_table are all required")

# Both are enforced by Auto Loader, and both fail late and confusingly if you get them wrong,
# so they are checked here where the message can say why.
#
# The archive cannot sit inside the source: Auto Loader would rediscover the files it had just
# moved and ingest them a second time. Nor can it sit in a different volume or bucket --
# cleanSource issues a move within one storage location and does not copy across them.
if ARCHIVE_PATH.startswith(f"{SOURCE_PATH}/"):
    raise ValueError(
        f"archive_path must not be inside source_path, or the archived files get re-ingested.\n"
        f"  source:  {SOURCE_PATH}\n  archive: {ARCHIVE_PATH}"
    )
if CHECKPOINT_PATH.startswith(f"{SOURCE_PATH}/"):
    raise ValueError(
        f"checkpoint_path must not be inside source_path, or Auto Loader reads its own state "
        f"files as input.\n  source:     {SOURCE_PATH}\n  checkpoint: {CHECKPOINT_PATH}"
    )

# DELETE is irreversible and Auto Loader refuses a retention under seven days for it. MOVE has
# no such floor, which is what lets the transcript demonstrate the whole thing in one minute.
if CLEAN_SOURCE == "DELETE":
    print("WARNING: cleanSource=DELETE removes processed files permanently. Retention must exceed 7 days.")

print(f"source     {SOURCE_PATH}")
print(f"archive    {ARCHIVE_PATH}")
print(f"checkpoint {CHECKPOINT_PATH}")
print(f"table      {TARGET_TABLE}")
print(f"cleanSource {CLEAN_SOURCE}, retention {RETENTION_DURATION}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Before
# MAGIC
# MAGIC Listing both folders first is what makes the run self-evidencing -- the transcript's
# MAGIC *"we see our files have actually been moved right here"* is a before-and-after, and only
# MAGIC one half of it is visible after the fact.

# COMMAND ----------


def listing(path: str) -> list:
    """Filenames in a volume folder, or an empty list if it does not exist yet."""
    try:
        return sorted(f for f in os.listdir(path) if not f.startswith("_"))
    except FileNotFoundError:
        return []


before_source = listing(SOURCE_PATH)
before_archive = listing(ARCHIVE_PATH)

print(f"source  {len(before_source):3d} files  {before_source[:5]}{' ...' if len(before_source) > 5 else ''}")
print(f"archive {len(before_archive):3d} files  {before_archive[:5]}{' ...' if len(before_archive) > 5 else ''}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## The read, the write, and the cleanup
# MAGIC
# MAGIC `binaryFile` gives one row per file: `path`, `modificationTime`, `length`, `content`.
# MAGIC `_metadata.file_path` is carried through as `source_file` for the same reason the pipeline
# MAGIC datasets carry it -- once cleanSource starts moving files, `path` points at where the
# MAGIC file *was*, and a row needs to stay traceable.

# COMMAND ----------

stream = (
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "binaryFile")
    # Supplied for free inside a pipeline. Out here it is mandatory, and pointing it at the
    # same directory as the checkpoint keeps one run's entire state in one place.
    .option("cloudFiles.schemaLocation", CHECKPOINT_PATH)
    # --- cleanSource: the new part -------------------------------------------------------
    .option("cloudFiles.cleanSource", CLEAN_SOURCE)
    .option("cloudFiles.cleanSource.moveDestination", ARCHIVE_PATH)
    .option("cloudFiles.cleanSource.retentionDuration", RETENTION_DURATION)
    # availableNow otherwise ends the stream as soon as the last row is written, which can be
    # before the moves have happened. This holds it open until cleanup is done.
    .option("cloudFiles.cleanSource.waitForCompletion", "true")
    .load(SOURCE_PATH)
    .withColumn("source_file", F.col("_metadata.file_path"))
)

query = (
    stream.writeStream.option("checkpointLocation", CHECKPOINT_PATH)
    # Batch, not a continuously running stream: process what is here, then stop.
    .trigger(availableNow=True)
    .toTable(TARGET_TABLE)
)

query.awaitTermination()
print(f"stream finished: {query.lastProgress.get('numInputRows') if query.lastProgress else 0} rows this run")

# COMMAND ----------

# MAGIC %md
# MAGIC ## After
# MAGIC
# MAGIC An empty archive here is the expected result of the first run or two, not a failure --
# MAGIC see "Why one run is not enough" at the top.

# COMMAND ----------

after_source = listing(SOURCE_PATH)
after_archive = listing(ARCHIVE_PATH)

moved = sorted(set(before_source) - set(after_source))

print(f"source  {len(before_source):3d} -> {len(after_source):3d}")
print(f"archive {len(before_archive):3d} -> {len(after_archive):3d}")

if moved:
    print(f"\ncleanSource moved {len(moved)} files: {moved}")
else:
    print(
        "\nNothing moved this run. Expected unless BOTH held:\n"
        f"  - this run had new files to ingest (cleanSource only runs alongside a batch)\n"
        f"  - {RETENTION_DURATION} has passed since an earlier run recorded their commit time\n"
        "Add files with `object_storage_generator --notebook-params mode=more_images`, then run again."
    )

print(f"\n{TARGET_TABLE}: {spark.table(TARGET_TABLE).count()} rows")
