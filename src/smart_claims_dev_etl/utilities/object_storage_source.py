"""Auto Loader reads for the object-storage datasets.

Part 3 ingests files from a volume rather than a stream or a change feed. Auto Loader
(`format("cloudFiles")`) is what makes that incremental: it tracks which files it has already
seen, so re-running a pipeline ingests only what is new. That is the question the transcript
opens with -- "how do I figure out which files have I already processed?" -- and the answer is
that you do not, because Auto Loader already did.

Inside a declarative pipeline the schema and checkpoint locations are supplied by the pipeline
itself. Outside one they must be set explicitly; see jobs/archive_claim_images.py.

This module lives in utilities/ rather than transformations_files/ because a pipeline's glob
executes every file it matches, and this file declares no datasets. utilities/ sits outside all
three pipelines' globs, which is also what lets each import from it.
"""

from pyspark.sql import functions as F


def _conf(spark, key: str, default: str = "") -> str:
    """Pipeline configuration value, defaulting rather than raising when unset."""
    try:
        return spark.conf.get(key, default)
    except Exception:
        return default


def read_files(spark, path_key: str, file_format: str, options: dict = None):
    """Incrementally read one volume path with Auto Loader.

    `path_key` names a pipeline configuration entry rather than taking a literal path, so the
    development-mode schema rename is resolved at deploy time rather than hardcoded here.

    `_metadata.file_path` is carried on every row. It is the modern replacement for the
    deprecated `input_file_name()`, and it makes a row traceable back to the file it came from
    -- which matters once cleanSource starts moving those files away.
    """
    path = _conf(spark, path_key)
    if not path:
        raise ValueError(
            f"{path_key} is not set. It is supplied by the object_storage pipeline's "
            "`configuration` block; this module is not meant to run in another pipeline."
        )

    reader = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", file_format)
    )
    for key, value in (options or {}).items():
        reader = reader.option(key, value)

    return reader.load(path).withColumn("source_file", F.col("_metadata.file_path"))
