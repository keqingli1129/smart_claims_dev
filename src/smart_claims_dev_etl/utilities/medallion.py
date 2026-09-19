"""Schema plumbing for the Part 4 transformations pipeline.

The `transformations` pipeline has its default schema set to SILVER, so a silver dataset can
be declared with a bare name and a gold dataset reads its siblings by bare name too. What that
default does NOT give us is a way to name the other two layers, so both come in through
pipeline configuration:

    bronze_schema   fully qualified (`catalog.schema`), because a two-part reference would
                    resolve against the pipeline's own catalog+schema -- i.e. silver.
    gold_schema     schema name only. Gold lives in the same catalog as the pipeline, and a
                    fully-qualified `name=` on a dataset is how one pipeline publishes into a
                    second schema (see the multi-schema pattern in the pipelines docs).

Both are resolved through bundle RESOURCES rather than variables, so development mode's
dev_<username>_ prefix is picked up without anything here knowing it exists.

This module lives in utilities/ for the same reason cdc_source.py and object_storage_source.py
do: a pipeline's glob executes every file it matches, and this file declares no datasets.
utilities/ sits outside all four pipelines' globs, which is also what lets each import from it.
"""


def conf(spark, key: str, default: str = "") -> str:
    """Pipeline configuration value, defaulting rather than raising when unset.

    Same guard the other two source modules use. A bare spark.conf.get raises an opaque error
    if this file is ever executed by a pipeline that does not set the key -- which is exactly
    what a mis-scoped glob would cause.
    """
    try:
        return spark.conf.get(key, default)
    except Exception:
        return default


def _require(spark, key: str) -> str:
    value = conf(spark, key)
    if not value:
        raise ValueError(
            f"{key} is not set. It is supplied by the transformations pipeline's "
            "`configuration` block; this module is not meant to run in another pipeline."
        )
    return value


def bronze(spark, table: str) -> str:
    """Fully-qualified name of a bronze table produced by Parts 1-3."""
    return f"{_require(spark, 'bronze_schema')}.{table}"


def gold(spark, table: str) -> str:
    """Name to publish a gold dataset under, from a pipeline whose default schema is silver."""
    return f"{_require(spark, 'gold_schema')}.{table}"


def read_bronze_stream(spark, table: str):
    """Streaming read of an APPEND-ONLY bronze table.

    Only safe for the tables nothing ever rewrites: `telematics`, `training_images` and
    `claims_images_metadata`, all of which are plain Auto Loader / stream appends.

    The three CDC tables are deliberately NOT read this way -- see read_bronze_batch.
    """
    return spark.readStream.table(bronze(spark, table))


def read_bronze_batch(spark, table: str):
    """Batch read of a bronze table that gets rewritten in place.

    `claim`, `customer` and `policy` are AUTO CDC targets running SCD Type 1: an update
    overwrites the row and a delete removes it. A structured-streaming read of a table whose
    files are rewritten fails outright ("Detected a data update/delete in the source table"),
    and the usual escape hatch -- `.option("skipChangeCommits", "true")` -- buys success by
    IGNORING exactly the commits Part 2 exists to demonstrate. Silver would then never see
    CLM-000001 flip from Total Loss to Minor Damage, and the deleted customer would live on.

    So these three become materialized views instead of streaming tables. An MV recomputes (or,
    on serverless with row tracking, incrementally refreshes) against the current state of its
    source, which is the only shape that tells the truth about an SCD Type 1 upstream.
    """
    return spark.read.table(bronze(spark, table))
