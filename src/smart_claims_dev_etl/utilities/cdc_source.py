"""Delta change feed -> bronze, the serverless stand-in for Lakeflow Connect.

Part 2 of the transcript ingests SQL Server through Lakeflow Connect: an ingestion gateway
reads the transaction log into a volume, and a managed ingestion pipeline upserts from there
into bronze. Both halves need classic compute and a real database, and neither is available
here.

What survives the substitution is the part that matters -- CDC semantics. A Delta table with
`delta.enableChangeDataFeed = true` produces a genuine change feed, so nothing below is faked;
only the origin of the feed differs. Swapping in the real connector later means deleting this
module and its datasets and pointing a gateway_definition + ingestion_definition at the same
bronze schema.

This module lives in utilities/ rather than transformations_dev/ because a pipeline's glob
executes every file it matches, and this file defines no datasets. utilities/ sits outside both
pipelines' globs, which is also what lets both import from it.
"""

from pyspark import pipelines as dp
from pyspark.sql import functions as F

# Change-feed bookkeeping. Dropped from the target so bronze matches what Lakeflow Connect
# would produce -- this is what keeps the swap boundary honest rather than approximate.
CDF_METADATA_COLUMNS = ["_change_type", "_commit_version", "_commit_timestamp"]


def _conf(spark, key: str, default: str = "") -> str:
    """Pipeline configuration value, defaulting rather than raising when unset.

    Same guard Part 1 uses. A bare spark.conf.get raises an opaque error if this file is ever
    executed by a pipeline that does not set the key -- which is exactly what a mis-scoped glob
    would cause.
    """
    try:
        return spark.conf.get(key, default)
    except Exception:
        return default


def read_change_feed(spark, table: str):
    """Stream one source table's change feed, after-images only.

    `update_preimage` is the row as it was BEFORE an update. AUTO CDC wants the new value, so
    the before-image is dropped here rather than downstream.
    """
    schema = _conf(spark, "source_schema")
    if not schema:
        raise ValueError(
            "source_schema is not set. It is supplied by the source_cdc pipeline's "
            "`configuration` block; this module is not meant to run in any other pipeline."
        )

    return (
        spark.readStream.option("readChangeFeed", "true")
        .table(f"{schema}.{table}")
        .filter(F.col("_change_type") != "update_preimage")
    )


def apply_cdc(target: str, source: str, keys: list, comment: str) -> None:
    """Create a bronze streaming table and the AUTO CDC flow that maintains it.

    `source` must be the NAME of a view or table -- create_auto_cdc_flow rejects a DataFrame --
    which is why each dataset file declares its own @dp.temporary_view first.

    `_commit_version` is Delta's analogue of a log sequence number: it orders changes across
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
