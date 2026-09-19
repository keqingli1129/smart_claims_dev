"""Transcript section: "Data Aggregation - Gold Layer" -- "let's start with a super simple one".

The transcript's exact example: group the telematics readings by chassis number, average the
speed, average the location. Its reasoning is worth keeping verbatim -- for streaming data "you
have lots of small records and most of the time for your actual analytical analysis you don't
need those super fine granular records".

MATERIALIZED VIEW over a BATCH read, and this is the one place where getting the dataset type
wrong is easy and silent. A streaming table is append-only: it could only ever append new
per-reading rows, never revise an average that a new reading has changed. An MV recomputes --
and on serverless, with row tracking on the source, refreshes incrementally rather than from
scratch, which is the transcript's "it will only update based on the new record".

`avg(speed)` here is only meaningful because silver cast speed to DOUBLE. Over bronze's strings
this would have been a type error or, worse, a lexicographic surprise.
"""

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from utilities.medallion import gold


@dp.materialized_view(
    name=gold(spark, "telematics_aggregated"),
    comment="Per-vehicle driving profile: average and peak speed, typical location, reading count.",
    table_properties={"quality": "gold"},
)
def telematics_aggregated():
    return (
        # Bare name: the silver dataset is a sibling in this pipeline, and the pipeline's
        # default schema is silver.
        spark.read.table("telematics")
        .groupBy("chassis_number")
        .agg(
            F.count("*").alias("reading_count"),
            F.round(F.avg("speed"), 2).alias("avg_speed"),
            # Kept alongside the average deliberately. An average hides exactly the behaviour a
            # claims model cares about -- one 140 km/h reading vanishes into a mean of 40.
            F.round(F.max("speed"), 2).alias("max_speed"),
            F.round(F.avg("latitude"), 6).alias("avg_latitude"),
            F.round(F.avg("longitude"), 6).alias("avg_longitude"),
            F.min("event_timestamp").alias("first_reading_at"),
            F.max("event_timestamp").alias("last_reading_at"),
        )
    )
