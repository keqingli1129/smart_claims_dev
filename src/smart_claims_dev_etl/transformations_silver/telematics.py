"""Transcript section: "Data Quality & Cleaning - Silver Layer" -- the real type-cast case.

This is the table the transcript's claim-date lesson actually describes in this project. Part 1
decided, explicitly, that bronze keeps the producer's shape: "we don't want to look into the
proper data formats, we just want to make everything a string" -- see the note at the top of
transformations/telematics_parsed.py. So `speed`, `latitude`, `longitude` and `event_timestamp`
all arrive as STRING, and silver is where they stop being strings.

That matters more than it looks. A STRING speed sorts "9.5" after "145.0", and `AVG(speed)`
either fails or quietly coerces. Every aggregate in the gold layer depends on this file.

STREAMING TABLE, unlike the three CDC tables in this directory: bronze telematics is a plain
append-only stream, so nothing is ever rewritten underneath it and an incremental read is both
safe and cheaper.
"""

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from utilities.medallion import read_bronze_stream


@dp.table(
    name="telematics",
    comment="Cleaned telematics readings: real numeric and timestamp types, bounds-checked.",
    table_properties={"quality": "silver"},
)
@dp.expect_all_or_drop({
    "valid_chassis_number": "chassis_number IS NOT NULL",
    "valid_event_timestamp": "event_timestamp IS NOT NULL",
    # A reading has to survive the cast to be usable. A non-numeric speed casts to NULL rather
    # than raising, so without this check the bad rows would reach gold and poison the average.
    "speed_parsed": "speed IS NOT NULL",
    # The transcript's `incident_hour BETWEEN 0 AND 23`, transplanted onto a column that exists.
    # The generator's deliberate tail tops out around 145, so 300 rejects corruption without
    # discarding the fast-driving rows a later risk model wants.
    "plausible_speed": "speed BETWEEN 0 AND 300",
    "valid_coordinates": "latitude BETWEEN -90 AND 90 AND longitude BETWEEN -180 AND 180",
})
def telematics():
    return (
        read_bronze_stream(spark, "telematics")
        .select(
            F.upper(F.trim("chassis_number")).alias("chassis_number"),
            # The producer writes ISO-8601 with an offset; to_timestamp reads that directly.
            F.to_timestamp("event_timestamp").alias("event_timestamp"),
            F.col("speed").cast("double").alias("speed"),
            F.col("latitude").cast("double").alias("latitude"),
            F.col("longitude").cast("double").alias("longitude"),
            # Kept: it is how you trace a suspect reading back to the shard and sequence number
            # it arrived on. Dropped from gold, where nobody needs it.
            F.col("stream_metadata"),
        )
    )
