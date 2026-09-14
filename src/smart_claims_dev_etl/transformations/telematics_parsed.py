"""Transcript section: "Decoding and Parsing".

Same stream -- but this time the bytes are cast to a string, the JSON body is decoded, and the
fields that matter later (speed above all) become real columns instead of one cryptic blob.

Everything is typed STRING on purpose: "we don't want to look into the proper data formats, we
just want to make everything a string." Real typing happens in the silver layer.
"""

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql.types import MapType, StringType

from utilities.telematics_source import read_telematics_stream

# MAP<STRING, STRING> accepts whatever keys the producer sends without the schema having to know
# them up front -- useful when the payload is still changing shape.
PAYLOAD_SCHEMA = MapType(StringType(), StringType())


@dp.table(
    name="telematics",
    comment="Parsed telematics stream, decoded into typed columns.",
    table_properties={"quality": "bronze"},
)
def telematics():
    return (
        read_telematics_stream(spark, "telematics")
        # The one line that makes the bytes readable.
        .withColumn("raw_json", F.col("data").cast("string"))
        .withColumn("decoded_data", F.from_json("raw_json", PAYLOAD_SCHEMA))
        .select(
            # Kinesis bookkeeping, kept together and out of the way.
            F.struct(
                F.col("partitionKey"),
                F.col("stream"),
                F.col("shardId"),
                F.col("sequenceNumber"),
                F.col("approximateArrivalTimestamp"),
            ).alias("stream_metadata"),
            # The telematics payload, one column per field.
            F.col("decoded_data.chassis_number").alias("chassis_number"),
            F.col("decoded_data.event_timestamp").alias("event_timestamp"),
            F.col("decoded_data.speed").alias("speed"),
            F.col("decoded_data.latitude").alias("latitude"),
            F.col("decoded_data.longitude").alias("longitude"),
            # Kept so nothing is lost if the producer adds a field before this file catches up.
            F.col("raw_json"),
        )
    )
