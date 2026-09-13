"""Transcript section: "Decoding and Parsing".

Same stream, same configuration block copied over -- but this time the bytes are cast to a
string, the JSON body is decoded, and the fields that matter later (speed above all) become real
columns instead of one cryptic blob.

Everything is typed STRING on purpose: "we don't want to look into the proper data formats, we
just want to make everything a string." Real typing happens in the silver layer.
"""

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql.types import MapType, StringType

# --- Stream configuration (copied from telematics_simple.py, as in the transcript) ------------

SOURCE_MODE = spark.conf.get("source_mode", "volume")
LANDING_VOLUME_PATH = spark.conf.get("landing_volume_path")
STREAM_NAME = spark.conf.get("kinesis.stream_name")

KINESIS_OPTIONS = {
    "streamName": STREAM_NAME,
    "region": spark.conf.get("kinesis.region"),
    "initialPosition": "earliest",
    "serviceCredential": spark.conf.get("kinesis.service_credential"),
}

# MAP<STRING, STRING> accepts whatever keys the producer sends without the schema having to know
# them up front -- useful when the payload is still changing shape.
PAYLOAD_SCHEMA = MapType(StringType(), StringType())


def read_telematics_stream(checkpoint_name: str):
    """Identical to the one in telematics_simple.py -- see the note there."""
    if SOURCE_MODE == "kinesis":
        return spark.readStream.format("kinesis").options(**KINESIS_OPTIONS).load()

    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.schemaLocation", f"{LANDING_VOLUME_PATH}/_schemas/{checkpoint_name}")
        .option(
            "cloudFiles.schemaHints",
            "partitionKey STRING, data STRING, sequenceNumber STRING, "
            "approximateArrivalTimestamp STRING, shardId STRING",
        )
        .load(f"{LANDING_VOLUME_PATH}/events")
        .select(
            F.col("partitionKey"),
            F.unbase64(F.col("data")).alias("data"),
            F.lit(STREAM_NAME).alias("stream"),
            F.col("shardId"),
            F.col("sequenceNumber"),
            F.to_timestamp("approximateArrivalTimestamp").alias("approximateArrivalTimestamp"),
        )
    )


@dp.table(
    name="telematics",
    comment="Parsed Kinesis data stream holding telematics data.",
    table_properties={"quality": "bronze"},
)
def telematics():
    return (
        read_telematics_stream("telematics")
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
