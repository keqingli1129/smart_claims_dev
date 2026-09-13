"""Transcript section: "Simple Streaming Ingestion".

The most minimal read there is -- connect to the stream, write what comes back into a Delta
table, look at it. The `data` column lands as raw bytes and is deliberately left that way:
seeing it unreadable is the point, and telematics_parsed.py is the fix.
"""

from pyspark import pipelines as dp
from pyspark.sql import functions as F

# --- Stream configuration --------------------------------------------------------------------
# Set in resources/smart_claims_dev_etl.pipeline.yml under `configuration:`, which is how a
# pipeline passes deploy-time values into pipeline code.

SOURCE_MODE = spark.conf.get("source_mode", "volume")
LANDING_VOLUME_PATH = spark.conf.get("landing_volume_path")
STREAM_NAME = spark.conf.get("kinesis.stream_name")

# Authentication goes through a Unity Catalog *service credential*, not a hard-coded key. Nothing
# to store, nothing to rotate -- UC brokers the underlying IAM role.
KINESIS_OPTIONS = {
    "streamName": STREAM_NAME,
    "region": spark.conf.get("kinesis.region"),
    "initialPosition": "earliest",
    "serviceCredential": spark.conf.get("kinesis.service_credential"),
}


def read_telematics_stream(checkpoint_name: str):
    """One row per telematics event, in the shape a Kinesis source hands you.

    Both branches yield the same columns -- partitionKey, data (bytes), stream, shardId,
    sequenceNumber, approximateArrivalTimestamp -- so everything downstream is identical and
    switching sources needs no code change, only `source_mode`.
    """
    if SOURCE_MODE == "kinesis":
        # The transcript's read, verbatim. Needs a real stream plus the service credential.
        return spark.readStream.format("kinesis").options(**KINESIS_OPTIONS).load()

    # Simulated stream. The generator job writes Kinesis-shaped envelopes as newline-delimited
    # JSON; `data` arrives base64-encoded, which is how Kinesis carries a record body over the
    # wire. Decoding it back to bytes reproduces the real source column faithfully.
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
    name="telematics_test",
    comment="Unparsed Kinesis data stream holding telematics data. `data` is still raw bytes.",
    table_properties={"quality": "bronze"},
)
def telematics_test():
    return read_telematics_stream("telematics_test")
