"""Transcript section: "Simple Streaming Ingestion".

The most minimal read there is -- connect to the stream, write what comes back into a Delta
table, look at it. The `data` column lands as raw bytes and is deliberately left that way:
seeing it unreadable is the point, and telematics_parsed.py is the fix.
"""

from pyspark import pipelines as dp

from utilities.telematics_source import read_telematics_stream


@dp.table(
    name="telematics_test",
    comment="Unparsed telematics stream. `data` is still raw bytes.",
    table_properties={"quality": "bronze"},
)
def telematics_test():
    # Kinesis, Kafka or the simulated volume, depending on `source_mode`. All three return the
    # same columns, so this line is unaffected by the choice.
    return read_telematics_stream(spark, "telematics_test")
