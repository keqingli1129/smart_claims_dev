"""Transcript section: "Autoloader Example" -- the minimal case.

Images of car crashes, used in a later part to train the damage-severity classifier. The
transcript calls this "the most minimal implementation that you can have", and the point is
what is ABSENT:

    no cloudFiles.schemaLocation      the pipeline owns this dataset's state and supplies it
    no checkpointLocation             same
    no schema, no schemaHints         binaryFile has a fixed schema; there is nothing to infer

Outside a declarative pipeline all three have to be named by hand -- which is exactly what
jobs/archive_claim_images.py has to do, and why it reads so much longer than this file for
what is fundamentally the same read.

`binaryFile` yields one row per file: path, modificationTime, length, content.
"""

from pyspark import pipelines as dp

from utilities.object_storage_source import read_files


@dp.table(
    name="training_images",
    comment="Car-crash images for training the damage-severity classifier. Raw bytes.",
    table_properties={"quality": "bronze"},
)
def training_images():
    return read_files(spark, "training_images_path", file_format="binaryFile")
