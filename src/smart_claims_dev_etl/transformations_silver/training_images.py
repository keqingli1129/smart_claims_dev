"""Transcript section: "Data Quality & Cleaning - Silver Layer" -- "let's not forget the images".

The transcript does cleanup on both image tables without showing the code. There is one obvious
thing to do here and it is not a type cast: DROP THE BYTES.

`binaryFile` puts every image's full `content` in a column. Bronze keeps it, because bronze
keeps what arrived. Silver is read by joins, counts and dashboards, none of which want a
multi-megabyte BLOB dragged through a shuffle. What they want is the catalogue: which files
exist, how big, how recent, and where to find the bytes if they are ever actually needed.

`path` is that pointer, so nothing is lost -- the image is one `spark.read.format("binaryFile")`
away, on demand, for the handful of rows that need it rather than all of them.

STREAMING TABLE: bronze training_images is Auto Loader append-only.
"""

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from utilities.medallion import read_bronze_stream


@dp.table(
    name="training_images",
    comment="Catalogue of training images -- metadata only, bytes left in bronze.",
    table_properties={"quality": "silver"},
)
@dp.expect_all_or_drop({
    "valid_path": "path IS NOT NULL",
    # A zero-length file is a failed upload, not an image. It would sail through any check that
    # only looked at the path.
    "non_empty_file": "length > 0",
})
def training_images():
    return (
        read_bronze_stream(spark, "training_images")
        .select(
            F.col("path"),
            # The bare filename, which is what identifies the image to a human or a label file.
            F.regexp_extract("path", r"([^/]+)$", 1).alias("file_name"),
            F.col("modificationTime").alias("modified_at"),
            F.col("length").alias("size_bytes"),
            # Note what is absent: `content`. That is the entire point of this file.
        )
    )
