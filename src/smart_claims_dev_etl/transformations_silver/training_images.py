"""Transcript section: "Data Quality & Cleaning - Silver Layer" -- "let's not forget the images".

The transcript does cleanup on both image tables without showing the code. There is one obvious
thing to do here and it is not a type cast: DROP THE BYTES.

`binaryFile` puts every image's full `content` in a column. Bronze keeps it, because bronze
keeps what arrived. Silver is read by joins, counts and dashboards, none of which want a
multi-megabyte BLOB dragged through a shuffle. What they want is the catalogue: which files
exist, how big, how recent, and where to find the bytes if they are ever actually needed.

`path` is that pointer, so nothing is lost -- the image is one `spark.read.format("binaryFile")`
away, on demand, for the handful of rows that need it rather than all of them. Part 5's training
notebook is the one consumer that genuinely needs the pixels, and it joins back to bronze on
`path` to get them.

The one thing silver ADDS is `label`. Part 5: *"in our pre-processing step in our last part, we
basically extracted this label out of the path right here as the names always have the label in
it."* That extraction belongs here rather than in the notebook, because a label parsed at train
time is invisible to everything else -- a dashboard counting the class balance, or a second model
next year, would each have to re-derive it from the same regex and hope they agreed.

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
# Note the column names: expectations are evaluated against this dataset's OUTPUT schema, not
# its input. `size_bytes` is what the select below renames bronze's `length` to, and naming the
# input column here fails analysis with UNRESOLVED_COLUMN before a single row is read.
@dp.expect_all_or_drop({
    "valid_path": "path IS NOT NULL",
    # A zero-length file is a failed upload, not an image. It would sail through any check that
    # only looked at the path.
    "non_empty_file": "size_bytes > 0",
    # An unlabelled image is not training data. `train_0036.png` -- the naming used before part 5
    # -- yields an empty label, fails here, and is dropped, while bronze keeps the row it always
    # had.
    #
    # THIS ONLY RETIRES OLD IMAGES ON A FULL REFRESH, and the distinction cost a debugging
    # session. A streaming table is append-only: adding `label` to the select evolves the schema
    # and backfills every ALREADY-COMMITTED row with NULL, but those rows are not re-read and
    # this expectation is never evaluated against them. Rows that predate the column therefore
    # survive it, with a NULL label rather than the empty string the predicate is written for --
    # and `NULL <> ''` would not have dropped them either.
    #
    # After re-seeding the volume, the silver table is only correct again once that has happened:
    #
    #     databricks bundle run transformations -t dev --full-refresh training_images
    #
    # which resets this one table and re-reads bronze from the beginning. It is safe to scope
    # that narrowly: nothing downstream of this table lives in the pipeline, so the gold
    # materialized views are excluded from the update and keep their contents.
    "labelled": "label <> ''",
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
            # `train_0007_minor_damage.png` -> `Minor Damage`. Anchored on the four-digit
            # sequence so an underscore inside the label cannot be mistaken for the separator,
            # and on the extension so it cannot swallow it. A name that does not match this
            # shape yields "" rather than a wrong guess, and the expectation above drops it.
            F.initcap(
                F.regexp_replace(
                    F.regexp_extract("path", r"_\d{4}_([a-z_]+)\.[^.]+$", 1), "_", " "
                )
            ).alias("label"),
            # Note what is absent: `content`. That is the entire point of this file.
        )
    )
