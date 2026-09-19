"""Transcript section: "Data Quality & Cleaning - Silver Layer" -- the second real cast case.

Part 3 ingested this CSV with types left off on purpose: Auto Loader only infers CSV types when
asked, and bronze keeps the source's shape. Every column is therefore a STRING, including
`uploaded_at`, and this is where that ends.

This table also inherits two artefacts of Part 3's schema-evolution demo, and silver is where
they get resolved rather than propagated:

  `new_column_one`  arrived under `addNewColumns` and became a real column for good. Rows that
                    predate it read NULL. It carries no meaning, so it is dropped here.

  `_rescued_data`   holds, as JSON, every unknown field seen since the mode was switched to
                    `rescue`. It is a quarantine bucket, not data. Dropped from the clean table
                    -- but see the note below on why that is a decision, not a cleanup.

STREAMING TABLE: bronze claims_images_metadata is Auto Loader append-only.
"""

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from utilities.medallion import read_bronze_stream

# The vocabulary the generator writes. Anything outside it is a typo or a new value nobody told
# the pipeline about, and either way it should be visible rather than silently averaged in.
KNOWN_SEVERITIES = "('Total Loss', 'Major Damage', 'Minor Damage')"
KNOWN_ANGLES = "('front', 'rear', 'left', 'right')"


@dp.table(
    name="claims_images_metadata",
    comment="Cleaned image metadata: typed upload timestamp, normalised vocabulary.",
    table_properties={"quality": "silver"},
)
@dp.expect_all_or_drop({
    "valid_image_id": "image_id IS NOT NULL",
    "valid_claim_number": "claim_number IS NOT NULL",
    # Same reasoning as the speed cast: a malformed timestamp string casts to NULL rather than
    # raising, so the check has to be on the result of the cast, not the input.
    "upload_timestamp_parsed": "uploaded_at IS NOT NULL",
})
# Warn, not drop. An unrecognised severity is a signal that the upload form changed -- worth
# seeing in the data-quality panel, but not worth discarding an otherwise usable image record
# over. This is the transcript's `expect_all` (warn) behaviour, used where it belongs.
@dp.expect_all({
    "known_severity": f"reported_severity IN {KNOWN_SEVERITIES}",
    "known_camera_angle": f"camera_angle IN {KNOWN_ANGLES}",
})
def claims_images_metadata():
    return (
        read_bronze_stream(spark, "claims_images_metadata")
        .select(
            F.trim("image_id").alias("image_id"),
            F.trim("claim_number").alias("claim_number"),
            F.trim("file_name").alias("file_name"),
            F.to_timestamp("uploaded_at").alias("uploaded_at"),
            F.lower(F.trim("camera_angle")).alias("camera_angle"),
            F.initcap(F.trim("reported_severity")).alias("reported_severity"),
            # Carried through from bronze. It is the only way to tell which file a row came
            # from once cleanSource has archived the original.
            F.col("source_file"),
        )
    )
