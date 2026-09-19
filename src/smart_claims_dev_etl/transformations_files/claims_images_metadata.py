"""Transcript section: "Schema Evolution" -- the CSV table.

Metadata describing the accident images customers upload: which image belongs to which claim,
from what angle, and the severity the customer reported. The transcript introduces it as
"exactly the same code that we have seen earlier just pointing to another subfolder", and that
is true -- the read is identical apart from the format and the path.

What makes it worth its own file is `cloudFiles.schemaEvolutionMode`, set here from pipeline
configuration so switching it is a deploy rather than an edit:

    addNewColumns   (Auto Loader's default) an unknown field fails the stream, is recorded, and
                    the restarted stream carries it as a real column. Rows that predate it read
                    NULL. Inside a pipeline the restart is automatic.

    rescue          the schema never changes. Unknown fields are stashed as JSON in
                    _rescued_data instead, which keeps a churning source from reshaping the
                    table underneath whatever reads it.

Columns stay strings. Auto Loader only infers CSV types when asked
(`cloudFiles.inferColumnTypes`), and leaving them as text matches what part 1 does with the
telematics payload: bronze keeps the source's shape, and typing belongs in silver.
"""

from pyspark import pipelines as dp

from utilities.object_storage_source import read_files


@dp.table(
    name="claims_images_metadata",
    comment="Metadata for customer-uploaded accident images, ingested incrementally from CSV.",
    table_properties={"quality": "bronze"},
)
def claims_images_metadata():
    mode = spark.conf.get("claims_metadata_schema_evolution_mode", "addNewColumns")
    return read_files(
        spark,
        "claims_metadata_path",
        file_format="csv",
        options={
            "header": "true",
            "cloudFiles.schemaEvolutionMode": mode,
        },
    )
