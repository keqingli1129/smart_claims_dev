"""Transcript section: "Data Quality & Cleaning - Silver Layer" -- the claim table.

The transcript's claim cleanup does two things: it casts three string dates to real dates, and
it checks `claim_number IS NOT NULL` plus `incident_hour BETWEEN 0 AND 23`. Neither applies
literally here. Our bronze claim comes from the simulated SQL Server source, which already
emits `DateType` for both dates and has no `incident_hour` column at all -- see the schema in
jobs/simulate_source_database.py.

Inventing dirt to clean would teach the wrong lesson. What IS real about this table:

  * a claim whose amount is zero or negative is nonsense and should not reach a dashboard
  * a claim reported BEFORE the incident it describes is a data-entry error
  * `days_to_report` is a number every downstream consumer would otherwise recompute

The genuine string-to-typed casts the transcript demonstrates do exist in this project -- in
telematics.py and claims_images_metadata.py, where bronze really is all strings.

MATERIALIZED VIEW, not a streaming table. Bronze claim is an AUTO CDC target (SCD Type 1), so
its files are rewritten in place; the reasoning is in utilities/medallion.py.
"""

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from utilities.medallion import read_bronze_batch


@dp.materialized_view(
    name="claim",
    comment="Cleaned claims: quality-enforced, with reporting lag derived.",
    table_properties={"quality": "silver"},
)
@dp.expect_all_or_drop({
    # The transcript's own first check, and the one that survives unchanged.
    "valid_claim_number": "claim_number IS NOT NULL",
    "valid_policy_number": "policy_number IS NOT NULL",
    # Stands in for the transcript's incident_hour range check: a bounds assertion on the
    # column this table actually has.
    "positive_claim_amount": "claim_amount > 0",
    # A claim cannot be filed before the accident happened.
    "incident_precedes_claim": "incident_date <= claim_date",
})
def claim():
    return (
        read_bronze_batch(spark, "claim")
        .select(
            F.col("claim_number"),
            F.col("policy_number"),
            F.col("incident_date"),
            F.initcap(F.trim("incident_type")).alias("incident_type"),
            F.initcap(F.trim("incident_severity")).alias("incident_severity"),
            F.col("claim_amount"),
            F.col("claim_date"),
            F.initcap(F.trim("claim_status")).alias("claim_status"),
            # How long the customer took to report. Cheap here, and it stops every downstream
            # query from re-deriving it slightly differently.
            F.datediff("claim_date", "incident_date").alias("days_to_report"),
            # `created_at` is dropped: it describes a row in the simulated source system, not
            # the claim. `updated_at` is kept under a name that says whose clock it came from.
            F.col("updated_at").alias("source_updated_at"),
        )
    )
