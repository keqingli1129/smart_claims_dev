"""Transcript section: "Lakeflow Connect Setup" -- the claim table.

One of the three tables Part 2 ingests from SQL Server. Here the change feed comes from a Delta
table instead of a transaction log; everything downstream of that is identical.

The transcript's update demo lands here: CLM-000001 flips from Total Loss to Minor Damage in
the source, and SCD Type 1 means bronze shows only the new value -- which is what makes the
transcript's verification query return one row reading "Minor Damage" rather than two rows.

This is also the table the source job's optional `churn` parameter mutates in bulk, so it is
the one that exercises many commits in a single pipeline run. CLM-000001 is excluded from that
churn so the verification query stays deterministic.
"""

from pyspark import pipelines as dp

from utilities.cdc_source import apply_cdc, read_change_feed


@dp.temporary_view()
def claim_changes():
    # Named, because apply_cdc references this view by name rather than by value.
    return read_change_feed(spark, "claim")


apply_cdc(
    target="claim",
    source="claim_changes",
    keys=["claim_number"],
    comment="Claims from the simulated SQL Server source, CDC-applied (SCD Type 1).",
)
