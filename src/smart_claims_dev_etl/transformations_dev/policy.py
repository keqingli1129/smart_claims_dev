"""Transcript section: "Lakeflow Connect Setup" -- the policy table.

One of the three tables Part 2 ingests from SQL Server. Here the change feed comes from a Delta
table instead of a transaction log; everything downstream of that is identical.

The transcript's insert demo lands here: POL-999001 does not exist in bronze until the source
job commits it and this pipeline runs.

`chassis_number` is the column that joins back to Part 1's telematics fleet. Nothing in this
file does that join -- bronze stays source-faithful, exactly as Lakeflow Connect would leave
it, and the join belongs to a later silver dataset.
"""

from pyspark import pipelines as dp

from utilities.cdc_source import apply_cdc, read_change_feed


@dp.temporary_view()
def policy_changes():
    # Named, because apply_cdc references this view by name rather than by value.
    return read_change_feed(spark, "policy")


apply_cdc(
    target="policy",
    source="policy_changes",
    keys=["policy_number"],
    comment="Policies from the simulated SQL Server source, CDC-applied (SCD Type 1).",
)
