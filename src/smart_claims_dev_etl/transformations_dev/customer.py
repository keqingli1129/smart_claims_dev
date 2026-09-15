"""Transcript section: "Lakeflow Connect Setup" -- the customer table.

One of the three tables Part 2 ingests from SQL Server. Here the change feed comes from a Delta
table instead of a transaction log; everything downstream of that is identical.

The transcript's delete demo lands here: one customer is removed in the source, and SCD Type 1
means the row disappears from bronze too.
"""

from pyspark import pipelines as dp

from utilities.cdc_source import apply_cdc, read_change_feed


@dp.temporary_view()
def customer_changes():
    # Named, because apply_cdc references this view by name rather than by value.
    return read_change_feed(spark, "customer")


apply_cdc(
    target="customer",
    source="customer_changes",
    keys=["customer_id"],
    comment="Customers from the simulated SQL Server source, CDC-applied (SCD Type 1).",
)
