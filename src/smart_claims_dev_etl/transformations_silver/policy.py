"""Transcript section: "Data Quality & Cleaning - Silver Layer" -- the policy table.

The transcript checks the policy number and then applies `abs()` to a premium column that can
go negative. Our premiums cannot: jobs/simulate_source_database.py draws them from
`_money(rng, 300, 2500)`, so `abs()` would be a no-op dressed up as a cleaning step.

The honest version of the same intent is an EXPECTATION rather than a repair. `abs()` silently
rewrites a value the source got wrong; an expectation drops the row and counts it, so the
breakage shows up in the pipeline's data-quality panel instead of being laundered into a
plausible-looking positive number. When you cannot tell whether -450.00 means "450, sign bug"
or "a 450 refund", guessing is the worse of the two options.

MATERIALIZED VIEW, not a streaming table -- bronze policy is an AUTO CDC target. See
utilities/medallion.py.
"""

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from utilities.medallion import read_bronze_batch


@dp.materialized_view(
    name="policy",
    comment="Cleaned policies: quality-enforced, with policy term length derived.",
    table_properties={"quality": "silver"},
)
@dp.expect_all_or_drop({
    "valid_policy_number": "policy_number IS NOT NULL",
    "valid_customer_id": "customer_id IS NOT NULL",
    # The transcript's abs(premium), restated as a check instead of a silent repair.
    "positive_premium": "premium > 0",
    "coverage_not_inverted": "expiry_date > effective_date",
})
def policy():
    return (
        read_bronze_batch(spark, "policy")
        .select(
            F.col("policy_number"),
            F.col("customer_id"),
            # The join key back to Part 1's telematics fleet. Bronze left it alone on purpose;
            # this is the layer that is allowed to care.
            F.upper(F.trim("chassis_number")).alias("chassis_number"),
            F.initcap(F.trim("policy_type")).alias("policy_type"),
            F.col("effective_date"),
            F.col("expiry_date"),
            F.col("premium"),
            F.col("sum_insured"),
            F.col("deductible"),
            F.datediff("expiry_date", "effective_date").alias("policy_term_days"),
            F.col("updated_at").alias("source_updated_at"),
        )
    )
