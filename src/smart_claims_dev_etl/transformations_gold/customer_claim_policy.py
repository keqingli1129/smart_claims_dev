"""Transcript section: "Data Aggregation - Gold Layer" -- the pre-joined table.

"We're just reading in the cleaned policy, claim and customer from our silver layer and then we
perform this join." The point is not the join itself but what a materialized view does with it:
when a silver customer row changes, this table is refreshed rather than recomputed end to end.

That is also why the three inputs are read by BARE NAME. A fully-qualified read would be an
ordinary external table reference; a bare name makes them pipeline datasets, which is what lets
the pipeline see the dependency and put these tables in the right order in its graph.

Every column is selected explicitly. All three inputs carry a join key, and two carry
`source_updated_at`, so a `select("*")` would produce duplicate column names that only fail
later, at read time, in somebody else's query.

OBSERVED FIRST-RUN BEHAVIOUR -- this table is EMPTY after the very first update of a new
pipeline, and correct from the second update on. Measured: run 1 produced 0 rows, run 2
produced 12,996, with no code change in between.

The flow ordering was not the problem; the pipeline ran the three silver flows to COMPLETED
before this one started. What distinguishes this table is that all three of its inputs are
materialized views being CREATED in that same update, and an MV reading another
just-created MV appears to plan against the pre-update (empty) snapshot. Note that
telematics_aggregated.py, whose input is a streaming table, produced its 10 rows correctly on
run 1 -- so the effect is specific to the MV-on-new-MV case.

In steady state this is invisible: the hourly job re-runs, and the next update fills the
table. It matters in exactly two situations, and in both the fix is to run the pipeline twice:
the first deploy to a fresh workspace, and any full refresh.
"""

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from utilities.medallion import gold


@dp.materialized_view(
    name=gold(spark, "customer_claim_policy"),
    comment="One row per claim, joined to its policy and the customer who holds it.",
    table_properties={"quality": "gold"},
)
def customer_claim_policy():
    claim = spark.read.table("claim").alias("c")
    policy = spark.read.table("policy").alias("p")
    customer = spark.read.table("customer").alias("cu")

    return (
        # INNER on both sides, on purpose. A claim with no matching policy, or a policy with no
        # matching customer, is referential breakage -- and Part 2's SCD Type 1 delete makes
        # that reachable: delete a customer in the source and their policies really do become
        # orphans here. An outer join would hide that behind a row full of NULLs.
        claim.join(policy, F.col("c.policy_number") == F.col("p.policy_number"), "inner")
        .join(customer, F.col("p.customer_id") == F.col("cu.customer_id"), "inner")
        .select(
            F.col("c.claim_number"),
            F.col("c.incident_date"),
            F.col("c.incident_type"),
            F.col("c.incident_severity"),
            F.col("c.claim_amount"),
            F.col("c.claim_date"),
            F.col("c.claim_status"),
            F.col("c.days_to_report"),
            F.col("p.policy_number"),
            F.col("p.policy_type"),
            F.col("p.chassis_number"),
            F.col("p.effective_date").alias("policy_effective_date"),
            F.col("p.expiry_date").alias("policy_expiry_date"),
            F.col("p.premium"),
            F.col("p.sum_insured"),
            F.col("p.deductible"),
            F.col("cu.customer_id"),
            F.col("cu.full_name").alias("customer_name"),
            F.col("cu.birth_year").alias("customer_birth_year"),
            F.col("cu.email").alias("customer_email"),
            F.col("cu.city").alias("customer_city"),
            F.col("cu.state").alias("customer_state"),
            F.col("cu.zip_code").alias("customer_zip_code"),
            # Derived here rather than in any of the three inputs, because it is the first
            # place both halves of it are in scope.
            F.round(F.col("c.claim_amount") / F.col("p.sum_insured"), 4).alias("claim_to_sum_insured_ratio"),
            # Was the accident inside the policy's coverage window? A claim outside it is not
            # wrong data -- it is a claim somebody has to look at.
            (
                F.col("c.incident_date").between(F.col("p.effective_date"), F.col("p.expiry_date"))
            ).alias("incident_within_coverage"),
        )
    )
