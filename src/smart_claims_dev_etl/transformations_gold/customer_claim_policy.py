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
