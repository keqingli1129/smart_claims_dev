"""Transcript section: "Data Quality & Cleaning - Silver Layer" -- the customer table.

The transcript's headline cleanup here is splitting one `name` column into `first_name` and
`last_name`. Our source never combined them -- jobs/simulate_source_database.py writes the two
columns separately -- so there is nothing to split.

What this table does instead is the same category of work pointed the other way: it composes
`full_name` for the consumers that want one string, and normalises the fields that a human
typed. Case and whitespace in `email`, `state` and `city` are real dirt in any customer table,
including this one, because they break joins and GROUP BYs without ever looking wrong.

`date_of_birth` is already a DATE, so the transcript's cast is again a no-op; the expectation
guards it instead, and `birth_year` is derived rather than `age`. Age would be computed from
`current_date()` and would therefore be correct only until the next refresh -- a column whose
value silently rots is worse than one the consumer computes when it asks.

MATERIALIZED VIEW, not a streaming table -- bronze customer is an AUTO CDC target, and Part 2's
delete demo depends on a deleted customer actually disappearing from here. See
utilities/medallion.py.
"""

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from utilities.medallion import read_bronze_batch


@dp.materialized_view(
    name="customer",
    comment="Cleaned customers: quality-enforced, normalised contact fields, composed full name.",
    table_properties={"quality": "silver"},
)
@dp.expect_all_or_drop({
    "valid_customer_id": "customer_id IS NOT NULL",
    # The transcript's shape of check, on a column that exists here.
    "valid_email": "email IS NOT NULL AND email LIKE '%@%.%'",
    # A plausible-date bound rather than a type cast: catches both the 1900 sentinel and any
    # birth date in the future.
    "plausible_date_of_birth": "date_of_birth > DATE'1900-01-01' AND date_of_birth < current_date()",
})
def customer():
    first = F.initcap(F.trim("first_name"))
    last = F.initcap(F.trim("last_name"))

    return (
        read_bronze_batch(spark, "customer")
        .select(
            F.col("customer_id"),
            first.alias("first_name"),
            last.alias("last_name"),
            # The transcript splits a name; this composes one. Same layer, same justification:
            # give downstream the shape it actually asks for.
            F.concat_ws(" ", first, last).alias("full_name"),
            F.col("date_of_birth"),
            F.year("date_of_birth").alias("birth_year"),
            F.lower(F.trim("email")).alias("email"),
            # Strip every non-digit so 713-555-0100, (713) 555-0100 and 7135550100 compare
            # equal. The formatting was never data.
            F.regexp_replace(F.trim("phone"), r"[^0-9]", "").alias("phone"),
            F.trim("address").alias("address"),
            F.initcap(F.trim("city")).alias("city"),
            F.upper(F.trim("state")).alias("state"),
            F.trim("zip_code").alias("zip_code"),
            F.col("updated_at").alias("source_updated_at"),
        )
    )
