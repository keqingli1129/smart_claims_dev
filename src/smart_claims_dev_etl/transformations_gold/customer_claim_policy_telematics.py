"""Transcript section: "Data Aggregation - Gold Layer" -- the final table, and the geopy failure.

Two things happen in the transcript here. The first is a join: take the customer/claim/policy
table and attach each vehicle's aggregated telematics. The second is an accident -- the run
fails because `geopy` is not installed on the serverless cluster, and the fix is to add it to
the pipeline's environment dependencies and re-run.

That second part is only worth reproducing if geopy is doing real work, so it does. `geodesic`
computes the distance along the Earth's surface between the vehicle's typical operating
location and the claims service centre, which is a number an adjuster would actually ask for
and one that is genuinely awkward without a geodesy library: the naive
`sqrt(dlat^2 + dlon^2)` answer is wrong by a factor that varies with latitude, and the
haversine formula assumes a sphere the Earth is not.

The dependency is declared in resources/transformations.pipeline.yml under
`environment.dependencies`. That block is the fix the transcript applies through the UI; here it
is version-pinned and in source control, so the pipeline cannot work on one machine and fail on
another.

LEFT join, unlike the inner joins in customer_claim_policy.py. Only ten vehicles in the fleet
carry a telematics device -- jobs/generate_telematics.py emits CH-00001..CH-00010 -- so the
overwhelming majority of claims have no readings. That is the normal case, not breakage, and
dropping those claims would turn a claims table into a telematics-subscriber table.
"""

from geopy.distance import geodesic
from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from utilities.medallion import gold

# Downtown Houston. The fleet's coordinates are drawn from (29.55..30.05, -95.75..-95.05),
# which is the Houston metro, and the customer addresses are Houston-area cities -- so this is
# the centre those vehicles are actually operating around, not an arbitrary origin.
SERVICE_CENTRE = (29.7604, -95.3698)


@F.udf(returnType=DoubleType())
def distance_from_service_centre_km(latitude, longitude):
    """Geodesic distance in kilometres, or NULL when the vehicle has no readings.

    A Python UDF rather than a SQL expression because geodesic is a Python function; this is
    exactly the case the `environment.dependencies` block exists for. The NULL guard is load
    bearing -- the LEFT join below produces NULL coordinates for every claim whose vehicle has
    no telematics device, and geodesic would raise on those.
    """
    if latitude is None or longitude is None:
        return None
    return round(geodesic(SERVICE_CENTRE, (latitude, longitude)).km, 3)


@dp.materialized_view(
    name=gold(spark, "customer_claim_policy_telematics"),
    comment="Claims enriched with the vehicle's driving profile and distance from the service centre.",
    table_properties={"quality": "gold"},
)
def customer_claim_policy_telematics():
    # Both inputs are gold datasets defined in this pipeline, so they are referenced by the
    # same fully-qualified names they were published under.
    claims = spark.read.table(gold(spark, "customer_claim_policy")).alias("ccp")
    telematics = spark.read.table(gold(spark, "telematics_aggregated")).alias("t")

    return (
        claims.join(telematics, F.col("ccp.chassis_number") == F.col("t.chassis_number"), "left")
        .select(
            # String form, not F.col("ccp.*"): a star is expanded by the analyser from a
            # qualifier, and select() accepts strings and Columns side by side.
            "ccp.*",
            F.col("t.reading_count").alias("telematics_reading_count"),
            F.col("t.avg_speed"),
            F.col("t.max_speed"),
            F.col("t.avg_latitude"),
            F.col("t.avg_longitude"),
            F.col("t.first_reading_at").alias("telematics_first_reading_at"),
            F.col("t.last_reading_at").alias("telematics_last_reading_at"),
            # True only where readings exist. Saves every downstream consumer from writing
            # `WHERE avg_speed IS NOT NULL` and guessing what the NULL meant.
            F.col("t.chassis_number").isNotNull().alias("has_telematics"),
        )
        .withColumn(
            "distance_from_service_centre_km",
            distance_from_service_centre_km(F.col("avg_latitude"), F.col("avg_longitude")),
        )
    )
