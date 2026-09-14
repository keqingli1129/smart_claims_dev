"""One reader, three sources.

The transcript reads an AWS Kinesis data stream. This module keeps that code verbatim and adds
two alternatives behind the same contract, chosen by the `source_mode` pipeline configuration:

    kinesis  the transcript's source. Needs an AWS stream and a UC service credential.
    kafka    Apache Kafka. Needs a broker list and a topic.
    eventhubs  Azure Event Hubs over the Kafka protocol -- the landing point when Azure
             Event Grid routes events into Databricks. Needs a namespace and a connection
             string. Event Grid itself is not a streaming source: it pushes events over
             HTTP, so something has to receive them, and an Event Hub is the usual choice.
    volume   the simulated stream, reading files the generator job writes. Needs nothing.

Every branch returns the SAME columns, in the Kinesis vocabulary:

    partitionKey  data (BINARY)  stream  shardId  sequenceNumber  approximateArrivalTimestamp

so nothing downstream knows or cares which one ran, and switching is a variable rather than an
edit. `data` is binary in all three -- keeping the raw bytes unparsed is the point of bronze, and
what makes telematics_parsed.py's decoding step real work.

This lives in utilities/ rather than transformations/ deliberately: the pipeline's glob executes
everything under transformations/**, and this module defines no datasets.
"""

from pyspark.sql import functions as F


def _conf(spark, key: str, default: str = "") -> str:
    """Pipeline configuration value, defaulting rather than raising when unset.

    Only one source's settings are ever populated, so the other two must tolerate absence.
    """
    try:
        return spark.conf.get(key, default)
    except Exception:
        return default


# --- Kinesis ---------------------------------------------------------------------------------

def _read_kinesis(spark):
    """The transcript's read, unchanged.

    Authentication goes through a Unity Catalog *service credential*, not a hard-coded key --
    nothing to store, nothing to rotate, UC brokers the underlying IAM role.
    """
    options = {
        "streamName": _conf(spark, "kinesis.stream_name"),
        "region": _conf(spark, "kinesis.region", "us-east-1"),
        "initialPosition": _conf(spark, "kinesis.initial_position", "earliest"),
        "serviceCredential": _conf(spark, "kinesis.service_credential"),
    }
    return spark.readStream.format("kinesis").options(**options).load()


# --- Kafka -----------------------------------------------------------------------------------

def _kafka_options(spark) -> dict:
    options = {
        "kafka.bootstrap.servers": _conf(spark, "kafka.bootstrap_servers"),
        "subscribe": _conf(spark, "kafka.topic"),
        # `earliest` replays the whole topic, matching Kinesis's initialPosition and the
        # transcript's "we want to read the whole stream right from the beginning".
        "startingOffsets": _conf(spark, "kafka.starting_offsets", "earliest"),
    }

    # SASL/PLAIN over TLS, the usual shape for Confluent Cloud and Event Hubs. Credentials come
    # from a Databricks secret scope -- never a literal, and never a pipeline config value, which
    # would be readable by anyone who can view the pipeline.
    scope = _conf(spark, "kafka.secret_scope")
    if scope:
        from databricks.sdk.runtime import dbutils

        username = dbutils.secrets.get(scope, _conf(spark, "kafka.secret_username_key", "username"))
        password = dbutils.secrets.get(scope, _conf(spark, "kafka.secret_password_key", "password"))
        options.update(
            {
                "kafka.security.protocol": "SASL_SSL",
                "kafka.sasl.mechanism": "PLAIN",
                "kafka.sasl.jaas.config": (
                    "org.apache.kafka.common.security.plain.PlainLoginModule required "
                    f'username="{username}" password="{password}";'
                ),
            }
        )

    return options


def _read_kafka(spark):
    """Kafka, renamed into the Kinesis column vocabulary.

    The two sources carry the same information under different names, so the mapping is a
    straight rename -- no data is invented and none is dropped:

        key       -> partitionKey     both identify the producer/entity
        value     -> data             both are the raw message body, already BINARY
        topic     -> stream           the named channel
        partition -> shardId          the unit of parallelism
        offset    -> sequenceNumber   position within that unit
        timestamp -> approximateArrivalTimestamp
    """
    return _kafka_to_kinesis(spark.readStream.format("kafka").options(**_kafka_options(spark)).load())


def _kafka_to_kinesis(df):
    """Rename a Kafka-source DataFrame into the Kinesis column vocabulary.

    Shared by the kafka and eventhubs modes -- Event Hubs speaks the Kafka protocol, so it
    produces exactly the same columns.
    """
    return df.select(
        F.col("key").cast("string").alias("partitionKey"),
        # Already binary. No decode needed, unlike the volume source's base64.
        F.col("value").alias("data"),
        F.col("topic").alias("stream"),
        F.col("partition").cast("string").alias("shardId"),
        F.col("offset").cast("string").alias("sequenceNumber"),
        F.col("timestamp").alias("approximateArrivalTimestamp"),
    )


# --- Azure Event Hubs (the Event Grid landing point) -----------------------------------------

def _read_eventhubs(spark):
    """Azure Event Hubs, read through its Kafka-protocol endpoint.

    This is how Azure Event Grid data reaches a streaming pipeline. Event Grid is a router, not
    a queue -- it delivers events over HTTP to a handler, and routing them to an Event Hub is
    what makes them replayable and readable by Spark.

    Event Hubs' SASL differs from ordinary Kafka in exactly two ways: the username is the
    literal string `$ConnectionString`, and the password is the namespace- or entity-level
    connection string (the one containing `SharedAccessKey=`). Everything else -- port 9093,
    SASL_SSL, PLAIN -- is standard.
    """
    namespace = _conf(spark, "eventhubs.namespace")
    scope = _conf(spark, "eventhubs.secret_scope")

    if not namespace or not scope:
        raise ValueError(
            "source_mode='eventhubs' needs eventhubs.namespace and eventhubs.secret_scope. "
            "The connection string must come from a secret scope, never pipeline configuration."
        )

    from databricks.sdk.runtime import dbutils

    connection_string = dbutils.secrets.get(
        scope, _conf(spark, "eventhubs.secret_key", "connection-string")
    )

    options = {
        # An Event Hubs namespace exposes a Kafka endpoint on 9093.
        "kafka.bootstrap.servers": f"{namespace}.servicebus.windows.net:9093",
        # One Event Hub == one Kafka topic.
        "subscribe": _conf(spark, "eventhubs.name"),
        "startingOffsets": _conf(spark, "eventhubs.starting_offsets", "earliest"),
        "kafka.security.protocol": "SASL_SSL",
        "kafka.sasl.mechanism": "PLAIN",
        "kafka.sasl.jaas.config": (
            "org.apache.kafka.common.security.plain.PlainLoginModule required "
            f'username="$ConnectionString" password="{connection_string}";'
        ),
    }

    return _kafka_to_kinesis(spark.readStream.format("kafka").options(**options).load())


# --- Simulated stream ------------------------------------------------------------------------

def _read_volume(spark, checkpoint_name: str):
    """Files the generator job writes, shaped like Kinesis records.

    `data` arrives base64-encoded because that is how Kinesis carries a record body over the
    wire; decoding it back to bytes reproduces the real source column faithfully rather than
    substituting a readable string.
    """
    base = _conf(spark, "landing_volume_path")
    stream_name = _conf(spark, "kinesis.stream_name", "telematics-stream")

    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "json")
        # Per-table, so the two datasets never fight over one inferred schema.
        .option("cloudFiles.schemaLocation", f"{base}/_schemas/{checkpoint_name}")
        .option(
            "cloudFiles.schemaHints",
            "partitionKey STRING, data STRING, sequenceNumber STRING, "
            "approximateArrivalTimestamp STRING, shardId STRING",
        )
        .load(f"{base}/events")
        .select(
            F.col("partitionKey"),
            F.unbase64(F.col("data")).alias("data"),
            F.lit(stream_name).alias("stream"),
            F.col("shardId"),
            F.col("sequenceNumber"),
            F.to_timestamp("approximateArrivalTimestamp").alias("approximateArrivalTimestamp"),
        )
    )


# --- Entry point -----------------------------------------------------------------------------

def read_telematics_stream(spark, checkpoint_name: str):
    """One row per telematics event, in the shape a Kinesis source hands you.

    `checkpoint_name` is only consulted by the volume source, which needs a per-table schema
    directory. Kinesis and Kafka checkpoint through the pipeline itself.
    """
    mode = _conf(spark, "source_mode", "volume")

    if mode == "kinesis":
        return _read_kinesis(spark)
    if mode == "kafka":
        return _read_kafka(spark)
    if mode == "eventhubs":
        return _read_eventhubs(spark)
    if mode == "volume":
        return _read_volume(spark, checkpoint_name)

    raise ValueError(
        f"Unknown source_mode {mode!r}. Expected 'kinesis', 'kafka', 'eventhubs' or 'volume'."
    )
