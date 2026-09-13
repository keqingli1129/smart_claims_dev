# Databricks notebook source
# MAGIC %md
# MAGIC # Telematics stream generator
# MAGIC
# MAGIC Stands in for the producer the transcript runs off-screen: *"let me under the hood send a
# MAGIC few records in real time into the stream by executing a script I have."*
# MAGIC
# MAGIC Each record is written in the envelope a Kinesis consumer sees -- `partitionKey`,
# MAGIC `sequenceNumber`, `approximateArrivalTimestamp`, `shardId`, and a base64 `data` body.
# MAGIC Base64 is how Kinesis carries a record body, so decoding it back yields genuine bytes and
# MAGIC the pipeline's parsing step is real work rather than a mock.
# MAGIC
# MAGIC * **One batch, then stop** -- leave the defaults and run the job once (40 records, matching
# MAGIC   the 40 already sitting in the transcript's stream).
# MAGIC * **Continuous demo** -- raise `num_batches`, start this job, and watch a continuous
# MAGIC   pipeline pick records up live.

# COMMAND ----------

import base64
import json
import os
import random
import time
import uuid
from datetime import datetime, timezone

dbutils.widgets.text("landing_volume_path", "", "Landing volume path")
dbutils.widgets.text("stream_name", "telematics-stream", "Stream name")
dbutils.widgets.text("num_batches", "5", "Number of batches")
dbutils.widgets.text("events_per_batch", "8", "Records per batch")
dbutils.widgets.text("sleep_seconds", "3", "Seconds between batches")

LANDING_VOLUME_PATH = dbutils.widgets.get("landing_volume_path").rstrip("/")
NUM_BATCHES = int(dbutils.widgets.get("num_batches"))
EVENTS_PER_BATCH = int(dbutils.widgets.get("events_per_batch"))
SLEEP_SECONDS = float(dbutils.widgets.get("sleep_seconds"))

if not LANDING_VOLUME_PATH:
    raise ValueError("landing_volume_path is required, e.g. /Volumes/<catalog>/<schema>/<volume>")

EVENTS_DIR = f"{LANDING_VOLUME_PATH}/events"
os.makedirs(EVENTS_DIR, exist_ok=True)
print(f"Writing to {EVENTS_DIR}")

# COMMAND ----------

# A small fleet, so the same chassis recurs and will join against policy data in a later part.
CHASSIS_NUMBERS = [f"CH-{n:05d}" for n in range(1, 11)]

# Arbitrary but consistent coordinates, so the points plot sensibly on a map later.
LAT_RANGE = (29.55, 30.05)
LON_RANGE = (-95.75, -95.05)


def make_record() -> dict:
    """One Kinesis-shaped envelope wrapping one telematics reading."""
    payload = {
        "chassis_number": random.choice(CHASSIS_NUMBERS),
        "event_timestamp": datetime.now(timezone.utc).isoformat(),
        # Mostly ordinary driving, with a deliberate tail above 75 so later parts have
        # fast-driving claims to catch.
        "speed": str(round(random.choice([random.uniform(0, 75)] * 9 + [random.uniform(75, 145)]), 1)),
        "latitude": str(round(random.uniform(*LAT_RANGE), 6)),
        "longitude": str(round(random.uniform(*LON_RANGE), 6)),
    }

    body = json.dumps(payload).encode("utf-8")

    return {
        "partitionKey": payload["chassis_number"],
        "sequenceNumber": f"{uuid.uuid4().int % (10**20):020d}",
        "approximateArrivalTimestamp": datetime.now(timezone.utc).isoformat(),
        "shardId": f"shardId-{random.randint(0, 1):012d}",
        "data": base64.b64encode(body).decode("ascii"),
    }


# COMMAND ----------

total = 0
for batch in range(1, NUM_BATCHES + 1):
    records = [make_record() for _ in range(EVENTS_PER_BATCH)]

    # One file per batch. Auto Loader treats each new file as a new micro-batch, which gives the
    # pipeline the same incremental behaviour a real stream has.
    filename = f"telematics_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:8]}.json"
    with open(f"{EVENTS_DIR}/{filename}", "w", encoding="utf-8") as fh:
        fh.write("\n".join(json.dumps(r) for r in records))

    total += len(records)
    print(f"batch {batch}/{NUM_BATCHES}: wrote {len(records)} records to {filename}")

    if batch < NUM_BATCHES and SLEEP_SECONDS > 0:
        time.sleep(SLEEP_SECONDS)

print(f"Done. {total} records across {NUM_BATCHES} files.")
