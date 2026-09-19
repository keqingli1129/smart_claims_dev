# Databricks notebook source
# MAGIC %md
# MAGIC # Object storage file generator
# MAGIC
# MAGIC Stands in for the S3 bucket the transcript reads: *"Someone pushes their CSV files or JSON
# MAGIC files or just images into your S3 bucket or Azure data lake."* That bucket arrives
# MAGIC pre-filled and the transcript's later uploads are clicked through the Databricks UI. This
# MAGIC project has neither, so both the initial load and the uploads happen here -- a clean
# MAGIC checkout reproduces the whole of Part 3 without anyone touching a console.
# MAGIC
# MAGIC ## Modes
# MAGIC
# MAGIC | `mode` | Writes | Used by |
# MAGIC |---|---|---|
# MAGIC | `seed` | 56 training images, the accident photos, and `metadata/initial.csv` | first run |
# MAGIC | `add_column_1` | one CSV row carrying an unseen `new_column_1` | schema evolution, default `addNewColumns` |
# MAGIC | `add_column_2` | one CSV row carrying `new_column_1` **and** `new_column_2` | schema evolution, `rescue` |
# MAGIC
# MAGIC The two `add_column_*` modes are the transcript's two UI uploads. Each writes exactly one
# MAGIC record, because the point being demonstrated is Auto Loader ingesting *one* new file and
# MAGIC reporting one written record -- proof it remembered the files it had already read.
# MAGIC
# MAGIC ## Why images are generated rather than downloaded
# MAGIC
# MAGIC The PNG encoder below is about thirty lines of `zlib` and `struct`. That is deliberate:
# MAGIC serverless job compute has no Pillow, and adding a dependency to produce placeholder
# MAGIC pixels would be a poor trade. The files are genuine PNGs of varying size and colour, so
# MAGIC the `content` column in bronze holds real, differing bytes rather than one repeated blob.

# COMMAND ----------

import csv
import io
import os
import random
import struct
import zlib
from datetime import datetime, timedelta, timezone

dbutils.widgets.dropdown("mode", "seed", ["seed", "add_column_1", "add_column_2"], "Mode")
dbutils.widgets.text("training_images_path", "", "Training images volume path")
dbutils.widgets.text("claims_path", "", "Claims volume path")
dbutils.widgets.text("num_training_images", "56", "Training images to write")
dbutils.widgets.text("num_claims", "12", "Claims with accident photos")

MODE = dbutils.widgets.get("mode")
TRAINING_IMAGES_PATH = dbutils.widgets.get("training_images_path").rstrip("/")
CLAIMS_PATH = dbutils.widgets.get("claims_path").rstrip("/")
NUM_TRAINING_IMAGES = int(dbutils.widgets.get("num_training_images"))
NUM_CLAIMS = int(dbutils.widgets.get("num_claims"))

if not TRAINING_IMAGES_PATH or not CLAIMS_PATH:
    raise ValueError(
        "training_images_path and claims_path are both required, "
        "e.g. /Volumes/<catalog>/<schema>/<volume>"
    )

# Three plain paths inside one volume, not three Unity Catalog objects. Auto Loader reads
# `images/` and `metadata/` separately, and cleanSource moves processed files into `archive/`.
# `archive/` has to be a SIBLING of `images/` rather than a child -- inside it, Auto Loader
# would find the archived files again on the next run and ingest them a second time.
CLAIMS_IMAGES_DIR = f"{CLAIMS_PATH}/images"
CLAIMS_METADATA_DIR = f"{CLAIMS_PATH}/metadata"
CLAIMS_ARCHIVE_DIR = f"{CLAIMS_PATH}/archive"

# COMMAND ----------

# MAGIC %md
# MAGIC ## A PNG encoder, in the standard library
# MAGIC
# MAGIC Truecolour, 8 bits per channel, one `IDAT` chunk. Every PNG chunk is
# MAGIC `length | tag | data | CRC32(tag + data)`, and every scanline in the raw stream is prefixed
# MAGIC with a filter byte -- `\x00` here, meaning "no filtering", which costs some compression and
# MAGIC saves the whole filter implementation.

# COMMAND ----------


def _chunk(tag: bytes, data: bytes) -> bytes:
    """One PNG chunk: big-endian length, four-byte tag, payload, CRC32 over tag+payload."""
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def png_bytes(rows: list[bytearray]) -> bytes:
    """Encode a list of RGB scanlines (3 bytes per pixel) as a PNG file."""
    height = len(rows)
    width = len(rows[0]) // 3

    # width, height, bit depth 8, colour type 2 (truecolour), deflate, adaptive filtering, no interlace
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + bytes(row) for row in rows)

    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(raw, 6))
        + _chunk(b"IEND", b"")
    )


def crash_photo(rng: random.Random) -> bytes:
    """A small synthetic photo: a flat body colour, a darker damage patch, and sensor noise.

    Stands in for a real car-crash photo. The three ingredients matter only in that they make
    every file a different size and a different byte sequence -- which is what a binary column
    full of identical blobs would fail to demonstrate.
    """
    width = rng.randrange(48, 129, 8)
    height = rng.randrange(48, 129, 8)

    body = (rng.randrange(60, 200), rng.randrange(60, 200), rng.randrange(60, 200))
    rows = [bytearray(bytes(body) * width) for _ in range(height)]

    # The damage: a darker rectangle somewhere in the frame.
    dx, dy = rng.randrange(0, width // 2), rng.randrange(0, height // 2)
    dw, dh = rng.randrange(width // 6, width // 2), rng.randrange(height // 6, height // 2)
    dark = bytes(max(0, c - rng.randrange(40, 60)) for c in body)
    for y in range(dy, min(dy + dh, height)):
        rows[y][dx * 3 : min(dx + dw, width) * 3] = dark * (min(dx + dw, width) - dx)

    # Sensor noise, so no two images with the same dimensions and colour compress alike.
    for _ in range(width * height // 20):
        x, y = rng.randrange(width), rng.randrange(height)
        rows[y][x * 3 : x * 3 + 3] = bytes(rng.randrange(256) for _ in range(3))

    return png_bytes(rows)


def write_csv(path: str, fieldnames: list[str], rows: list[dict]) -> None:
    """Write a header-bearing CSV. Auto Loader infers the schema from that header."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(buf.getvalue())


# COMMAND ----------

# MAGIC %md
# MAGIC ## The metadata contract
# MAGIC
# MAGIC The transcript describes the CSVs as *"additional metadata describing what these images are
# MAGIC about ... what kind of image ID it actually is, what kind of claim number this uploaded
# MAGIC image refers to."* Those two columns are the ones that matter; the rest is plausible
# MAGIC filler.
# MAGIC
# MAGIC `claim_number` is drawn from the range Part 2's source database seeds
# MAGIC (`CLM-000001`..`CLM-013000`), and deliberately from the low end of it, so every row here
# MAGIC joins against a row in `bronze.claim` rather than dangling.

# COMMAND ----------

BASE_FIELDS = ["image_id", "claim_number", "file_name", "image_type", "uploaded_at", "uploaded_by"]

IMAGE_TYPES = ["front_bumper", "rear_bumper", "driver_side", "passenger_side", "windscreen", "odometer"]

# Fixed seed: reseeding the volume produces the same metadata, so a row count that changed
# between runs points at ingestion, not at the generator.
rng = random.Random(20260918)


def metadata_row(image_id: int, claim_number: str, file_name: str, uploaded_at: datetime) -> dict:
    return {
        "image_id": f"IMG-{image_id:06d}",
        "claim_number": claim_number,
        "file_name": file_name,
        "image_type": rng.choice(IMAGE_TYPES),
        "uploaded_at": uploaded_at.isoformat(timespec="seconds"),
        "uploaded_by": f"customer_{rng.randrange(1000, 9999)}",
    }


# COMMAND ----------

# MAGIC %md
# MAGIC ## `seed` -- the bucket as the transcript finds it

# COMMAND ----------

if MODE == "seed":
    for directory in (TRAINING_IMAGES_PATH, CLAIMS_IMAGES_DIR, CLAIMS_METADATA_DIR, CLAIMS_ARCHIVE_DIR):
        os.makedirs(directory, exist_ok=True)
    print(f"archive/ created empty at {CLAIMS_ARCHIVE_DIR} -- cleanSource fills it later")

    # 56 training images, matching the "56 written records" the transcript's first pipeline run
    # reports. These are the damage-severity training set, unrelated to any specific claim.
    for i in range(1, NUM_TRAINING_IMAGES + 1):
        with open(f"{TRAINING_IMAGES_PATH}/training_{i:04d}.png", "wb") as fh:
            fh.write(crash_photo(rng))
    print(f"wrote {NUM_TRAINING_IMAGES} training images to {TRAINING_IMAGES_PATH}")

    # The accident photos customers uploaded against their claims, and one metadata row each.
    rows = []
    image_id = 0
    start = datetime.now(timezone.utc) - timedelta(days=30)

    for claim_index in range(1, NUM_CLAIMS + 1):
        claim_number = f"CLM-{claim_index:06d}"

        for shot in range(1, rng.randint(1, 3) + 1):
            image_id += 1
            file_name = f"{claim_number}_{shot}.png"

            with open(f"{CLAIMS_IMAGES_DIR}/{file_name}", "wb") as fh:
                fh.write(crash_photo(rng))

            rows.append(metadata_row(image_id, claim_number, file_name, start + timedelta(hours=image_id)))

    write_csv(f"{CLAIMS_METADATA_DIR}/claims_images_initial.csv", BASE_FIELDS, rows)
    print(f"wrote {image_id} accident photos to {CLAIMS_IMAGES_DIR}")
    print(f"wrote {len(rows)} metadata rows to {CLAIMS_METADATA_DIR}/claims_images_initial.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ## `add_column_1` -- the first upload
# MAGIC
# MAGIC The transcript: *"it looks exactly the same like the previous ones with the exception that
# MAGIC here we have this new column ... and we specify it as a value new column value one."*
# MAGIC
# MAGIC With `cloudFiles.schemaEvolutionMode` at its default `addNewColumns`, ingesting this file
# MAGIC adds `new_column_1` to the Delta schema and leaves it null for every row already there.

# COMMAND ----------

if MODE == "add_column_1":
    os.makedirs(CLAIMS_METADATA_DIR, exist_ok=True)

    row = metadata_row(900001, f"CLM-{rng.randint(1, 13000):06d}", "CLM-900001_1.png", datetime.now(timezone.utc))
    row["new_column_1"] = "new column value 1"

    path = f"{CLAIMS_METADATA_DIR}/claims_images_new_column_1.csv"
    write_csv(path, BASE_FIELDS + ["new_column_1"], [row])
    print(f"wrote 1 record with new_column_1 to {path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## `add_column_2` -- the second upload
# MAGIC
# MAGIC *"It has even one additional column, new column two that was not there before ... not even
# MAGIC in the second file that we uploaded."*
# MAGIC
# MAGIC Run this one **after** switching the pipeline to `cloudFiles.schemaEvolutionMode = rescue`.
# MAGIC `new_column_1` is already in the Delta schema by then, so it lands in its own column;
# MAGIC `new_column_2` is not, and under `rescue` it goes into `_rescued_data` as JSON instead of
# MAGIC widening the table.

# COMMAND ----------

if MODE == "add_column_2":
    os.makedirs(CLAIMS_METADATA_DIR, exist_ok=True)

    row = metadata_row(900002, f"CLM-{rng.randint(1, 13000):06d}", "CLM-900002_1.png", datetime.now(timezone.utc))
    row["new_column_1"] = "new column value 1"
    row["new_column_2"] = "new column value 2"

    path = f"{CLAIMS_METADATA_DIR}/claims_images_new_column_2.csv"
    write_csv(path, BASE_FIELDS + ["new_column_1", "new_column_2"], [row])
    print(f"wrote 1 record with new_column_1 and new_column_2 to {path}")
