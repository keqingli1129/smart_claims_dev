# Databricks notebook source
# MAGIC %md
# MAGIC # Object storage file generator
# MAGIC
# MAGIC Plays the part of "someone pushes their files into your S3 bucket" -- the premise part 3
# MAGIC opens with. The transcript reads an external volume somebody else had already filled;
# MAGIC this fills two managed volumes instead. Auto Loader cannot tell the difference: it reads
# MAGIC a `/Volumes/...` path either way.
# MAGIC
# MAGIC Four modes, matching the transcript's four upload moments:
# MAGIC
# MAGIC | mode | writes | demonstrates |
# MAGIC |---|---|---|
# MAGIC | `seed` | 56 training images, 12 claim images, 1 metadata CSV | the first pipeline run |
# MAGIC | `add_column` | one CSV row with one extra column | `schemaEvolutionMode = addNewColumns` |
# MAGIC | `add_two_columns` | one CSV row with two extra columns | `schemaEvolutionMode = rescue` |
# MAGIC | `more_images` | four more claim images | re-running `cleanSource` |
# MAGIC
# MAGIC 56 is not arbitrary -- it is the count the transcript's first run reports on screen.

# COMMAND ----------

import csv
import io
import os
import random
import struct
import zlib
from datetime import datetime, timezone

dbutils.widgets.dropdown(
    "mode", "seed", ["seed", "add_column", "add_two_columns", "more_images"], "Mode"
)
dbutils.widgets.text("training_images_path", "", "Training images volume path")
dbutils.widgets.text("claims_path", "", "Claims volume path")
dbutils.widgets.text("source_schema", "", "Source schema (for real claim numbers)")
dbutils.widgets.text("random_seed", "7", "Random seed")

MODE = dbutils.widgets.get("mode")
TRAINING_PATH = dbutils.widgets.get("training_images_path").rstrip("/")
CLAIMS_PATH = dbutils.widgets.get("claims_path").rstrip("/")
SOURCE_SCHEMA = dbutils.widgets.get("source_schema").strip()
RANDOM_SEED = int(dbutils.widgets.get("random_seed"))

if not TRAINING_PATH or not CLAIMS_PATH:
    raise ValueError("training_images_path and claims_path are both required")

IMAGES_DIR = f"{CLAIMS_PATH}/images"
METADATA_DIR = f"{CLAIMS_PATH}/metadata"
ARCHIVE_DIR = f"{CLAIMS_PATH}/archive"

N_TRAINING_IMAGES = 56          # the count the transcript's first run reports
N_CLAIM_IMAGES = 12

print(f"mode={MODE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Image rendering
# MAGIC
# MAGIC Pillow if the runtime has it, so each file carries a readable label. If not, a
# MAGIC hand-written PNG -- flat colour, but a genuine PNG that `binaryFile` reads identically.
# MAGIC Either way nothing is installed and `pyproject.toml` stays dependency-free.

# COMMAND ----------

try:
    from PIL import Image, ImageDraw
    RENDERER = "pillow"
except ImportError:
    RENDERER = "stdlib"

print(f"renderer: {RENDERER}")

WIDTH, HEIGHT = 320, 240


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def _flat_png(rgb) -> bytes:
    """A minimal valid PNG of one solid colour, using only the standard library.

    Each scanline is prefixed with filter byte 0 (None), which is what the PNG spec requires
    even when no filtering is applied.
    """
    row = bytes([0]) + bytes(rgb) * WIDTH
    raw = row * HEIGHT
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw, 6))
        + _png_chunk(b"IEND", b"")
    )


def render_image(label: str, rgb) -> bytes:
    if RENDERER == "stdlib":
        return _flat_png(rgb)

    img = Image.new("RGB", (WIDTH, HEIGHT), tuple(rgb))
    draw = ImageDraw.Draw(img)
    # A crude "damaged panel" so the images are not uniform: a darker wedge plus the label.
    dark = tuple(max(0, c - 60) for c in rgb)
    draw.polygon([(0, HEIGHT), (WIDTH // 2, HEIGHT // 3), (WIDTH, HEIGHT)], fill=dark)
    draw.text((12, 12), label, fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def write_image(path: str, payload: bytes) -> None:
    with open(path, "wb") as fh:
        fh.write(payload)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Claim numbers
# MAGIC
# MAGIC Read back out of part 2's source database, so the metadata references claims that really
# MAGIC exist. Same idea as `chassis_number` linking parts 1 and 2 -- three parts, one dataset.
# MAGIC Falls back to synthetic identifiers if part 2 was never run.

# COMMAND ----------

def claim_numbers(n: int, rng) -> list:
    if SOURCE_SCHEMA and spark.catalog.tableExists(f"{SOURCE_SCHEMA}.claim"):
        rows = (
            spark.table(f"{SOURCE_SCHEMA}.claim")
            .select("claim_number")
            .limit(2000)
            .collect()
        )
        if rows:
            pool = [r["claim_number"] for r in rows]
            print(f"claim numbers: {len(pool)} read from {SOURCE_SCHEMA}.claim")
            return [rng.choice(pool) for _ in range(n)]

    print("claim numbers: synthetic (part 2's source.claim not found)")
    return [f"CLM-{rng.randint(1, 13000):06d}" for _ in range(n)]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Metadata CSV
# MAGIC
# MAGIC The header is written per file, which is what makes the schema-evolution demo work: a
# MAGIC later file can simply carry more columns than an earlier one.

# COMMAND ----------

SEVERITIES = ["Total Loss", "Major Damage", "Minor Damage"]
ANGLES = ["front", "rear", "driver side", "passenger side", "interior"]


def write_csv(path: str, rows: list, columns: list) -> None:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(buf.getvalue())
    print(f"wrote {path}: {len(rows)} rows, columns {columns}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## seed
# MAGIC
# MAGIC Creates the folder structure -- including the empty `archive/`, which cleanSource needs
# MAGIC to exist before it can move anything into it.
# MAGIC
# MAGIC ## The label lives in the training filename
# MAGIC
# MAGIC Part 5 fine-tunes a ResNet on these images, and the transcript gets its labels the same
# MAGIC way: *"the names always have the label in it"*. So a training image is written as
# MAGIC `train_0007_minor_damage.png`, and `transformations_silver/training_images.py` pulls the
# MAGIC label back out of the path with a regex.
# MAGIC
# MAGIC The vocabulary is `SEVERITIES` -- the same three values the claim metadata uses for
# MAGIC `reported_severity`. That is deliberate and not cosmetic: part 5's whole point is
# MAGIC comparing what the model predicts against what the customer claimed, and two vocabularies
# MAGIC would need a mapping table nobody would maintain. (The transcript's own labels are
# MAGIC minor/major/okay, which do not line up with its own metadata. Ours do.)
# MAGIC
# MAGIC **What this does NOT do is make the images learnable.** The colour is still
# MAGIC `rng.randint` per image, drawn independently of the label, so there is no pixel signal
# MAGIC for any model to find. The label is real; the correlation is not. Part 5's notebook says
# MAGIC so where the confusion matrix would otherwise imply a result.

# COMMAND ----------

rng = random.Random(RANDOM_SEED)

BASE_COLUMNS = ["image_id", "claim_number", "file_name", "uploaded_at",
                "camera_angle", "reported_severity"]

if MODE == "seed":
    for d in (TRAINING_PATH, IMAGES_DIR, METADATA_DIR, ARCHIVE_DIR):
        os.makedirs(d, exist_ok=True)
    print(f"folders ready (including the empty {ARCHIVE_DIR})")

    for i in range(1, N_TRAINING_IMAGES + 1):
        # Round-robin, not rng.choice. 56 images over three classes gives 19/19/18 every run,
        # where random assignment can leave one class with four members -- and a stratified
        # train/test split of a 56-row dataset has no way to recover from that.
        severity = SEVERITIES[(i - 1) % len(SEVERITIES)]
        slug = severity.lower().replace(" ", "_")
        # The rendered text is the severity too, so a human who opens the file sees the same
        # label the pipeline will extract from its name.
        rgb = (rng.randint(40, 215), rng.randint(40, 215), rng.randint(40, 215))
        write_image(f"{TRAINING_PATH}/train_{i:04d}_{slug}.png", render_image(severity, rgb))
    print(f"wrote {N_TRAINING_IMAGES} training images to {TRAINING_PATH}")
    print(f"labels in filenames: {', '.join(SEVERITIES)}")

    claims = claim_numbers(N_CLAIM_IMAGES, rng)
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for i, claim_no in enumerate(claims, start=1):
        image_id = f"IMG-{i:05d}"
        file_name = f"{image_id}.png"
        rgb = (rng.randint(40, 215), rng.randint(40, 215), rng.randint(40, 215))
        write_image(f"{IMAGES_DIR}/{file_name}", render_image(image_id, rgb))
        rows.append({
            "image_id": image_id,
            "claim_number": claim_no,
            "file_name": file_name,
            "uploaded_at": now,
            "camera_angle": rng.choice(ANGLES),
            "reported_severity": rng.choice(SEVERITIES),
        })
    print(f"wrote {len(rows)} claim images to {IMAGES_DIR}")

    write_csv(f"{METADATA_DIR}/claims_images_0001.csv", rows, BASE_COLUMNS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## add_column / add_two_columns
# MAGIC
# MAGIC One new CSV file with extra columns, matching the transcript's manual uploads. The first
# MAGIC lands while the pipeline is on the default `addNewColumns`, so the column joins the
# MAGIC table schema and every earlier row reads NULL. The second lands after the mode is
# MAGIC switched to `rescue`, so its extra column goes to `_rescued_data` instead.

# COMMAND ----------

if MODE in ("add_column", "add_two_columns"):
    extra = {"new_column_one": "new column value one"}
    suffix = "0002"
    if MODE == "add_two_columns":
        extra["new_column_two"] = "new column value two"
        suffix = "0003"

    claim_no = claim_numbers(1, rng)[0]
    image_id = f"IMG-9{suffix}"
    row = {
        "image_id": image_id,
        "claim_number": claim_no,
        "file_name": f"{image_id}.png",
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "camera_angle": "front",
        "reported_severity": "Major Damage",
        **extra,
    }
    write_csv(f"{METADATA_DIR}/claims_images_{suffix}.csv", [row],
              BASE_COLUMNS + list(extra))
    print(f"added {len(extra)} new column(s): {', '.join(extra)}")
    print(f"look for image_id = '{image_id}' after the next pipeline run")

# COMMAND ----------

# MAGIC %md
# MAGIC ## more_images
# MAGIC
# MAGIC Fresh claim images so the cleanSource job in step 6 has something to process and archive
# MAGIC on a second run.

# COMMAND ----------

if MODE == "more_images":
    os.makedirs(IMAGES_DIR, exist_ok=True)
    existing = len([f for f in os.listdir(IMAGES_DIR) if f.endswith(".png")])
    for i in range(existing + 1, existing + 5):
        image_id = f"IMG-{i:05d}"
        rgb = (rng.randint(40, 215), rng.randint(40, 215), rng.randint(40, 215))
        write_image(f"{IMAGES_DIR}/{image_id}.png", render_image(image_id, rgb))
    print(f"wrote 4 more claim images to {IMAGES_DIR}")
