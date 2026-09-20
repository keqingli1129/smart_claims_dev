# Databricks notebook source
# MAGIC %md
# MAGIC # Fine-tune a damage-severity classifier
# MAGIC
# MAGIC Transcript section: *"Training the Model"* and *"Batch & Real-Time Inference"* (part 5).
# MAGIC
# MAGIC Takes the car-crash images ingested in part 3, fine-tunes a pre-trained ResNet to classify
# MAGIC the severity of the damage, tracks the run in MLflow, registers the result into Unity
# MAGIC Catalog, and scores every image back through it as a Spark UDF.
# MAGIC
# MAGIC ```
# MAGIC bronze.training_images   silver.training_images
# MAGIC        (content)              (label)
# MAGIC             \                   /
# MAGIC              join on path      /
# MAGIC                    |
# MAGIC        silver.training_images_resized      224x224, the size ResNet was trained on
# MAGIC                    |
# MAGIC            MLflow run: fine-tune, log params + metrics
# MAGIC                    |
# MAGIC        UC model  gold.claims_damage_level@prod
# MAGIC                    |
# MAGIC            spark_udf over the original images
# MAGIC                    |
# MAGIC          gold.damage_predictions  + confusion matrix
# MAGIC ```
# MAGIC
# MAGIC ## Read this before you believe the confusion matrix
# MAGIC
# MAGIC **The training images contain no signal.** `jobs/generate_object_storage_files.py` picks
# MAGIC each image's colour with `rng.randint`, independently of the label it writes into the
# MAGIC filename. The wedge shape is identical in every image. So the pixels and the labels are
# MAGIC statistically unrelated, and no model — this one or a better one — can do better than
# MAGIC chance on them.
# MAGIC
# MAGIC That is a property of the synthetic data, not of the code below. Everything here is the
# MAGIC real mechanism: a real fine-tune, real MLflow tracking, a real UC registration, real
# MAGIC batch scoring. Point it at real labelled photographs and it produces a real model. Run it
# MAGIC on what is in the volume today and the confusion matrix should come out roughly uniform,
# MAGIC and **that is the correct result** — a matrix with a strong diagonal here would mean
# MAGIC something had leaked, not that the model had learned.
# MAGIC
# MAGIC The notebook prints this warning next to the matrix for exactly that reason.
# MAGIC
# MAGIC ## Why a classic ML cluster
# MAGIC
# MAGIC Every other compute in this bundle is serverless. This one is not, per the transcript:
# MAGIC *"we will use a machine learning cluster as it has lots of those frameworks and libraries
# MAGIC that we need already pre-installed"*. `torch`, `torchvision` and `scikit-learn` come with
# MAGIC the ML runtime; only `transformers` and `datasets` are installed below. The cluster spec
# MAGIC lives in `resources/train_damage_classifier.job.yml`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Libraries
# MAGIC
# MAGIC Deliberately unpinned. Pinning here would mean guessing versions that are compatible with
# MAGIC whatever `torch` the chosen ML runtime ships, and a wrong guess fails at import rather
# MAGIC than at review. Instead the resolved versions are read back after install, logged as
# MAGIC MLflow parameters, and written into the model's `pip_requirements` — so the run is
# MAGIC reproducible after the fact even though it was not pinned in advance.

# COMMAND ----------

# MAGIC %pip install -q transformers datasets accelerate

# COMMAND ----------

# restartPython is what makes the freshly installed wheels visible to this session. Everything
# above this line is gone afterwards -- widgets survive, Python state does not -- which is why
# the imports and configuration start below rather than above.
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC
# MAGIC Schema names arrive already resolved from the job definition, so development mode's
# MAGIC `dev_<user>_` prefix is picked up rather than reconstructed here.

# COMMAND ----------

import io
import json
from importlib.metadata import version
from pathlib import Path

dbutils.widgets.text("catalog", "", "Catalog")
dbutils.widgets.text("bronze_schema", "", "Bronze schema (resolved)")
dbutils.widgets.text("silver_schema", "", "Silver schema (resolved)")
dbutils.widgets.text("gold_schema", "", "Gold schema (resolved)")
dbutils.widgets.text("model_name", "claims_damage_level", "Registered model name")
dbutils.widgets.text("experiment_path", "", "MLflow experiment (blank = under /Users/<you>)")
dbutils.widgets.text("base_model", "microsoft/resnet-50", "Hugging Face checkpoint")
dbutils.widgets.text("image_size", "224", "Resize edge, px")
dbutils.widgets.text("epochs", "5", "Training epochs")
dbutils.widgets.text("batch_size", "8", "Per-device batch size")
dbutils.widgets.text("learning_rate", "5e-5", "Learning rate")
dbutils.widgets.text("test_fraction", "0.25", "Held-out fraction")
dbutils.widgets.text("random_seed", "42", "Random seed")

CATALOG = dbutils.widgets.get("catalog").strip()
BRONZE = dbutils.widgets.get("bronze_schema").strip()
SILVER = dbutils.widgets.get("silver_schema").strip()
GOLD = dbutils.widgets.get("gold_schema").strip()

if not all([CATALOG, BRONZE, SILVER, GOLD]):
    raise ValueError("catalog, bronze_schema, silver_schema and gold_schema are all required")

MODEL_NAME = dbutils.widgets.get("model_name").strip()
BASE_MODEL = dbutils.widgets.get("base_model").strip()
IMAGE_SIZE = int(dbutils.widgets.get("image_size"))
EPOCHS = float(dbutils.widgets.get("epochs"))
BATCH_SIZE = int(dbutils.widgets.get("batch_size"))
LEARNING_RATE = float(dbutils.widgets.get("learning_rate"))
TEST_FRACTION = float(dbutils.widgets.get("test_fraction"))
SEED = int(dbutils.widgets.get("random_seed"))

SOURCE_TABLE = f"{CATALOG}.{SILVER}.training_images"
BYTES_TABLE = f"{CATALOG}.{BRONZE}.training_images"
RESIZED_TABLE = f"{CATALOG}.{SILVER}.training_images_resized"
PREDICTIONS_TABLE = f"{CATALOG}.{GOLD}.damage_predictions"
FULL_MODEL_NAME = f"{CATALOG}.{GOLD}.{MODEL_NAME}"

CURRENT_USER = spark.sql("SELECT current_user()").first()[0]
EXPERIMENT_PATH = (
    dbutils.widgets.get("experiment_path").strip()
    or f"/Users/{CURRENT_USER}/smart_claims/image_claims_classifier"
)

print(f"labels + paths : {SOURCE_TABLE}")
print(f"image bytes    : {BYTES_TABLE}")
print(f"resized        : {RESIZED_TABLE}")
print(f"model          : {FULL_MODEL_NAME}")
print(f"predictions    : {PREDICTIONS_TABLE}")
print(f"experiment     : {EXPERIMENT_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## The input data
# MAGIC
# MAGIC The transcript reads one table — its silver `training_images` carries both the label and
# MAGIC the image bytes. Ours does not, and on purpose: `transformations_silver/training_images.py`
# MAGIC drops `content` so that every join, count and dashboard over silver stops dragging a BLOB
# MAGIC through a shuffle. Bronze still has the bytes, because bronze keeps what arrived.
# MAGIC
# MAGIC Training is the one consumer that actually wants pixels, so it is the one that pays for
# MAGIC them — a join back to bronze on `path`. That is the design working, not a workaround.
# MAGIC
# MAGIC `dropDuplicates` on the bronze side is defensive: bronze is an append-only streaming table
# MAGIC and a re-ingested file would otherwise multiply every training row it matched.

# COMMAND ----------

from pyspark.sql import functions as F

labels_df = spark.table(SOURCE_TABLE).select("path", "file_name", "label")
bytes_df = spark.table(BYTES_TABLE).select("path", "content").dropDuplicates(["path"])

images = labels_df.join(bytes_df, "path", "inner")

n_images = images.count()
print(f"{n_images} labelled images with bytes")
if n_images == 0:
    raise ValueError(
        f"No rows joined between {SOURCE_TABLE} and {BYTES_TABLE}. If the training images "
        "predate part 5 they have no label in the filename, the silver `labelled` expectation "
        "drops them, and this join is empty. Re-seed the volume and re-run the object_storage "
        "pipeline -- the run order is in the README."
    )

display(images.groupBy("label").count().orderBy("label"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resize to what ResNet expects
# MAGIC
# MAGIC *"This model was trained on image sizes of 224, and therefore we need to resize all of our
# MAGIC input images to basically match this preferred size."*
# MAGIC
# MAGIC The image processor loaded later would resize anyway, so this step is not strictly
# MAGIC required for correctness. It earns its place for a different reason: it makes the resize
# MAGIC a **table** rather than a step hidden inside a training loop. `training_images_resized` is
# MAGIC inspectable, diffable and re-readable by the next experiment without re-decoding the
# MAGIC originals — which is the whole argument for materialising a preprocessing step.
# MAGIC
# MAGIC Written to silver rather than gold because that is what it is: a cleaned, standardised
# MAGIC version of a silver dataset. Note that unlike its neighbours it is written by this
# MAGIC notebook, not by the declarative pipeline, so a pipeline full refresh leaves it alone and
# MAGIC re-running this notebook is what refreshes it.

# COMMAND ----------

from PIL import Image
from pyspark.sql.types import BinaryType


@F.udf(returnType=BinaryType())
def resize_image_udf(content):
    if content is None:
        return None
    img = Image.open(io.BytesIO(content)).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    buf = io.BytesIO()
    # JPEG, matching the transcript's "our current data set still has just those JPEG bytes".
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


(
    images.select(
        "path",
        "file_name",
        "label",
        resize_image_udf(F.col("content")).alias("content"),
    )
    .write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(RESIZED_TABLE)
)

print(f"wrote {spark.table(RESIZED_TABLE).count()} rows to {RESIZED_TABLE}")
display(spark.table(RESIZED_TABLE).select("file_name", "label").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow experiment
# MAGIC
# MAGIC The transcript creates this by hand in the UI. Doing it in code means a fresh workspace
# MAGIC reproduces it, and the experiment is named rather than defaulting to the notebook path —
# MAGIC which matters once the same notebook runs from a job, where the default would scatter runs
# MAGIC across per-run notebook copies.
# MAGIC
# MAGIC `set_experiment` does **not** create missing parent folders, so the `mkdirs` is not
# MAGIC decoration: without it the first run of a fresh checkout fails with
# MAGIC `NOT_FOUND: Parent directory does not exist`.
# MAGIC
# MAGIC `set_registry_uri("databricks-uc")` is the transcript's *"one more thing that we forgot
# MAGIC here — we need to specify that Unity Catalog is basically our model registry"*. Without
# MAGIC it the model silently lands in the deprecated workspace registry, which has no catalog, no
# MAGIC grants and no lineage.

# COMMAND ----------

import mlflow
from databricks.sdk import WorkspaceClient
from mlflow.tracking import MlflowClient

WorkspaceClient().workspace.mkdirs(str(Path(EXPERIMENT_PATH).parent))

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_PATH)

LIB_VERSIONS = {
    lib: version(lib) for lib in ("mlflow", "torch", "transformers", "datasets", "pillow")
}
print(json.dumps(LIB_VERSIONS, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Split
# MAGIC
# MAGIC 56 images is small enough to collect to the driver, and small enough that the split
# MAGIC matters: `stratify_by_column` keeps the class balance in the held-out set, which random
# MAGIC splitting of ~19 images per class does not reliably do.

# COMMAND ----------

from datasets import ClassLabel, Dataset

pdf = spark.table(RESIZED_TABLE).select("file_name", "label", "content").toPandas()
LABELS = sorted(pdf["label"].unique().tolist())
print(f"{len(pdf)} rows, {len(LABELS)} classes: {LABELS}")

dataset = Dataset.from_pandas(pdf, preserve_index=False).cast_column(
    "label", ClassLabel(names=LABELS)
)
splits = dataset.train_test_split(
    test_size=TEST_FRACTION, seed=SEED, stratify_by_column="label"
)
print(f"train {splits['train'].num_rows}  test {splits['test'].num_rows}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pre-trained model and its pre-processing
# MAGIC
# MAGIC *"Even though we resized the image already, our current data set still has just those JPEG
# MAGIC bytes, and neural networks do not understand this — they work with tensors, and on top of
# MAGIC that our ResNet model expects the input to be normalized."*
# MAGIC
# MAGIC The processor that ships with the checkpoint holds the exact normalisation statistics it
# MAGIC was trained with, so it is used rather than reimplemented — getting the mean and standard
# MAGIC deviation subtly wrong is a classic way to produce a model that trains without error and
# MAGIC predicts badly.
# MAGIC
# MAGIC `ignore_mismatched_sizes=True` is what makes this a fine-tune: the checkpoint's 1000-way
# MAGIC ImageNet classification head is discarded and replaced with a freshly initialised
# MAGIC `len(LABELS)`-way head, while every learned layer beneath it is kept. Without the flag the
# MAGIC load fails on the shape mismatch instead.
# MAGIC
# MAGIC `with_transform` decodes lazily, per batch, so the images are never all in memory as
# MAGIC tensors at once. It also means `remove_unused_columns=False` is mandatory below — the
# MAGIC Trainer would otherwise drop `content` before the transform ever sees it.

# COMMAND ----------

import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

processor = AutoImageProcessor.from_pretrained(BASE_MODEL)

model = AutoModelForImageClassification.from_pretrained(
    BASE_MODEL,
    num_labels=len(LABELS),
    id2label={i: label for i, label in enumerate(LABELS)},
    label2id={label: i for i, label in enumerate(LABELS)},
    ignore_mismatched_sizes=True,
)


def to_tensors(batch):
    images = [Image.open(io.BytesIO(c)).convert("RGB") for c in batch["content"]]
    encoded = processor(images, return_tensors="pt")
    # `list(...)` splits the processor's stacked (B, 3, H, W) tensor back into B tensors of
    # (3, H, W). It matters: the DataLoader samples ONE index at a time, so `datasets` has to
    # unbatch whatever this returns, and it can only do that for a sequence -- handed a single
    # stacked tensor it would give every example the whole batch.
    return {"pixel_values": list(encoded["pixel_values"]), "labels": batch["label"]}


splits = splits.with_transform(to_tensors)


def collate(examples):
    # `as_tensor` rather than assuming torch tensors survive the transform: whether `datasets`
    # hands these back as tensors or as nested lists depends on its version, and this costs
    # nothing when they are already tensors.
    return {
        "pixel_values": torch.stack([torch.as_tensor(e["pixel_values"]) for e in examples]),
        "labels": torch.tensor([e["labels"] for e in examples]),
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fine-tune
# MAGIC
# MAGIC *"Typically in a more production scenario we would also have this hyperparameter tuning
# MAGIC step here in between. But for demo purposes, we keep it simple."* Same here — one set of
# MAGIC hyperparameters, exposed as widgets so a sweep can be driven from the job rather than by
# MAGIC editing this file.
# MAGIC
# MAGIC Two deliberate choices in `TrainingArguments`:
# MAGIC
# MAGIC - **`report_to=["mlflow"]`** is what puts the per-step training loss into the run. It logs
# MAGIC   into the *active* run rather than starting its own, which is why the `start_run` below
# MAGIC   wraps the whole thing.
# MAGIC - **No `eval_strategy`.** That argument was renamed from `evaluation_strategy` in
# MAGIC   transformers 4.46, and since the install above is unpinned, naming either spelling would
# MAGIC   make this notebook version-fragile. Evaluating once at the end with `trainer.evaluate()`
# MAGIC   is equivalent for a five-epoch run and survives both versions.

# COMMAND ----------

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, TrainingArguments


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(eval_pred.label_ids, predictions),
        "f1_macro": f1_score(eval_pred.label_ids, predictions, average="macro"),
    }


training_args = TrainingArguments(
    output_dir="/tmp/damage_classifier_trainer",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    seed=SEED,
    logging_steps=1,
    save_strategy="no",
    remove_unused_columns=False,
    report_to=["mlflow"],
)

# COMMAND ----------

with mlflow.start_run(run_name="resnet-finetune") as run:
    RUN_ID = run.info.run_id

    # Only the things the Trainer does NOT already know about. `report_to=["mlflow"]` logs every
    # TrainingArguments field into this same run when train() starts -- including learning_rate,
    # seed, num_train_epochs and per_device_train_batch_size -- and MLflow rejects a second,
    # different value for a parameter already set. Logging them here too would either duplicate
    # or collide depending on how each side stringifies a float.
    #
    # What is left is what the Trainer cannot see: which data, which checkpoint, which library
    # versions. Those are what make this run reconstructable a year from now.
    mlflow.log_params({
        "base_model": BASE_MODEL,
        "image_size": IMAGE_SIZE,
        "test_fraction": TEST_FRACTION,
        "labels": ",".join(LABELS),
        "n_train": splits["train"].num_rows,
        "n_test": splits["test"].num_rows,
        "source_table": RESIZED_TABLE,
        **{f"version_{k}": v for k, v in LIB_VERSIONS.items()},
    })

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=splits["train"],
        eval_dataset=splits["test"],
        data_collator=collate,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    metrics = trainer.evaluate()
    mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
    print(json.dumps(metrics, indent=2))

    # Saved in the checkpoint's own on-disk format, processor alongside it. The pyfunc below
    # reloads exactly this directory, so training-time and serving-time pre-processing cannot
    # drift apart -- they are literally the same files.
    ARTIFACT_DIR = "/tmp/damage_classifier_model"
    trainer.save_model(ARTIFACT_DIR)
    processor.save_pretrained(ARTIFACT_DIR)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wrap it so it can be given an image
# MAGIC
# MAGIC *"We basically wrap the model in this format that you can see right here, so that later on
# MAGIC we can first of all easily register it as an MLflow model in the Unity Catalog model
# MAGIC registry, and we can also easily then load it back and do inference on top of it."*
# MAGIC
# MAGIC The wrapper exists because of what the callers hold. A Spark UDF and a serving endpoint
# MAGIC both have **image bytes** — a `binaryFile` column, or a base64 body. Neither has a
# MAGIC normalised float tensor. Logging the bare `transformers` model would push the decode and
# MAGIC the normalisation onto every caller, and the first one to get the normalisation wrong
# MAGIC would get quietly bad predictions rather than an error.
# MAGIC
# MAGIC Written to a file and logged with `python_model="<path>"` — the *Models from Code* pattern
# MAGIC — rather than passing a class instance. An instance would be pickled, and an unpickle
# MAGIC across a Python version boundary (training on the ML runtime, loading on a serving
# MAGIC endpoint) is a well-worn way to produce a model that registers fine and refuses to load.

# COMMAND ----------

PYFUNC_SOURCE = '''
import io

import mlflow
import pandas as pd
from mlflow.pyfunc import PythonModel


class DamageClassifier(PythonModel):
    # Accepts a single column of raw image bytes and returns the predicted severity label.
    # Nulls pass through as None instead of failing the batch: one unreadable upload in a
    # million-row scoring job should cost that row, not the job.

    def load_context(self, context):
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        path = context.artifacts["model"]
        self.torch = torch
        self.processor = AutoImageProcessor.from_pretrained(path)
        self.model = AutoModelForImageClassification.from_pretrained(path)
        self.model.eval()

    def predict(self, context, model_input, params=None):
        from PIL import Image

        if isinstance(model_input, pd.DataFrame):
            blobs = model_input[model_input.columns[0]].tolist()
        else:
            blobs = list(model_input)

        images, positions = [], []
        for position, blob in enumerate(blobs):
            if blob is None or len(blob) == 0:
                continue
            images.append(Image.open(io.BytesIO(blob)).convert("RGB"))
            positions.append(position)

        out = [None] * len(blobs)
        if not images:
            return pd.Series(out, dtype="object")

        inputs = self.processor(images, return_tensors="pt")
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
        for position, index in zip(positions, logits.argmax(-1).tolist()):
            out[position] = self.model.config.id2label[index]
        return pd.Series(out, dtype="object")


mlflow.models.set_model(DamageClassifier())
'''

PYFUNC_PATH = "/tmp/damage_classifier_pyfunc.py"
Path(PYFUNC_PATH).write_text(PYFUNC_SOURCE)
print(f"wrote {PYFUNC_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register into Unity Catalog
# MAGIC
# MAGIC *"Now we register a new model using mlflow.register_model. So this is registering it in our
# MAGIC Unity Catalog, and we specify the name — we will store this in the gold schema and call it
# MAGIC claims damage level. And we will also give it an alias that this is the prod model."*
# MAGIC
# MAGIC The alias is the part worth dwelling on. UC dropped MLflow's old `Staging`/`Production`
# MAGIC *stages*; `@prod` is a movable pointer instead. Everything downstream loads
# MAGIC `models:/...@prod` and never names a version number, so promoting a better model later is
# MAGIC one `set_registered_model_alias` call and no consumer changes.
# MAGIC
# MAGIC `pip_requirements` is pinned here even though the install was not, using the versions
# MAGIC actually resolved at the top. A serving endpoint rebuilds its environment from this list,
# MAGIC and an unpinned list is how a model that worked in May stops loading in August.

# COMMAND ----------

import inspect

import pandas as pd
from mlflow.models import infer_signature

sample_input = pd.DataFrame({"content": [pdf["content"].iloc[0]]})
sample_output = pd.Series([LABELS[0]], dtype="object")

# MLflow 3 spells this `name=`; MLflow 2 spells it `artifact_path=`. Which one ships is decided
# by the cluster's ML runtime, so this is resolved from the installed signature rather than bet
# on a DBR version -- the same reasoning as the unpinned installs at the top.
artifact_kwarg = (
    {"name": "model"}
    if "name" in inspect.signature(mlflow.pyfunc.log_model).parameters
    else {"artifact_path": "model"}
)

with mlflow.start_run(run_id=RUN_ID):
    logged = mlflow.pyfunc.log_model(
        **artifact_kwarg,
        python_model=PYFUNC_PATH,
        artifacts={"model": ARTIFACT_DIR},
        signature=infer_signature(sample_input, sample_output),
        input_example=sample_input,
        pip_requirements=[
            f"mlflow=={LIB_VERSIONS['mlflow']}",
            f"torch=={LIB_VERSIONS['torch']}",
            f"transformers=={LIB_VERSIONS['transformers']}",
            f"pillow=={LIB_VERSIONS['pillow']}",
            "pandas",
        ],
        registered_model_name=FULL_MODEL_NAME,
    )

client = MlflowClient(registry_uri="databricks-uc")
MODEL_VERSION = getattr(logged, "registered_model_version", None)
if MODEL_VERSION is None:
    # Older MLflow returns no version on the ModelInfo. Highest version wins, which is safe here
    # because the registration above is the only writer and has already completed.
    MODEL_VERSION = max(
        client.search_model_versions(f"name='{FULL_MODEL_NAME}'"), key=lambda v: int(v.version)
    ).version

client.set_registered_model_alias(FULL_MODEL_NAME, "prod", MODEL_VERSION)
print(f"registered {FULL_MODEL_NAME} version {MODEL_VERSION}, alias @prod")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch inference
# MAGIC
# MAGIC *"To do batch inference within Databricks, what you can do is you can register a Spark
# MAGIC UDF."* One artifact, two consumption paths — the same registered model that a serving
# MAGIC endpoint would load is here turned into a UDF and run across a Delta table in parallel.
# MAGIC
# MAGIC Scored against the **original** images from bronze, not the resized table, exactly as the
# MAGIC transcript does (*"we use our silver initial input images"*). The wrapper's own processor
# MAGIC resizes them, which is the point of having wrapped it: callers hand over whatever bytes
# MAGIC they have.
# MAGIC
# MAGIC `env_manager="local"` reuses this cluster's environment instead of rebuilding the model's
# MAGIC from `pip_requirements`. Correct here because scoring and training are the same session;
# MAGIC scoring from a *different* runtime needs `"virtualenv"` or `"uv"` instead.

# COMMAND ----------

predict_damage_udf = mlflow.pyfunc.spark_udf(
    spark,
    model_uri=f"models:/{FULL_MODEL_NAME}@prod",
    env_manager="local",
    result_type="string",
)

(
    images.withColumn("damage_prediction", predict_damage_udf(F.col("content")))
    .select("path", "file_name", "label", "damage_prediction")
    .withColumn("scored_at", F.current_timestamp())
    .write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(PREDICTIONS_TABLE)
)

predictions = spark.table(PREDICTIONS_TABLE)
print(f"wrote {predictions.count()} rows to {PREDICTIONS_TABLE}")
display(predictions.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Confusion matrix
# MAGIC
# MAGIC Printed as text as well as plotted, and logged to the MLflow run — a job run has nobody
# MAGIC watching `display()`, and the figure is part of the evidence for this model version rather
# MAGIC than a thing that scrolls past once.
# MAGIC
# MAGIC **Expect a roughly uniform matrix.** See the warning at the top: the labels are real and
# MAGIC the pixels are random noise, so chance-level performance is the honest outcome, and a
# MAGIC strong diagonal would be evidence of leakage rather than of learning.

# COMMAND ----------

import matplotlib.pyplot as plt

results = predictions.select("label", "damage_prediction").toPandas()
matrix = pd.crosstab(
    results["label"], results["damage_prediction"], rownames=["actual"], colnames=["predicted"]
).reindex(index=LABELS, columns=LABELS, fill_value=0)

accuracy = float((results["label"] == results["damage_prediction"]).mean())
baseline = 1.0 / len(LABELS)

print(matrix.to_string())
print(f"\naccuracy {accuracy:.3f} vs {baseline:.3f} for guessing among {len(LABELS)} classes")
print(
    "Close to the guessing baseline is the CORRECT result on this synthetic data -- the image "
    "colours are drawn independently of the labels, so there is no signal to learn."
)

fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.imshow(matrix.values, cmap="Blues")
ax.set_xticks(range(len(LABELS)), LABELS, rotation=30, ha="right")
ax.set_yticks(range(len(LABELS)), LABELS)
ax.set_xlabel("predicted")
ax.set_ylabel("actual")
ax.set_title(f"{MODEL_NAME} v{MODEL_VERSION} -- accuracy {accuracy:.2f}")
for i in range(len(LABELS)):
    for j in range(len(LABELS)):
        ax.text(j, i, matrix.values[i, j], ha="center", va="center", color="black")
fig.tight_layout()

with mlflow.start_run(run_id=RUN_ID):
    mlflow.log_figure(fig, "confusion_matrix.png")
    mlflow.log_metric("batch_accuracy", accuracy)
    mlflow.log_metric("guessing_baseline", baseline)

display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Result
# MAGIC
# MAGIC `dbutils.notebook.exit` is how a job run hands structured output back — `print` is not
# MAGIC reliably captured, and the caller can read this from `.notebook_output.result`.
# MAGIC
# MAGIC Not done here, deliberately: the real-time serving endpoint. The transcript creates one at
# MAGIC this point, and it bills for as long as it exists. The model is registered and aliased, so
# MAGIC creating one later is a single call against `models:/{...}@prod` with nothing here to
# MAGIC change.

# COMMAND ----------

dbutils.notebook.exit(json.dumps({
    "model": FULL_MODEL_NAME,
    "model_version": MODEL_VERSION,
    "run_id": RUN_ID,
    "eval_accuracy": metrics.get("eval_accuracy"),
    "batch_accuracy": accuracy,
    "guessing_baseline": baseline,
    "rows_scored": int(predictions.count()),
    "predictions_table": PREDICTIONS_TABLE,
}))
