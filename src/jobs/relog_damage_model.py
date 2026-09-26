# Databricks notebook source
# MAGIC %md
# MAGIC # Re-log the damage classifier with a corrected dependency list
# MAGIC
# MAGIC **A one-off repair, already applied.** It is in the repository because it is the only
# MAGIC record of how version 8 of `claims_damage_level` came to exist, and
# MAGIC `resources/damage_classifier.serving.yml` pins that version. Without this file a fresh
# MAGIC clone would reference a version nothing in the repo can produce.
# MAGIC
# MAGIC **It is deliberately NOT a job resource.** Nothing should run it on a schedule; it exists
# MAGIC to be read, and to be re-run by hand if the same fault recurs. Run it from the workspace
# MAGIC after `bundle deploy` has uploaded it.
# MAGIC
# MAGIC ## What went wrong
# MAGIC
# MAGIC Version 7 could not be served. The endpoint failed to start with
# MAGIC `DEPLOYMENT_FAILED -- A Python import chain failed during model load`, and the real error,
# MAGIC recovered by loading the model in a notebook pinned to its own recorded versions, was:
# MAGIC
# MAGIC ```
# MAGIC ImportError: AutoImageProcessor requires the Torchvision library
# MAGIC ```
# MAGIC
# MAGIC The training notebook's `%pip` cell installs torch, torchvision, transformers, datasets,
# MAGIC accelerate and scikit-learn. Its `pip_requirements` recorded only mlflow, torch,
# MAGIC transformers, pillow and pandas. **torchvision fell in the gap**: provided by the training
# MAGIC runtime, absent from a serving container, and therefore invisible until something with a
# MAGIC smaller environment tried to load the model.
# MAGIC
# MAGIC `train_damage_classifier.py` has since been fixed, so a retrain records torchvision. This
# MAGIC notebook repairs the EXISTING version instead.
# MAGIC
# MAGIC ## Why re-log rather than retrain
# MAGIC
# MAGIC v7's weights are fine; only its metadata is wrong. A retrain would fix the metadata and
# MAGIC also produce different weights -- it is a real retrain on 56 images with a 25% split, so
# MAGIC the predictions would move. Re-logging copies the artifacts verbatim and changes only what
# MAGIC is recorded about them, so nothing downstream shifts.
# MAGIC
# MAGIC `accelerate` is also undeclared. It is deliberately NOT added: adding only torchvision was
# MAGIC tested and the model loads, so nothing shows accelerate is needed.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Environment
# MAGIC
# MAGIC Two reasons these pins are not optional:
# MAGIC
# MAGIC 1. **mlflow must match the version that logged the model.** The stock serverless notebook
# MAGIC    image ships mlflow 2.11.4 on Python 3.11, and every Unity Catalog artifact download
# MAGIC    fails with `HTTP 400 Bad Request` on HeadObject. A diagnostic that skips this never
# MAGIC    reaches the code under test. (`databricks fs ls` on the artifact path is no help either
# MAGIC    -- "Public DBFS root is disabled".)
# MAGIC 2. **torchvision must be present** to load the model at all, which is how this notebook
# MAGIC    proves the repair before moving the alias.

# COMMAND ----------

# MAGIC %pip install --quiet mlflow==3.8.1 torch==2.14.0 torchvision transformers==5.17.0 pillow==11.1.0

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821

# COMMAND ----------

import json
import traceback
from pathlib import Path

import mlflow
from mlflow.models import Model
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")

# Hardcoded rather than taken from widgets: this repairs one specific model in one specific
# schema, and it has already been run. Edit it if you are repairing something else.
FULL_MODEL_NAME = "smart_claims_dev.dev_keqingli1129_gold.claims_damage_level"
SOURCE_URI = f"models:/{FULL_MODEL_NAME}@prod"

report = {"source": SOURCE_URI}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Pull the current @prod version down whole

# COMMAND ----------

local_dir = mlflow.artifacts.download_artifacts(SOURCE_URI)
report["local_dir"] = local_dir
report["files"] = sorted(
    str(p.relative_to(local_dir)) for p in Path(local_dir).rglob("*") if p.is_file()
)
print("\n".join(report["files"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Add torchvision to the recorded requirements
# MAGIC
# MAGIC The existing `requirements.txt` is EDITED rather than retyped, so anything else the model
# MAGIC recorded survives untouched. The pin is read from the installed torchvision rather than
# MAGIC written by hand, so it cannot drift from what was actually tested.

# COMMAND ----------

import torchvision

req_path = Path(local_dir) / "requirements.txt"
report["original_requirements"] = req_path.read_text()

TORCHVISION_PIN = f"torchvision=={torchvision.__version__.split('+')[0]}"

lines = [ln.strip() for ln in report["original_requirements"].splitlines() if ln.strip()]
if not any(ln.lower().startswith("torchvision") for ln in lines):
    # Placed directly after torch, where a reader expects to find it.
    insert_at = next(
        (i for i, ln in enumerate(lines) if ln.lower().startswith("torch==")), len(lines) - 1
    )
    lines.insert(insert_at + 1, TORCHVISION_PIN)
report["new_requirements"] = lines
print("\n".join(lines))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Log it again
# MAGIC
# MAGIC Same pyfunc source file, same weights directory, corrected requirements. The signature is
# MAGIC CARRIED ACROSS rather than re-inferred -- re-inferring needs a sample image and could
# MAGIC quietly produce a different schema than the model already advertises.

# COMMAND ----------

signature = Model.load(local_dir).signature
report["signature"] = str(signature)

with mlflow.start_run(run_name="relog_with_torchvision") as run:
    mlflow.set_tag("relog_of_version", "7")
    mlflow.set_tag("reason", "pip_requirements omitted torchvision; serving could not load it")
    logged = mlflow.pyfunc.log_model(
        name="model",
        python_model=str(Path(local_dir) / "damage_classifier_pyfunc.py"),
        artifacts={"model": str(Path(local_dir) / "artifacts" / "damage_classifier_model")},
        signature=signature,
        pip_requirements=lines,
        registered_model_name=FULL_MODEL_NAME,
    )
    report["run_id"] = run.info.run_id

NEW_VERSION = getattr(logged, "registered_model_version", None)
if NEW_VERSION is None:
    client = MlflowClient(registry_uri="databricks-uc")
    NEW_VERSION = max(
        int(v.version) for v in client.search_model_versions(f"name='{FULL_MODEL_NAME}'")
    )
report["new_version"] = str(NEW_VERSION)
print("registered version", NEW_VERSION)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Prove it loads BEFORE moving the alias
# MAGIC
# MAGIC The ordering is the point. If the new version cannot load, `@prod` stays where it is and
# MAGIC nothing downstream has been touched.

# COMMAND ----------

try:
    mlflow.pyfunc.load_model(f"models:/{FULL_MODEL_NAME}/{NEW_VERSION}")
    report["load_new_version"] = "OK"
except Exception:
    report["load_new_version"] = "FAILED"
    report["load_traceback"] = traceback.format_exc()[-3000:]
    print(report["load_traceback"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Move @prod, on success only
# MAGIC
# MAGIC **The alias is half the job.** `resources/damage_classifier.serving.yml` pins
# MAGIC `entity_version` explicitly, because the bundle schema has no field for a UC alias. Moving
# MAGIC the alias here does NOT move the endpoint -- that number must be edited and the bundle
# MAGIC redeployed, or the registry and the endpoint quietly disagree about what "production" is.

# COMMAND ----------

if report.get("load_new_version") == "OK":
    MlflowClient(registry_uri="databricks-uc").set_registered_model_alias(
        FULL_MODEL_NAME, "prod", NEW_VERSION
    )
    report["alias_prod"] = f"moved to version {NEW_VERSION}"
    print(f"@prod -> {NEW_VERSION}. NOW EDIT entity_version IN "
          f"resources/damage_classifier.serving.yml AND REDEPLOY.")
else:
    report["alias_prod"] = "LEFT WHERE IT WAS -- the new version failed to load"

# COMMAND ----------

dbutils.notebook.exit(json.dumps(report, indent=2, default=str))  # noqa: F821
