#!/usr/bin/env python3
"""Check the things `git clone && databricks bundle deploy` does NOT reproduce.

    uv run --no-project --with databricks-sdk python scripts/bootstrap.py --target dev

READ-ONLY BY DESIGN. It reports; it does not fix. Each gap below needs either a secret value or a
decision, and a script that silently created either would be worse than one that tells you what is
missing. The remedy for each is printed alongside it.

WHY THESE THREE EXIST AT ALL. Almost everything in this project is declarative -- the pipelines,
the dashboard, the Genie space, the Lakebase project, the app and its resource grants all come
from `bundle deploy`. Three things cannot:

1. THE SECRET SCOPE AND THE OPENAI KEY. The bundle names a scope and key for the assistant, but a
   secret VALUE can never live in git, and declaring an already-existing scope as a bundle
   resource risks a deploy conflict. So the scope is created once by hand.

2. THE LAKEBASE GRANT. Attaching the `postgres` resource grants CAN_CONNECT_AND_CREATE and nothing
   more; reading a table the sync pipeline owns needs a real Postgres GRANT that no bundle field
   expresses. See scripts/grant_lakebase_access.py, which this script only points at.

3. THE SERVED MODEL VERSION. resources/damage_classifier.serving.yml pins `entity_version`
   because the bundle schema has no field for a Unity Catalog alias. A model registry with no
   version 8 -- a fresh workspace, say -- leaves the endpoint pointing at nothing. The repair
   notebook src/jobs/relog_damage_model.py, or a plain retrain, produces one.

The check for (3) is also the drift check worth running after ANY retrain: it compares what @prod
resolves to against what the bundle actually serves, and those two are only kept in step by hand.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _serving_yaml_version() -> str | None:
    """The entity_version the bundle pins, read from the YAML as text.

    Parsed with a regex rather than a YAML loader to keep this script's dependencies to the SDK
    alone -- it is a pre-flight check and should run before anything else is installed.
    """
    path = REPO / "resources" / "damage_classifier.serving.yml"
    if not path.exists():
        return None
    match = re.search(r'^\s*entity_version:\s*"?(\d+)"?', path.read_text(), re.MULTILINE)
    return match.group(1) if match else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="dev", help="bundle target (dev, prod); default: dev")
    parser.add_argument("--profile", default="DEFAULT", help="Databricks CLI profile")
    args = parser.parse_args()

    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient(profile=args.profile)
    scope = f"smart-claims-{args.target}"
    problems = 0

    print(f"Checking target '{args.target}' via profile '{args.profile}'\n")

    # --- 1. secret scope + key -----------------------------------------------------------------
    # list_secrets returns NAMES and timestamps only, never values -- enough to verify the key is
    # there without ever reading it.
    try:
        scopes = {s.name for s in w.secrets.list_scopes()}
        if scope not in scopes:
            print(f"  MISSING  secret scope '{scope}'")
            print(f"           databricks secrets create-scope {scope} --profile {args.profile}")
            problems += 1
        else:
            keys = {k.key for k in w.secrets.list_secrets(scope)}
            if "openai-api-key" not in keys:
                print(f"  MISSING  key 'openai-api-key' in scope '{scope}'")
                print(f"           databricks secrets put-secret {scope} openai-api-key "
                      f"--profile {args.profile}")
                problems += 1
            else:
                print(f"  ok       secret scope '{scope}' has 'openai-api-key'")
    except Exception as err:  # noqa: BLE001
        print(f"  ERROR    could not check secrets: {err}")
        problems += 1

    # --- 2. the Lakebase grant -----------------------------------------------------------------
    # Not verified here: proving it would mean connecting to Postgres, which needs psycopg and a
    # minted credential. grant_lakebase_access.py already does that AND asserts the result, so
    # this points at it rather than half-reimplementing it.
    print("\n  MANUAL   the Lakebase GRANT is not reproduced by bundle deploy")
    print("           uv run --no-project --with databricks-sdk --with psycopg2-binary \\")
    print(f"               python scripts/grant_lakebase_access.py --target {args.target}")
    print("           (idempotent -- safe and advisable after every deploy)")

    # --- 3. served model version vs @prod ------------------------------------------------------
    pinned = _serving_yaml_version()
    model = "smart_claims_dev.dev_keqingli1129_gold.claims_damage_level"
    if pinned is None:
        print("\n  ERROR    could not read entity_version from the serving resource")
        problems += 1
    else:
        try:
            prod = w.model_versions.get_by_alias(full_name=model, alias="prod")
            prod_version = str(prod.version)
            if prod_version == pinned:
                print(f"\n  ok       serving pins v{pinned} and @prod resolves to v{prod_version}")
            else:
                # Not fatal, and not always wrong -- but it does mean the endpoint is serving a
                # different model than the registry calls production, which is worth saying out
                # loud rather than discovering through a prediction that looks odd.
                print(f"\n  DRIFT    serving pins v{pinned} but @prod is v{prod_version}")
                print("           the alias moved without the bundle; edit entity_version in")
                print("           resources/damage_classifier.serving.yml and redeploy")
                problems += 1
        except Exception as err:  # noqa: BLE001
            print(f"\n  MISSING  no @prod alias on {model} ({type(err).__name__})")
            print("           run src/jobs/relog_damage_model.py, or the train_damage_classifier")
            print("           job, to produce and alias a version")
            problems += 1

    print(f"\n{problems} item(s) need attention." if problems else "\nNothing outstanding.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
