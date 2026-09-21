#!/usr/bin/env python3
"""Regenerate resources/claims_investigation.geniespace.yml from its JSON source of truth.

Genie has no equivalent of the dashboard's `dataset_catalog` / `dataset_schema` fields -- every
table is named in full inside the payload -- so the space must be inlined as YAML for the bundle
to interpolate it. YAML is not a pleasant format to hand-edit at this size, hence this script:
edit src/genie/claims_investigation.genie.json, run this, commit both.

    python3 scripts/build_genie_space.py

The JSON holds the concrete dev identifier so it stays directly testable against a workspace; the
substitution below is what turns it into a per-target reference on the way into the bundle.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "src/genie/claims_investigation.genie.json"
TARGET = ROOT / "resources/claims_investigation.geniespace.yml"

TABLE = "customer_claim_policy_telematics"
FQN_DEV = f"smart_claims_dev.dev_keqingli1129_gold.{TABLE}"
FQN_VAR = "${var.catalog}.${resources.schemas.gold.name}." + TABLE


def retarget(node):
    """Swap the concrete dev identifier for the bundle's per-target one, everywhere it appears.

    Recursive rather than a single top-level fix: the identifier shows up in the data source AND
    inside every example SQL string, and missing one would leave the space half-pinned to dev.
    """
    if isinstance(node, str):
        return node.replace(FQN_DEV, FQN_VAR)
    if isinstance(node, list):
        return [retarget(x) for x in node]
    if isinstance(node, dict):
        return {k: retarget(v) for k, v in node.items()}
    return node


def emit(node, indent):
    """Minimal YAML writer for the dict/list/scalar subset this payload uses.

    Every scalar goes out `json.dumps`-encoded. YAML is a superset of JSON, so that is valid, and
    it removes any question of quoting or escaping in the long description strings -- several
    contain colons and percent signs that bare YAML scalars would mangle.
    """
    pad = "  " * indent
    out = []
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, (dict, list)) and value:
                out.append(f"{pad}{key}:")
                out.extend(emit(value, indent + 1))
            else:
                out.append(f"{pad}{key}: {json.dumps(value)}")
    elif isinstance(node, list):
        for item in node:
            if isinstance(item, dict):
                lines = emit(item, indent + 1)
                out.append(f"{pad}- {lines[0].lstrip()}")
                out.extend(lines[1:])
            else:
                out.append(f"{pad}- {json.dumps(item)}")
    return out


def main() -> None:
    space = retarget(json.loads(SOURCE.read_text()))
    header = TARGET.read_text().split("      serialized_space:\n")[0]
    TARGET.write_text(header + "      serialized_space:\n" + "\n".join(emit(space, 4)) + "\n")
    print(f"regenerated {TARGET.relative_to(ROOT)} from {SOURCE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
