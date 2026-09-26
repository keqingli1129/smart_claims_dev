#!/usr/bin/env python3
"""Grant the Databricks App's service principal read access to the Lakebase synced tables.

    uv run --no-project --with databricks-sdk --with psycopg2-binary \
        python scripts/grant_lakebase_access.py --target dev

WHY THIS SCRIPT HAS TO EXIST -- it is the one piece of the deployment that `bundle deploy` cannot
do for you.

Attaching a resource in resources/smart_claims.app.yml grants Unity Catalog permissions
automatically: the `gold-claims-table` entry gives the service principal SELECT on that table, and
`claims-volume` gives it WRITE_VOLUME, with no GRANT written anywhere. Postgres is different. The
`postgres` resource offers exactly one permission, CAN_CONNECT_AND_CREATE, which lets the SP
connect and create objects OF ITS OWN. It says nothing about reading tables somebody else owns,
and no bundle field expresses that -- Databricks provisions the database but stays out of who may
read what inside it.

Measured on 2026-09-22, with the postgres resource already attached and the app deployed:

    USAGE on schema public   : True     <- from CAN_CONNECT_AND_CREATE
    CREATE on database       : True     <- from CAN_CONNECT_AND_CREATE
    SELECT on synced table   : False    <- nothing had granted this

Without the grant below the app starts cleanly and then fails on its first Lakebase query, which
is a slow way to discover a permissions problem.

Idempotent: GRANT is a no-op when the privilege is already held, so re-running after every deploy
is safe and is the recommended habit.
"""

import argparse
import sys

import psycopg2
from databricks.sdk import WorkspaceClient

# The synced tables the app must be able to read. Deliberately an explicit list rather than
# `GRANT SELECT ON ALL TABLES IN SCHEMA public`: `public` also holds Databricks' own bookkeeping
# tables and the Neon engine's performance counters, and granting across all of them hands the
# app read access to a dozen things it has no business reading. Add a line here when a new
# synced table is added to resources/lakebase.yml.
SYNCED_TABLES = ["public.customer_claim_policy_telematics"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="dev", help="bundle target (dev, prod); default: dev")
    parser.add_argument("--profile", default="DEFAULT", help="Databricks CLI profile")
    args = parser.parse_args()

    # Both names carry the target suffix, because development mode does NOT rename apps or
    # Lakebase projects the way it renames jobs -- see the comments in resources/lakebase.yml.
    app_name = f"smart-claims-{args.target}"
    endpoint = f"projects/smart-claims-{args.target}/branches/production/endpoints/primary"

    w = WorkspaceClient(profile=args.profile)

    # DISCOVERED, never hardcoded. The service principal is created by Databricks when the app is
    # first deployed, so its id is not knowable until then -- and it changes if the app is
    # recreated.
    app = w.apps.get(name=app_name)
    sp = app.service_principal_client_id
    if not sp:
        print(f"ERROR: {app_name} has no service principal yet -- deploy the app first.")
        return 1
    print(f"app          : {app_name}")
    print(f"service princ: {sp}")

    # The endpoint hostname is generated and changes whenever the endpoint is recreated (three
    # different values in one day of building this), so it is read back rather than remembered.
    ep = w.postgres.get_endpoint(endpoint)
    host = ep.status.hosts.host
    # A short-lived OAuth token, ~60 minutes. There is no static password to reuse.
    token = w.postgres.generate_database_credential(endpoint).token
    print(f"endpoint host: {host}")

    # Connects as the caller -- you. Postgres requires a GRANT to come from the object's owner or
    # a superuser, and the project creator is both. The app's own service principal could not run
    # this even though it is the beneficiary.
    with psycopg2.connect(
        host=host, dbname="databricks_postgres", user=w.current_user.me().user_name,
        password=token, port=5432, sslmode="require", connect_timeout=30,
    ) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            print(f"\ngranting to {sp}:")
            cur.execute(f'GRANT USAGE ON SCHEMA public TO "{sp}"')
            print("  USAGE on schema public")
            for table in SYNCED_TABLES:
                cur.execute(f'GRANT SELECT ON {table} TO "{sp}"')
                print(f"  SELECT on {table}")

            print("\nverifying:")
            ok = True
            for table in SYNCED_TABLES:
                cur.execute("SELECT has_table_privilege(%s, %s, 'SELECT')", (sp, table))
                granted = cur.fetchone()[0]
                ok &= granted
                print(f"  SELECT on {table}: {granted}")
                # The write boundary must stay shut. A synced table is maintained by the sync
                # pipeline, and an INSERT against it succeeds at the SQL level while silently
                # corrupting the sync -- verified, it is not blocked by the engine. Read-only
                # grants are what actually prevent it.
                for priv in ("INSERT", "UPDATE", "DELETE"):
                    cur.execute("SELECT has_table_privilege(%s, %s, %s)", (sp, table, priv))
                    if cur.fetchone()[0]:
                        print(f"  WARNING: {sp} has {priv} on {table} -- it can corrupt the sync")
                        ok = False

    print("\nOK" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
