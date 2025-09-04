#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] applying database migrations..."
superset db upgrade

# ------------------------------------------------------------------
# Ensure an admin user exists (env-overridable; sensible defaults)
# ------------------------------------------------------------------
ADMIN_USERNAME="${ADMIN_USERNAME:-admin}"
ADMIN_FIRSTNAME="${ADMIN_FIRSTNAME:-Superset}"
ADMIN_LASTNAME="${ADMIN_LASTNAME:-Admin}"
ADMIN_EMAIL="${ADMIN_EMAIL:-admin@example.com}"
ADMIN_PASSWORD="${ADMIN_PASSWORD:-admin}"

echo "[entrypoint] creating admin user if missing..."
if ! superset fab list-users | grep -q "^${ADMIN_USERNAME}\s"; then
  superset fab create-admin \
    --username "${ADMIN_USERNAME}" \
    --firstname "${ADMIN_FIRSTNAME}" \
    --lastname "${ADMIN_LASTNAME}" \
    --email "${ADMIN_EMAIL}" \
    --password "${ADMIN_PASSWORD}"
else
  echo "[entrypoint] admin already exists"
fi

# ------------------------------------------------------------------
# Build the Spark/Thrift URI for PyHive (binary, NOSASL)
# and ensure a Database entry exists/has the correct URI
# ------------------------------------------------------------------
SPARK_HOST="${SPARK_HOST:-spark-thrift-server}"
SPARK_PORT="${SPARK_PORT:-10000}"
export SPARK_URI="hive://${SPARK_HOST}:${SPARK_PORT}/default?auth=NOSASL"

echo "[entrypoint] ensuring 'Iceberg via Spark Thrift' database exists (${SPARK_URI})..."

# Use Superset app context non-interactively (NOT `superset shell`)
python - <<'PY'
import os
from superset.app import create_app

app = create_app()
with app.app_context():
    from superset import db
    from superset.models.core import Database

    NAME = "Iceberg via Spark Thrift"
    URI  = os.environ.get("SPARK_URI")

    existing = db.session.query(Database).filter_by(database_name=NAME).one_or_none()
    if existing:
        if existing.sqlalchemy_uri != URI:
            existing.set_sqlalchemy_uri(URI)
            db.session.commit()
            print("[entrypoint] updated database URI for:", NAME)
        else:
            print("[entrypoint] database already present with correct URI:", NAME)
    else:
        new_db = Database(database_name=NAME)
        new_db.set_sqlalchemy_uri(URI)
        db.session.add(new_db)
        db.session.commit()
        print("[entrypoint] created database:", NAME)
PY

echo "[entrypoint] initializing Superset..."
superset init

# If args were provided (e.g., override to run a worker), exec them
if [[ $# -gt 0 ]]; then
  echo "[entrypoint] exec CMD: $*"
  exec "$@"
fi

# Otherwise, launch the Superset dev webserver so the container stays up
# (Use Gunicorn in production; this is convenient for local/dev)
echo "[entrypoint] launching Superset webserver..."
exec superset run -h 0.0.0.0 -p "${SUPERSET_PORT:-8088}" --with-threads --reload --debugger
