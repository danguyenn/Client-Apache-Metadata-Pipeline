# superset_config.py
import os
import logging
import pyhive_spark_patch  # keep your existing patch import

# 1) Keep Superset's internal metadata DB on Postgres
SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL")

# 2) Spark Thrift host/port for Iceberg access
SPARK_HOST = os.environ.get("SPARK_HOST", "spark-thrift-server")
SPARK_PORT = os.environ.get("SPARK_PORT", "10000")

# 3) Register Iceberg-on-Spark as a secondary database for SQL Lab
# Connect to the default schema first, then you can switch to nyc in queries
DATABASES = {
    "iceberg_thrift": {
        "name": "Iceberg via Spark Thrift",
        # Connect to default schema for compatibility
        "SQLALCHEMY_URI": f"hive://{SPARK_HOST}:{SPARK_PORT}/default?auth=NOSASL",
        "engine_params": {
            "connect_args": {
                "auth": "NOSASL",
            },
        },
        "expose_in_sqllab": True,
    }
}

# --- make charts/dashboards wait longer before showing the 60s warning ---
SUPERSET_WEBSERVER_TIMEOUT = 60000  # seconds (e.g., 10 minutes)

# --- SQL Lab: sync + async limits ---
# Sync SQL Lab queries (when not using Celery). Still constrained by the webserver/proxy:
SQLLAB_TIMEOUT = 60000  # seconds

# If/when you enable async SQL Lab (Celery), allow long-running queries:
SQLLAB_ASYNC_TIME_LIMIT_SEC = 60 * 60 * 6  # 6 hours

# --- Spark/Hive: rewrite unsupported SHOW CREATE VIEW to SHOW CREATE TABLE ---
from sqlalchemy import event
from sqlalchemy.engine import Engine

_logger = logging.getLogger(__name__)

def _hive_rewrite_show_create_view(conn, cursor, statement, parameters, context, executemany):
    """
    Spark SQL doesn't support `SHOW CREATE VIEW`. Superset sometimes issues it
    when fetching metadata. This hook rewrites it to `SHOW CREATE TABLE`, which
    Spark understands for both tables and Iceberg views.
    """
    try:
        eng = getattr(conn, "engine", None)
        if eng and getattr(eng, "name", "") == "hive":
            s = statement.lstrip()
            up = s.upper()
            prefix = "SHOW CREATE VIEW"
            if up.startswith(prefix):
                fixed = "SHOW CREATE TABLE" + s[len(prefix):]
                # reflect the change back into SQLAlchemy if possible
                if context is not None and hasattr(context, "statement"):
                    context.statement = fixed
                return fixed, parameters
    except Exception as ex:
        _logger.warning("[spark-patch] SHOW CREATE VIEW rewrite skipped due to: %r", ex)
    return statement, parameters

# Register once at import time
event.listen(Engine, "before_cursor_execute", _hive_rewrite_show_create_view, retval=True)
_logger.info("[spark-patch] Registered SHOW CREATE VIEW → SHOW CREATE TABLE rewriter for Hive engines")

# Additional debugging settings
# SQLLAB_TIMEOUT = 300
# SQLLAB_ASYNC_TIME_LIMIT_SEC = 300

# Enable logging for debugging
# logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
# logging.getLogger('pyhive').setLevel(logging.DEBUG)
