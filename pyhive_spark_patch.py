# pyhive_spark_patch.py
from sqlalchemy import text
from pyhive import sqlalchemy_hive

# Set this to True if you want to completely skip fetching view SQL
SKIP_VIEW_DDL = False

def _spark_get_table_names(self, connection, schema=None, **kw):
    if not schema:
        return []
    # Spark: SHOW TABLES IN <db> returns (database, tableName, isTemporary)
    rows = connection.execute(text(f"SHOW TABLES IN `{schema}`")).fetchall()
    return [r[1] if len(r) >= 2 else r[0] for r in rows]

def _spark_get_view_names(self, connection, schema=None, **kw):
    if not schema:
        return []
    rows = connection.execute(text(f"SHOW VIEWS IN `{schema}`")).fetchall()
    return [r[1] if len(r) >= 2 else r[0] for r in rows]

def _spark_get_view_definition(self, connection, view_name, schema=None, **kw):
    if SKIP_VIEW_DDL:
        return None  # stop Superset from ever issuing a DDL probe
    if not view_name:
        return None
    ident = f"`{schema}`.`{view_name}`" if schema else f"`{view_name}`"
    try:
        # Spark supports SHOW CREATE TABLE for both tables and views
        rows = connection.execute(text(f"SHOW CREATE TABLE {ident}")).fetchall()
        if not rows:
            return None
        # Some stacks return 1 long row; others return many rows (one per line)
        return "\n".join(str(r[0]) for r in rows if r and r[0])
    except Exception:
        return None

# Monkey-patch PyHive's dialect methods
sqlalchemy_hive.HiveDialect.get_table_names = _spark_get_table_names
sqlalchemy_hive.HiveDialect.get_view_names = _spark_get_view_names
sqlalchemy_hive.HiveDialect.get_view_definition = _spark_get_view_definition
