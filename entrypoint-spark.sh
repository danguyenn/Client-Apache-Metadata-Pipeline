#!/usr/bin/env bash
set -euo pipefail

# =========================================
# Spark Thrift Server (Iceberg) entrypoint
# =========================================
# Tunables (override via env)
SPARK_LOCAL_THREADS="${SPARK_LOCAL_THREADS:-2}"               # modest local concurrency
SPARK_DRIVER_MEMORY="${SPARK_DRIVER_MEMORY:-8g}"              # driver heap
SPARK_DRIVER_OVERHEAD="${SPARK_DRIVER_OVERHEAD:-1g}"          # off-heap/overhead
SPARK_DEFAULT_CATALOG="${SPARK_DEFAULT_CATALOG:-iceberg}"     # default Spark SQL catalog
SPARK_MASTER="${SPARK_MASTER:-local[${SPARK_LOCAL_THREADS}]}" # or local[*]/cluster URL

WAREHOUSE="${WAREHOUSE_URI:-file:///warehouse}"
THRIFT_HOST="${THRIFT_HOST:-0.0.0.0}"
THRIFT_PORT="${THRIFT_PORT:-10000}"

# Optional: skip importer (set RUN_IMPORTER=0 to skip)
RUN_IMPORTER="${RUN_IMPORTER:-1}"
IMPORTER_JAR="${IMPORTER_JAR:-/opt/app/target/iceberg-importer-1.0.0.jar}"
IMPORTER_INPUT_PATH="${IMPORTER_INPUT_PATH:-/data}"
IMPORTER_OUTPUT_NS="${IMPORTER_OUTPUT_NS:-nyc}"

# Optional: Spark UI toggles
SPARK_UI_ENABLED="${SPARK_UI_ENABLED:-true}"
SPARK_UI_PORT="${SPARK_UI_PORT:-4040}"

# Shuffle/scan knobs
SPARK_SQL_SHUFFLE_PARTITIONS="${SPARK_SQL_SHUFFLE_PARTITIONS:-64}"
SPARK_SQL_ADVISORY_PARTITION_SIZE="${SPARK_SQL_ADVISORY_PARTITION_SIZE:-64m}"
SPARK_SQL_COALESCE_MIN_PARTITION_SIZE="${SPARK_SQL_COALESCE_MIN_PARTITION_SIZE:-16m}"
SPARK_SQL_FILES_MAX_PARTITION_BYTES="${SPARK_SQL_FILES_MAX_PARTITION_BYTES:-64m}"

# Timeouts & misc perf
SPARK_SQL_BROADCAST_TIMEOUT="${SPARK_SQL_BROADCAST_TIMEOUT:-600}" # seconds
SPARK_NETWORK_TIMEOUT="${SPARK_NETWORK_TIMEOUT:-600s}"
SPARK_EXECUTOR_HEARTBEAT_INTERVAL="${SPARK_EXECUTOR_HEARTBEAT_INTERVAL:-60s}"
SPARK_SQL_AUTO_BROADCAST_JOIN_THRESHOLD="${SPARK_SQL_AUTO_BROADCAST_JOIN_THRESHOLD:-67108864}" # 64MB
SPARK_SQL_PARTITION_DISCOVERY_PARALLELISM="${SPARK_SQL_PARTITION_DISCOVERY_PARALLELISM:-100}"

# Thrift Server behavior
SPARK_THRIFT_SINGLE_SESSION="${SPARK_THRIFT_SINGLE_SESSION:-false}"
SPARK_THRIFT_INCREMENTAL_COLLECT="${SPARK_THRIFT_INCREMENTAL_COLLECT:-true}"

echo "[spark-entrypoint] configuration:"
echo "  SPARK_MASTER=${SPARK_MASTER}"
echo "  WAREHOUSE=${WAREHOUSE}"
echo "  THRIFT=${THRIFT_HOST}:${THRIFT_PORT}"
echo "  DRIVER_MEMORY=${SPARK_DRIVER_MEMORY} (+${SPARK_DRIVER_OVERHEAD} overhead)"
echo "  SHUFFLE_PARTITIONS=${SPARK_SQL_SHUFFLE_PARTITIONS}"
echo "  RUN_IMPORTER=${RUN_IMPORTER} (input=${IMPORTER_INPUT_PATH} -> ns=${IMPORTER_OUTPUT_NS})"

# ------------------------------------------------
# 1) Optional: Java metadata import into Iceberg
# ------------------------------------------------
if [[ "${RUN_IMPORTER}" = "1" ]]; then
  echo "[spark-entrypoint] launching Java metadata importer…"
  java -jar "${IMPORTER_JAR}" \
    --input-path "${IMPORTER_INPUT_PATH}" \
    --output-path "${IMPORTER_OUTPUT_NS}" \
    --warehouse-uri "${WAREHOUSE}"
else
  echo "[spark-entrypoint] skipping metadata importer (RUN_IMPORTER=${RUN_IMPORTER})"
fi

echo "[spark-entrypoint] launching Spark Thrift Server…"

# ------------------------------------------------
# 2) Build spark-submit command (array for safe quoting)
# ------------------------------------------------
cmd=(
  "${SPARK_HOME}/bin/spark-submit"
  --class org.apache.spark.sql.hive.thriftserver.HiveThriftServer2
  --jars "${SPARK_HOME}/jars/iceberg-spark-runtime-3.4_2.12-1.9.2.jar,${SPARK_HOME}/jars/iceberg-core-1.9.2.jar"
  --master "${SPARK_MASTER}"
  --driver-memory "${SPARK_DRIVER_MEMORY}"
  --conf "spark.driver.memoryOverhead=${SPARK_DRIVER_OVERHEAD}"
  --conf "spark.driver.extraJavaOptions=-XX:+UseG1GC -XX:+ExitOnOutOfMemoryError"

  # ---- Spark UI ----
  --conf "spark.ui.enabled=${SPARK_UI_ENABLED}"
  --conf "spark.ui.port=${SPARK_UI_PORT}"

  # ---- Iceberg wiring ----
  --conf "spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions"

  # Make Iceberg the default catalog so `USE nyc` hits Iceberg
  --conf "spark.sql.defaultCatalog=${SPARK_DEFAULT_CATALOG}"
  --conf "spark.sql.catalog.${SPARK_DEFAULT_CATALOG}=org.apache.iceberg.spark.SparkCatalog"
  --conf "spark.sql.catalog.${SPARK_DEFAULT_CATALOG}.type=hadoop"
  --conf "spark.sql.catalog.${SPARK_DEFAULT_CATALOG}.warehouse=${WAREHOUSE}"
  --conf "spark.sql.catalog.${SPARK_DEFAULT_CATALOG}.io-impl=org.apache.iceberg.hadoop.HadoopFileIO"

  # Ensure built-in spark_catalog is Iceberg-backed too
  --conf "spark.sql.catalog.spark_catalog=org.apache.iceberg.spark.SparkSessionCatalog"
  --conf "spark.sql.catalog.spark_catalog.type=hadoop"

  # Keep Spark's warehouse dir aligned
  --conf "spark.sql.warehouse.dir=${WAREHOUSE}"

  # ---- HiveServer2 (via Hadoop conf pass-through) ----
  # These MUST be set as spark.hadoop.* to reach Hive's configuration.
  --conf "spark.hadoop.hive.server2.thrift.bind.host=${THRIFT_HOST}"
  --conf "spark.hadoop.hive.server2.thrift.port=${THRIFT_PORT}"
  --conf "spark.hadoop.hive.server2.transport.mode=binary"
  --conf "spark.hadoop.hive.server2.authentication=NOSASL"

  # ---- Thrift Server behavior & streaming results to client ----
  --conf "spark.sql.hive.thriftServer.singleSession=${SPARK_THRIFT_SINGLE_SESSION}"
  --conf "spark.sql.hive.thriftServer.incrementalCollect=${SPARK_THRIFT_INCREMENTAL_COLLECT}"

  # ---- Memory-pressure, shuffle & scan tuning ----
  --conf "spark.sql.execution.useObjectHashAggregate=false"

  # Adaptive Query Execution + sensible partition sizing/coalesce
  --conf "spark.sql.adaptive.enabled=true"
  --conf "spark.sql.shuffle.partitions=${SPARK_SQL_SHUFFLE_PARTITIONS}"
  --conf "spark.sql.adaptive.advisoryPartitionSizeInBytes=${SPARK_SQL_ADVISORY_PARTITION_SIZE}"
  --conf "spark.sql.adaptive.coalescePartitions.enabled=true"
  --conf "spark.sql.adaptive.coalescePartitions.minPartitionSize=${SPARK_SQL_COALESCE_MIN_PARTITION_SIZE}"
  --conf "spark.sql.adaptive.skewJoin.enabled=true"

  # File scan sizing & vectorization
  --conf "spark.sql.files.maxPartitionBytes=${SPARK_SQL_FILES_MAX_PARTITION_BYTES}"
  --conf "spark.sql.sources.parallelPartitionDiscovery.parallelism=${SPARK_SQL_PARTITION_DISCOVERY_PARALLELISM}"
  --conf "spark.sql.parquet.enableVectorizedReader=true"
  --conf "spark.sql.parquet.filterPushdown=true"

  # Broadcasts & timeouts
  --conf "spark.sql.autoBroadcastJoinThreshold=${SPARK_SQL_AUTO_BROADCAST_JOIN_THRESHOLD}"
  --conf "spark.sql.broadcastTimeout=${SPARK_SQL_BROADCAST_TIMEOUT}"
  --conf "spark.network.timeout=${SPARK_NETWORK_TIMEOUT}"
  --conf "spark.executor.heartbeatInterval=${SPARK_EXECUTOR_HEARTBEAT_INTERVAL}"

  # Optional: fair scheduling to keep multi-client queries responsive
  --conf "spark.scheduler.mode=FAIR"

  # Keep any defaults you may have
  --properties-file "${SPARK_HOME}/conf/spark-defaults.conf"
)

exec "${cmd[@]}"
