import os
import argparse
from pathlib import Path
from pyspark.sql import functions as F, Window
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, BucketedRandomProjectionLSH
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import DoubleType

# -------- CLI --------
parser = argparse.ArgumentParser(description="Anomaly detection over Parquet directory using z-score, IQR, KMeans, and KNN.")
parser.add_argument("-i", "--input", required=False,
                    default=os.getenv("INPUT_PATH", "./data/stats_test/*.prq"),
                    help="Input Parquet path: directory or glob (e.g., ./data/dir or ./data/*.parquet).")
parser.add_argument("-f", "--feature", required=False,
                    default=os.getenv("FEATURE_COL", "temperature"),
                    help="Numeric column/feature to analyze (default: temperature).")
parser.add_argument("--timestamp-col", required=False,
                    default=os.getenv("TIMESTAMP_COL", "timestamp"),
                    help="Timestamp column name (default: timestamp).")
parser.add_argument("--system-col", required=False,
                    default=os.getenv("SYSTEM_COL", "system_id"),
                    help="System/group column name (default: system_id).")
parser.add_argument("-o", "--out", required=False,
                    default=os.getenv("OUTPUT_DIR", "preds"),
                    help="Output directory for results (default: preds).")
args = parser.parse_args()

INPUT_PATH   = args.input
FEATURE_COL  = args.feature
TS_COL       = args.timestamp_col
SYS_COL      = args.system_col
OUT_DIR      = args.out

# -------- Tunables (env overrides) --------
Z_THRESH        = float(os.getenv("Z_THRESH", "2"))
MIN_POINTS      = int(os.getenv("MIN_POINTS", "20"))
TOPK            = int(os.getenv("TOPK", "50"))

KMEANS_K        = int(os.getenv("KMEANS_K", "2"))
KM_PR_CUTOFF    = float(os.getenv("KM_PR_CUTOFF", "0.99"))  # top 1% farthest from centroid

KNN_MODE        = os.getenv("KNN_MODE", "time")              # "time" (default) or "lsh"
KNN_K           = int(os.getenv("KNN_K", "5"))
KNN_PR_CUTOFF   = float(os.getenv("KNN_PR_CUTOFF", "0.99"))  # top 1% by knn_dist
# LSH tuning (only used if KNN_MODE=lsh)
LSH_BUCKET      = float(os.getenv("LSH_BUCKET", "1.0"))
LSH_THRESH      = float(os.getenv("LSH_THRESH", "1.5"))
MAX_GROUP_ROWS  = int(os.getenv("MAX_GROUP_ROWS", "250000")) # cap group size for LSH path

# Console print cap
PRINT_LIMIT     = int(os.getenv("PRINT_LIMIT", "100"))

spark = (
    SparkSession.builder
    .appName("ClassifyTemps")
    .master("local[*]")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.driver.host", "127.0.0.1")
    # --- memory & shuffle tuning ---
    .config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY", "6g"))
    .config("spark.sql.shuffle.partitions", os.getenv("SHUFFLE_PARTITIONS", "64"))
    .getOrCreate()
)

# ---- INPUT ----
df = spark.read.parquet(INPUT_PATH)  # directory or glob is fine

# quick schema sanity
missing = [c for c in [SYS_COL, TS_COL, FEATURE_COL] if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in input data: {missing}. "
                     f"Available columns: {df.columns}")

# ensure timestamp is a real timestamp (if it's string in your data)
if dict(df.dtypes).get(TS_COL) != "timestamp":
    df = df.withColumn(TS_COL, F.to_timestamp(F.col(TS_COL)))

# ensure feature is numeric
df = df.withColumn(FEATURE_COL, F.col(FEATURE_COL).cast(DoubleType()))

# ---- Z-SCORE OUTLIERS PER SYSTEM ----
w_sys = Window.partitionBy(SYS_COL)
scored = (
    df
    .withColumn("n", F.count(F.lit(1)).over(w_sys))
    .withColumn("mean_feat", F.mean(F.col(FEATURE_COL)).over(w_sys))
    .withColumn("std_feat",  F.stddev_samp(F.col(FEATURE_COL)).over(w_sys))
    .withColumn(
        "z",
        F.when(F.col("std_feat").isNull() | (F.col("std_feat") == 0), F.lit(0.0))
         .otherwise((F.col(FEATURE_COL) - F.col("mean_feat")) / F.col("std_feat"))
    )
    .withColumn("abs_z", F.abs("z"))
).cache()  # cached since we reuse it several times

outliers_z = (
    scored
    .filter((F.col("n") >= MIN_POINTS) & (F.col("abs_z") > F.lit(Z_THRESH)))
    .select(SYS_COL, TS_COL, FEATURE_COL, "z", "abs_z")
)

# Always surface the strongest candidates
suspects_topk = (
    scored
    .orderBy(F.desc("abs_z"))
    .select(SYS_COL, TS_COL, FEATURE_COL, "z", "abs_z")
    .limit(TOPK)
)

# ---- ROBUST IQR OUTLIERS (per system) ----
# q1, q3 are per-system; we compute thresholds and expose detailed outliers
iqr_stats = df.groupBy(SYS_COL).agg(
    F.expr(f"percentile_approx({FEATURE_COL}, 0.25)").alias("q1"),
    F.expr(f"percentile_approx({FEATURE_COL}, 0.75)").alias("q3"),
)
with_iqr = (
    df.join(iqr_stats, SYS_COL, "left")
      .withColumn("iqr", F.col("q3") - F.col("q1"))
      .withColumn("low",  F.col("q1") - 1.5 * F.col("iqr"))
      .withColumn("high", F.col("q3") + 1.5 * F.col("iqr"))
)

# Per-system IQR thresholds table (unique per system)
iqr_bounds = (
    with_iqr.groupBy(SYS_COL)
    .agg(
        F.first("q1").alias("q1"),
        F.first("q3").alias("q3"),
        F.first("iqr").alias("iqr"),
        F.first("low").alias("low"),
        F.first("high").alias("high"),
        F.count("*").alias("n"),
        F.sum(
            F.when((F.col(FEATURE_COL) < F.col("low")) | (F.col(FEATURE_COL) > F.col("high")), 1).otherwise(0)
        ).alias("n_outliers")
    )
)

# Detailed IQR outliers table (every row that violates low/high)
outliers_iqr_detailed = (
    with_iqr
    .withColumn(
        "where",
        F.when(F.col(FEATURE_COL) < F.col("low"), F.lit("below_low")).otherwise(F.lit("above_high"))
    )
    .withColumn(
        "dist_from_bound",
        F.when(F.col(FEATURE_COL) < F.col("low"), F.col("low") - F.col(FEATURE_COL))
         .otherwise(F.col(FEATURE_COL) - F.col("high"))
    )
    .filter((F.col(FEATURE_COL) < F.col("low")) | (F.col(FEATURE_COL) > F.col("high")))
    .select(SYS_COL, TS_COL, FEATURE_COL, "q1", "q3", "iqr", "low", "high", "where", "dist_from_bound")
)

# For parity, keep the simple/compact IQR output as well
outliers_iqr = outliers_iqr_detailed.select(SYS_COL, TS_COL, FEATURE_COL)

# =======================
#    K-MEANS ANOMALIES
# =======================
km_input = scored.select(SYS_COL, TS_COL, FEATURE_COL, "z")
vec_km   = VectorAssembler(inputCols=["z"], outputCol="rawFeatures")
scale_km = StandardScaler(inputCol="rawFeatures", outputCol="features", withMean=True, withStd=True)
kmeans   = KMeans(k=KMEANS_K, seed=42, featuresCol="features", predictionCol="cluster")

km_pipe  = Pipeline(stages=[vec_km, scale_km, kmeans])
km_model = km_pipe.fit(km_input)

# Distance to nearest centroid as anomaly score
centers = km_model.stages[-1].clusterCenters()
def dist_to_center(feat, cid):
    c = centers[cid]
    from math import sqrt  # noqa: F401
    return float(Vectors.squared_distance(feat, c)) ** 0.5
dist_udf = F.udf(dist_to_center, DoubleType())

km_scored = (km_model.transform(km_input)
             .withColumn("km_dist", dist_udf(F.col("features"), F.col("cluster"))))

w_km = Window.partitionBy(SYS_COL).orderBy(F.col("km_dist"))
km_with_rank = km_scored.withColumn("km_pr", F.percent_rank().over(w_km))
outliers_kmeans = (km_with_rank
                   .where(F.col("km_pr") >= F.lit(KM_PR_CUTOFF))
                   .select(SYS_COL, TS_COL, FEATURE_COL, "z", "cluster", "km_dist", "km_pr"))

# =======================
#          KNN
# =======================
if KNN_MODE.lower() == "time":
    # ---- Time-adjacent KNN (linear; very memory-friendly) ----
    w_time = Window.partitionBy(SYS_COL).orderBy(TS_COL)
    lead_dists = [F.abs(F.col("z") - F.lead("z", i).over(w_time)) for i in range(1, KNN_K+1)]
    lag_dists  = [F.abs(F.col("z") - F.lag("z",  i).over(w_time)) for i in range(1, KNN_K+1)]
    with_dists = scored.withColumn("dists", F.array(*(lead_dists + lag_dists)))
    with_knn = with_dists.withColumn(
        "knn_dist",
        F.expr(f"""
            CASE WHEN size(filter(dists, x -> x is not null)) > 0 THEN
              aggregate(
                slice(array_sort(filter(dists, x -> x is not null)), 1, {KNN_K}),
                0D,
                (acc,x) -> acc + x
              ) / LEAST({KNN_K}, size(filter(dists, x -> x is not null)))
            ELSE NULL END
        """)
    )
    w_tail = Window.partitionBy(SYS_COL).orderBy(F.col("knn_dist"))
    outliers_knn = (with_knn
        .withColumn("knn_pr", F.percent_rank().over(w_tail))
        .where(F.col("knn_pr") >= F.lit(KNN_PR_CUTOFF))
        .select(SYS_COL, TS_COL, FEATURE_COL, "z", "knn_dist", "knn_pr"))

else:
    # ---- LSH KNN (heavy). Restricted to avoid OOM; fallback to time-adjacent for big groups. ----
    from pyspark.sql import Window as W2
    df_id = scored.select(SYS_COL, TS_COL, FEATURE_COL, "z") \
                  .withColumn("id", F.monotonically_increasing_id())
    counts = df_id.groupBy(SYS_COL).count()
    small_ids = counts.filter(F.col("count") <= MAX_GROUP_ROWS).select(SYS_COL)
    big_ids   = counts.filter(F.col("count") >  MAX_GROUP_ROWS).select(SYS_COL)

    small_df = df_id.join(small_ids, SYS_COL)
    big_df   = df_id.join(big_ids, SYS_COL)  # will use time-adjacent KNN

    # LSH on small groups
    vec_knn = VectorAssembler(inputCols=["z"], outputCol="features")
    vdf = vec_knn.transform(small_df)
    lsh = BucketedRandomProjectionLSH(
        inputCol="features", outputCol="hashes",
        bucketLength=LSH_BUCKET, numHashTables=2
    )
    lshm = lsh.fit(vdf)

    joined = (lshm.approxSimilarityJoin(vdf, vdf, LSH_THRESH, distCol="dist")
      .where(F.col("datasetA.id") != F.col("datasetB.id"))
      .where(F.col("datasetA."+SYS_COL) == F.col("datasetB."+SYS_COL))
      .select(
          F.col("datasetA.id").alias("idA"),
          F.col("datasetA."+SYS_COL).alias(SYS_COL),
          F.col("datasetA."+TS_COL).alias(TS_COL),
          F.col("datasetA."+FEATURE_COL).alias(FEATURE_COL),
          F.col("datasetA.z").alias("z"),
          F.col("datasetB.id").alias("idB"),
          "dist"))

    w_knn = W2.partitionBy("idA").orderBy(F.col("dist"))
    knn = joined.withColumn("rn", F.row_number().over(w_knn)).where(F.col("rn") <= KNN_K)
    knn_scores = knn.groupBy("idA").agg(F.avg("dist").alias("knn_dist"))

    small_ranked = (small_df.alias("d")
        .join(knn_scores, F.col("d.id") == F.col("idA"), "left")
        .drop("idA").fillna({"knn_dist": 0.0}))

    w_tail = Window.partitionBy(SYS_COL).orderBy(F.col("knn_dist"))
    small_out = (small_ranked
        .withColumn("knn_pr", F.percent_rank().over(w_tail))
        .where(F.col("knn_pr") >= F.lit(KNN_PR_CUTOFF))
        .select(SYS_COL, TS_COL, FEATURE_COL, "z", "knn_dist", "knn_pr"))

    # Fallback time-adjacent KNN for big groups
    w_time = Window.partitionBy(SYS_COL).orderBy(TS_COL)
    lead_dists = [F.abs(F.col("z") - F.lead("z", i).over(w_time)) for i in range(1, KNN_K+1)]
    lag_dists  = [F.abs(F.col("z") - F.lag("z",  i).over(w_time)) for i in range(1, KNN_K+1)]
    big_with_d = big_df.withColumn("dists", F.array(*(lead_dists + lag_dists)))
    big_knn = big_with_d.withColumn(
        "knn_dist",
        F.expr(f"""
            CASE WHEN size(filter(dists, x -> x is not null)) > 0 THEN
              aggregate(
                slice(array_sort(filter(dists, x -> x is not null)), 1, {KNN_K}),
                0D,
                (acc,x) -> acc + x
              ) / LEAST({KNN_K}, size(filter(dists, x -> x is not null)))
            ELSE NULL END
        """))
    w_tail2 = Window.partitionBy(SYS_COL).orderBy(F.col("knn_dist"))
    big_out = (big_knn
        .withColumn("knn_pr", F.percent_rank().over(w_tail2))
        .where(F.col("knn_pr") >= F.lit(KNN_PR_CUTOFF))
        .select(SYS_COL, TS_COL, FEATURE_COL, "z", "knn_dist", "knn_pr"))

    outliers_knn = small_out.unionByName(big_out)

# ---- OUTPUT ----
base = (Path.cwd() / OUT_DIR).resolve()
for sub in ["temp_outliers_zscore", "temp_suspects_topk", "temp_outliers_iqr",
            "temp_outliers_kmeans", "temp_outliers_knn",
            "iqr_bounds", "iqr_outliers_detailed"]:
    (base / sub).mkdir(parents=True, exist_ok=True)

# Keep writers modest to avoid memory spikes; adjust to taste
outliers_z.coalesce(1).write.mode("overwrite").parquet((base / "temp_outliers_zscore").as_posix())
suspects_topk.coalesce(1).write.mode("overwrite").parquet((base / "temp_suspects_topk").as_posix())
outliers_iqr.coalesce(1).write.mode("overwrite").parquet((base / "temp_outliers_iqr").as_posix())
outliers_kmeans.coalesce(1).write.mode("overwrite").parquet((base / "temp_outliers_kmeans").as_posix())
outliers_knn.coalesce(1).write.mode("overwrite").parquet((base / "temp_outliers_knn").as_posix())

# NEW: write the IQR tables
iqr_bounds.coalesce(1).write.mode("overwrite").parquet((base / "iqr_bounds").as_posix())
outliers_iqr_detailed.coalesce(1).write.mode("overwrite").parquet((base / "iqr_outliers_detailed").as_posix())

# ---- CONSOLE QUICKLOOK ----
print("rows:", df.count(), "systems:", df.select(SYS_COL).distinct().count())
print(f"feature: {FEATURE_COL}, z-threshold: {Z_THRESH}, min points per system: {MIN_POINTS}")

print("\n--- KMeans (top by distance) ---")
km_scored.orderBy(F.desc("km_dist")).select(SYS_COL, TS_COL, FEATURE_COL,
                                            "z","cluster","km_dist").show(PRINT_LIMIT, truncate=False)

print("\n--- KNN outliers (top by knn_dist) ---")
outliers_knn.orderBy(F.desc("knn_dist")).show(PRINT_LIMIT, truncate=False)

print("\n=== IQR thresholds per system ===")
iqr_bounds.orderBy(SYS_COL).show(PRINT_LIMIT, truncate=False)

print("\n=== IQR outliers (with thresholds) ===")
outliers_iqr_detailed.orderBy(F.desc("dist_from_bound")).show(PRINT_LIMIT, truncate=False)

print("\n--- Top z-score outliers ---")
outliers_z.orderBy(F.desc("abs_z")).show(PRINT_LIMIT, truncate=False)

spark.stop()
