# Apache-Metadata-Pipeline
Walks a dataset directory holding parquet files, generates Apache Hadoop Catalog and corresponding tables of metadata. Deploys Apache Superset server, Postgres server (holds Superset metadata), and Spark Thrift Query Engine in individual containers.

# Prerequisites:
    - Create and populate data directory
        - Inside data directory will be folders (representing different tables) holding your raw parquet files that will be aggregated to their respective tables
        - Example NYC Taxi Parquet Files: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page 

    Example Directory:
    data/
    ├── for_hire/
    │   ├── fhv_tripdata_2025-01.parquet
    │   ├── fhv_tripdata_2025-02.parquet
    │   └── fhv_tripdata_2025-03.parquet
    ├── green_taxi/
    └── yellow_taxi/

# If your parquet files contain incompatible data types with iceberg vectorization reads (like UINT64), must cast data to compatible type:
    # Activate virtual environment:
    /opt/homebrew/bin/python3.12 -m venv .venv
    source .venv/bin/activate

    # Install PyArrow depedency:
    pip install --upgrade pip
    pip install pyarrow==16.1.0

    # Run sanitize script:

    python sanitize_parquet.py \
      --in ./data/<Insert_Parquet_File_Name> \
      --out ./<Insert_Output_Directory>/Insert_Parquet_File_Name \

    Example:
    python sanitize_parquet.py \
      --in ./data/System_Interface_Counters \
      --out ./data_casted/System_Interface_Counters \


    (If this step is done, must replace the unsanitized data with the generated sanitized data)


# Run Project:
    docker-compose up -d --build

    This runs various servers in individual contaienrs:
    - Postgres server (stores Superset's metadata)
    - Superset server (your dashboard)
    - Spark Thrift server (your query engine)
    
# Once servers are up and running: 
    - Access Superset Dashboard through localhost:8088
    - If prompted to log in: 
        - Username: admin 
        - Password: admin

    - Get started by making sure your data directory is successfully connected and create a dataset
        - If dataset does not automatically connect, select Apache Spark SQL as database and use URI: hive://spark-thrift-server:10000/default?auth=NOSASL
        - Test Connection button should display 'Connection looks good!' on the bottom right upon successful connection

    - Create an empty dashboard and start exploring charts
    - Create your first chart by navigating to charts tab, click add chart 
    - Choose your dataset and select the type of chart you want to visualize
    - Create the new chart and drag/drop column features to designated query options
    - Add chart to your dashboard, you can now navigate to your dashboards and view your charts!

# Anomaly Detection / Classification:
    - Start virtual environment: python3.11 -m venv .venv
    - Activate virtual environment: source .venv/bin/activate
    - Install requirements: pip install -r requirements.txt

    - Run Anomaly Detection script:
        python anomalyDetection.py -i ./data/<table_name>/<parquet_files> -f <feature_to_run_script_on>

    # Example: analyze all parquet files in a directory on column "temperature"
        python anomalyDetection.py -i ./data/stats_table -f temperature

    # Example: glob works too (e.g., .prq)
        python anomalyDetection.py -i "./data/stats_test/*.prq" -f temperature

    # Example: override timestamp/system column names and output dir (optional)
        python anomalyDetection.py -i ./data/any -f cpu_temp --timestamp-col ts --system-col device --out preds_cpu