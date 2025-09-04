package com.example;

import org.apache.hadoop.conf.Configuration;
import org.apache.iceberg.Schema;
import org.apache.iceberg.PartitionSpec;
import org.apache.iceberg.Table;
import org.apache.iceberg.catalog.TableIdentifier;
import org.apache.iceberg.hadoop.HadoopCatalog;
import org.apache.iceberg.types.Types;

public class App {

  public static void main(String[] args) {

    // 1️⃣  Tell Iceberg to write inside the /warehouse mount
    String warehousePath = "file:///warehouse";

    Configuration conf = new Configuration();
    HadoopCatalog catalog = new HadoopCatalog(conf, warehousePath);
    TableIdentifier tableId = TableIdentifier.of("namespace", "table_name");

    // trivial demo schema
    Schema schema = new Schema(
        Types.NestedField.required(1, "id",   Types.IntegerType.get()),
        Types.NestedField.optional(2, "data", Types.StringType.get())
    );
    PartitionSpec spec = PartitionSpec.unpartitioned();

    // 2️⃣  Re-create fresh metadata every run (dev-only convenience)
    if (catalog.tableExists(tableId)) {
      catalog.dropTable(tableId, /* purge = */ true);   // delete old metadata
    }
    Table table = catalog.createTable(tableId, schema, spec);

    System.out.println("Created Iceberg table at: " + table.location());
  }
}
