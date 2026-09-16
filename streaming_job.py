from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.ml import PipelineModel
from pyspark.sql.functions import concat_ws

# Load the pre-trained network anomaly detection model
MODEL_PATH = "hdfs://192.168.56.101:9000/home/dan/project/oop_custom_network_model"
INPUT_PATH = "file:///home/dan/project2/network_input"
OUTPUT_PATH = "file:///home/dan/project2/logs/network_stream_logs"
CHECKPOINT_PATH = "file:///home/dan/project2/checkpoint"

# Start Spark session
spark = SparkSession.builder.appName("NetworkAnomalyDetectionFixed").getOrCreate()

# Define the full schema used during model training
schema = StructType([
    StructField("id", StringType(), True),
    StructField("Time", DoubleType(), True),
    StructField("Source", StringType(), True),
    StructField("Destination", StringType(), True),
    StructField("Protocol", StringType(), True),
    StructField("length", DoubleType(), True),
    StructField("Source Port", DoubleType(), True),
    StructField("Destination Port", DoubleType(), True),
    StructField("syn", DoubleType(), True),
    StructField("Sequence number", DoubleType(), True),
    StructField("Window size value", DoubleType(), True),
    StructField("label", StringType(), True),
])

# Read stream
stream_df = spark.readStream \
    .option("header", True) \
    .schema(schema) \
    .csv(INPUT_PATH)

# Fill nulls to prevent assembler errors
stream_df = stream_df.fillna(0)

# Generate domain-specific flags BEFORE model
stream_df = stream_df \
    .withColumn("bigPacketFlag", when(col("length") > 1000, 1).otherwise(0)) \
    .withColumn("suspiciousPortFlag", when((col("Destination Port") < 20) | (col("Destination Port") > 1024), 1).otherwise(0)) \
    .withColumn("synFlag", when(col("syn") == 1, 1).otherwise(0)) \
    .withColumn("protocolFlag", when(~col("Protocol").isin("TCP", "UDP"), 1).otherwise(0))

# Load model
# Load model
model = PipelineModel.load(MODEL_PATH)

# Transform with model
predicted_df = model.transform(stream_df)

# Add predicted label AFTER transform
predicted_df = predicted_df.withColumn(
    "predicted_label",
    when(col("probability_custom") >= 0.3, "Anomaly").otherwise("Normal")
)



# Add a reason column explaining why flagged
predicted_df = predicted_df.withColumn(
    "reason",
    concat_ws(",", 
        when(col("bigPacketFlag") == 1, "large_packet").otherwise(None),
        when(col("suspiciousPortFlag") == 1, "invalid_port").otherwise(None),
        when(col("synFlag") == 1, "SYN_flag").otherwise(None),
        when(col("protocolFlag") == 1, "suspicious_protocol").otherwise(None)
    )
)


# Write to local logs
query = predicted_df.select(
    "id", "Time", "Source", "Destination", "Protocol", "length",
    "Source Port", "Destination Port", "syn", "Sequence number",
    "Window size value", "label", "predicted_label", "reason"
).writeStream \
    .outputMode("append") \
    .format("csv") \
    .option("header", True) \
    .option("checkpointLocation", CHECKPOINT_PATH) \
    .option("path", OUTPUT_PATH) \
    .start()


query.awaitTermination()
