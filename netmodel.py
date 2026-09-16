# netmodel.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Import the custom classes
from custom_transformers import (
    NetworkDomainRuleTransformer,
    CustomLogisticRegression
)

from config import NETWORK_DATASET_PATH, NETWORK_MODEL_PATH

spark = SparkSession.builder.appName("NetworkCustomLogisticRegTraining").getOrCreate()

network_schema = StructType([
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
    StructField("label", StringType(), True)  # "0" or "1"
])

df = spark.read.csv(
    NETWORK_DATASET_PATH,
    header=True,
    schema=network_schema
).dropna()

# Convert label to double
df = df.withColumn("indexedLabel", col("label").cast("double"))

domain_transformer = NetworkDomainRuleTransformer(bigPacketThreshold=1000)

feature_cols = [
    "length","Source Port","Destination Port","syn",
    "Sequence number","Window size value",
    "bigPacketFlag","synFlag","suspiciousPortFlag","protocolFlag"
]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="rawFeatures")
scaler = StandardScaler(inputCol="rawFeatures", outputCol="features", withStd=True, withMean=False)

myLR = CustomLogisticRegression(labelCol="indexedLabel", featuresCol="features", maxIter=30, lr=0.01)

pipeline = Pipeline(stages=[domain_transformer, assembler, scaler, myLR])

train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)
model = pipeline.fit(train_df)

predictions = model.transform(test_df)
eval = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = eval.evaluate(predictions)
print(f"[NetworkModel] Accuracy: {accuracy:.3f}")

model.write().overwrite().save(NETWORK_MODEL_PATH)
print("[NetworkModel] Pipeline saved successfully!")
spark.stop()
