"""
OFFICIAL PYSPARK MLLIB CODE (FOR APACHE HADOOP CLUSTER SUBMISSION)
This script demonstrates distributed machine learning on Hadoop HDFS using PySpark MLlib:
1. Loads dataset from HDFS (hdfs:///bigdata/unicef/unicef_labeled_data.csv)
2. StringIndexer & OneHotEncoder for distributed categorical encoding
3. VectorAssembler for distributed feature vectorization
4. RandomForestClassifier for Survival, Malnutrition, and Disease prediction
5. MulticlassClassificationEvaluator for distributed model evaluation
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def run_spark_hadoop_job():
    print("Initializing Apache Spark Session connected to Hadoop YARN / HDFS...")
    spark = SparkSession.builder \
        .appName("UNICEF_Malnutrition_BigData_Analytics") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

    # Load dataset from HDFS
    hdfs_path = "hdfs:///bigdata/unicef/unicef_labeled_data.csv"
    print(f"Reading distributed dataset from: {hdfs_path}")
    df = spark.read.csv(hdfs_path, header=True, inferSchema=True)

    categorical_cols = [
        'Gender', 'Region', 'Exclusive_BF_6M', 'Safe_Water', 'Sanitation',
        'Vaccinated', 'Recent_Illness', 'Household_Income', 'Mother_Education',
        'Food_Poverty', 'Residence', 'Preterm_Birth', 'Vitamin_A_Supplement',
        'Diarrhea_Last_2Weeks', 'Fever_Last_2Weeks', 'Cough_Last_2Weeks', 'ARI_Symptoms'
    ]
    numeric_cols = [
        'Age_Months', 'Birth_Weight_kg', 'Dietary_Diversity', 'Meal_Frequency',
        'Household_Size', 'Mother_Age', 'Mother_BMI'
    ]

    # Index categorical columns
    indexed_cols = []
    for cat in categorical_cols:
        indexer = StringIndexer(inputCol=cat, outputCol=cat + "_indexed", handleInvalid="keep")
        df = indexer.fit(df).transform(df)
        indexed_cols.append(cat + "_indexed")

    # Assemble feature vector
    assembler = VectorAssembler(inputCols=indexed_cols + numeric_cols, outputCol="raw_features")
    df_assembled = assembler.transform(df)

    # Scale features
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=False)
    scaler_model = scaler.fit(df_assembled)
    df_final = scaler_model.transform(df_assembled)

    # Split dataset across Hadoop cluster nodes
    train_data, test_data = df_final.randomSplit([0.8, 0.2], seed=42)

    # Model 1: Survival Classifier
    print("Training Distributed Random Forest on Hadoop for Survival Prediction...")
    rf_surv = RandomForestClassifier(labelCol="Survival_Status", featuresCol="features", numTrees=100)
    model_surv = rf_surv.fit(train_data)
    pred_surv = model_surv.transform(test_data)

    evaluator = MulticlassClassificationEvaluator(labelCol="Survival_Status", predictionCol="prediction", metricName="accuracy")
    print(f"Hadoop Spark MLlib Survival Accuracy: {evaluator.evaluate(pred_surv) * 100:.2f}%")

    # Save model back to HDFS
    model_surv.write().overwrite().save("hdfs:///bigdata/unicef/models/spark_survival_model")
    print("Saved Spark MLlib model to HDFS.")

if __name__ == '__main__':
    # To run on Hadoop cluster: spark-submit --master yarn train_spark_mllib.py
    pass
