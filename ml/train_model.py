
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier

from pyspark.sql import SparkSession
from data_access import read_dataframe
from data_processing import TemperatureDiscretizer

import os


def prepare(df):
    temp_discretizer = TemperatureDiscretizer(inputCol="temp", outputCol="dtemp")
    return temp_discretizer.transform(df)


def define_pipeline():
    temp_indexer = StringIndexer(inputCol="dtemp", outputCol="idtemp")
    nausea_indexer = StringIndexer(inputCol="nausea", outputCol="inausea")
    lumbar_indexer = StringIndexer(inputCol="lumbar", outputCol="ilumbar")
    urine_indexer = StringIndexer(inputCol="urine", outputCol="iurine")
    micturition_indexer = StringIndexer(inputCol="micturition", outputCol="imicturition")
    urethra_indexer = StringIndexer(inputCol="urethra", outputCol="iurethra")
    bladder_indexer = StringIndexer(inputCol="bladder", outputCol="label")
    vector_assembler = VectorAssembler(inputCols=["idtemp", "inausea", "ilumbar", "iurine", "imicturition", "iurethra"],
                                       outputCol="features")

    label_indexer = StringIndexer(inputCol="bladder", outputCol="label_bladder")

    rf = RandomForestClassifier(predictionCol="rf_prediction", probabilityCol="rf_probability",
                                rawPredictionCol="rf_rawPrediction", numTrees=10)
    dt = DecisionTreeClassifier(predictionCol="dt_prediction",
                                probabilityCol="dt_probability", rawPredictionCol="dt_rawPrediction")

    res_pipeline = Pipeline(
        stages=[temp_indexer, nausea_indexer, lumbar_indexer, urine_indexer, micturition_indexer,
                urethra_indexer, bladder_indexer, vector_assembler, label_indexer, rf, dt])

    return res_pipeline


def train(train_pipeline, data):
    trained_model = train_pipeline.fit(data)
    return trained_model


def score(trained_model, df):
    predictions = trained_model.transform(df)
    return predictions


def evaluate(trained_model, data, prediction_col_name):
    predictions = trained_model.transform(data)
    predictions.show()
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol=prediction_col_name,
                                                  metricName="accuracy")
    model_accuracy = evaluator.evaluate(predictions)
    return model_accuracy


def store_model(name):
    if not os.path.exists('models'):
        os.mkdir('models')
    model.write().overwrite().save('models/'+name)


if __name__ == '__main__':

    spark = SparkSession.builder.master("local").appName("train_model").getOrCreate()

    # Read data
    diagnosis_df = read_dataframe(spark)
    diagnosis_df.show()

    # Prepare data
    clean_df = prepare(diagnosis_df)

    # Split data
    (train_data, test_data) = clean_df.randomSplit([0.8, 0.2])
    train_data.cache()
    test_data.cache()

    # Define the pipeline
    pipeline = define_pipeline()

    # Train model
    model = train(pipeline, train_data)

    # Evaluate model
    rf_accuracy = evaluate(model, test_data, "rf_prediction")
    dt_accuracy = evaluate(model, test_data, "dt_prediction")
    print("Random Forest accuracy is", rf_accuracy, "and Decision Tree accuracy is", dt_accuracy)

    # Store model
    store_model('model.m')
