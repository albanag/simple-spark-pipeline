from pyspark.ml.pipeline import PipelineModel
from pyspark.sql import SparkSession

from train_model import prepare, score
from data_access import read_dataframe


if __name__ == '__main__':
    spark = SparkSession.builder.master("local").appName("score_model").getOrCreate()
    # Read
    df = read_dataframe(spark)

    # Prepare
    clean_df = prepare(df)

    # Load model
    model = PipelineModel.load("models/model.m")

    # Score model
    predictions = score(model, clean_df)

    predictions.show()


