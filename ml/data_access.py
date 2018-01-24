

def read_dataframe(spark):
    df = spark.read \
        .format('csv') \
        .option('sep', '\t') \
        .option('header', 'false') \
        .load('data/diagnosis.data') \
        .toDF('temp', 'nausea', 'lumbar', 'urine', 'micturition', 'urethra', 'bladder', 'renal')
    return df
