# Model Training and Scoring using Spark Pipelines


The same codebase can be used to create two applications: one for training the model and one for making predictions through scoring.

To ease the installation and preparation of a running environment, the applications run in Docker containers, hence the two Dockerfiles, one for each application. 

The Docker images are based on a parent image containing Spark 2.2.0 and other Python libraries ([docker-spark](https://hub.docker.com/r/albanag/docker-spark/)).

In addition to the two Docker containers, a Docker volume is created to store the model by the training application and retrieve it later by the scoring app.

## Training application

### Build Docker image

```
./build_train.sh
```

### Create Docker volume: 

```
./create_volume.sh
```

### Run training app:

```
./run_train.sh
```


The application runs in foreground and, if successful, it stores a model in the Docker volume created earlier.

## Scoring application

For simplicity, the scoring application uses the same data used in the training stage. The application loads the data and the model and makes predictions.

### Build Docker image:

```
./build_score.sh
```

### Run the scoring app:

```
./run_score.sh
```
----
# Final notes
- The dataset used: [acute inflammations dataset](https://archive.ics.uci.edu/ml/datasets/Acute+Inflammations).
- For a Scala implementation of the Spark Pipelines, refer to [categorical-bayes](https://github.com/bbiletskyy/categorical-bayes).
