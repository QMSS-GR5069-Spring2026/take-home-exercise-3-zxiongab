# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC <div><img src="https://www.databricks.com/sites/default/files/styles/max_1000x1000/public/2020/04/databricks-adds-access-control-to-mlflow-model-registry_01.jpg" style="height: 400px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md
# MAGIC Import a dataset of Airbnb listings and featurize the data.  We'll use this to train a model.

# COMMAND ----------

import pandas as pd

# COMMAND ----------

df_airbnb = spark.read.csv("/Volumes/gr5069/raw/airbnb/airbnb-cleaned-mlflow.csv", header=True)

# COMMAND ----------

display(df_airbnb)

# COMMAND ----------

# MAGIC %md
# MAGIC Perform a train/test split.

# COMMAND ----------

from sklearn.model_selection import train_test_split

df = df_airbnb.toPandas()
X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df[["price"]].values.ravel(), random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC Log a basic experiment by doing the following:<br><br>
# MAGIC
# MAGIC 1. Start an experiment using `mlflow.start_run()` and passing it a name for the run
# MAGIC 2. Train your model
# MAGIC 3. Log the model using `mlflow.sklearn.log_model()`
# MAGIC 4. Log the model error using `mlflow.log_metric()`
# MAGIC 5. Print out the run id using `run.info.run_uuid`

# COMMAND ----------

!pip install --upgrade typing_extensions mlflow


# COMMAND ----------

!pip install --upgrade typing_extensions mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

with mlflow.start_run(run_name="Basic RF Experiment") as run:
  # Create model, train it, and create predictions
  rf = RandomForestRegressor()
  rf.fit(X_train, y_train)
  predictions = rf.predict(X_test)
  
  # Log model
  mlflow.sklearn.log_model(rf, "random-forest-model")
  
  # Create metrics
  mse = mean_squared_error(y_test, predictions)
  print("  mse: {}".format(mse))
  
  # Log metrics
  mlflow.log_metric("mse", mse)
  
  runID = run.info.run_id
  experimentID = run.info.experiment_id
  
  print("Inside MLflow Run with run_id {} and experiment_id {}".format(runID, experimentID))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Parameters, Metrics, and Artifacts
# MAGIC
# MAGIC But wait, there's more!  In the last example, you logged the run name, an evaluation metric, and your model itself as an artifact.  Now let's log parameters, multiple metrics, and other artifacts including the feature importances.
# MAGIC
# MAGIC First, create a function to perform this.
# MAGIC
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> To log artifacts, we have to save them somewhere before MLflow can log them.  This code accomplishes that by using a temporary file that it then deletes.

# COMMAND ----------

def log_rf(experimentID, run_name, params, X_train, X_test, y_train, y_test):
  import os
  import matplotlib.pyplot as plt
  import mlflow.sklearn
  import seaborn as sns
  from sklearn.ensemble import RandomForestRegressor
  from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
  import tempfile

  with mlflow.start_run(experiment_id=experimentID, run_name=run_name) as run:
    # Create model, train it, and create predictions
    rf = RandomForestRegressor(**params)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)

    # Log model
    mlflow.sklearn.log_model(rf, "random-forest-model")

    # Log params
    [mlflow.log_param(param, value) for param, value in params.items()]

    # Create metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print("  mse: {}".format(mse))
    print("  mae: {}".format(mae))
    print("  R2: {}".format(r2))

    # Log metrics
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)  
    mlflow.log_metric("r2", r2)  
    
    # Create feature importance
    importance = pd.DataFrame(list(zip(df.columns, rf.feature_importances_)), 
                                columns=["Feature", "Importance"]
                              ).sort_values("Importance", ascending=False)
    
    # Log importances using a temporary file
    temp = tempfile.NamedTemporaryFile(prefix="feature-importance-", suffix=".csv")
    temp_name = temp.name
    try:
      importance.to_csv(temp_name, index=False)
      mlflow.log_artifact(temp_name, "feature-importance.csv")
    finally:
      temp.close() # Delete the temp file
    
    # Create plot
    fig, ax = plt.subplots()

    sns.residplot(x=predictions, y=y_test.astype(float), lowess=True)
    plt.xlabel("Predicted values for Price ($)")
    plt.ylabel("Residual")
    plt.title("Residual Plot")

    # Log residuals using a temporary file
    temp = tempfile.NamedTemporaryFile(prefix="residuals-", suffix=".png")
    temp_name = temp.name
    try:
      fig.savefig(temp_name)
      mlflow.log_artifact(temp_name, "residuals.png")
    finally:
      temp.close() # Delete the temp file
      
    display(fig)
    return run.info.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC Run with new parameters.

# COMMAND ----------

params_1 = {
  "n_estimators": 100,
  "max_depth": 5,
  "random_state": 42
}

log_rf(experimentID, "First Run", params_1, X_train, X_test, y_train, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC Check the UI to see how this appears.  Take a look at the artifact to see where the plot was saved.
# MAGIC
# MAGIC Now, run a third run.

# COMMAND ----------

params_2 = {
  "n_estimators": 1000,
  "max_depth": 10,
  "random_state": 42
}

log_rf(experimentID, "Second Run", params_2, X_train, X_test, y_train, y_test)

# COMMAND ----------

params_3 = {
  "n_estimators": 3000,
  "max_depth": 7,
  "random_state": 62
}

log_rf(experimentID, "Third Run", params_3, X_train, X_test, y_train, y_test)

# COMMAND ----------

params_4 = {
  "n_estimators": 1000,
  "max_depth": 15,
  "random_state": 62
}

log_rf(experimentID, "Forth Run", params_4, X_train, X_test, y_train, y_test)

# COMMAND ----------

params_5 = {
  "n_estimators": 1500,
  "max_depth": 15,
  "random_state": 62
}

log_rf(experimentID, "Fifth Run", params_5, X_train, X_test, y_train, y_test)

# COMMAND ----------

params_6 = {
  "n_estimators": 1300,
  "max_depth": 21,
  "random_state": 62
}

log_rf(experimentID, "Sixth Run", params_6, X_train, X_test, y_train, y_test)

# COMMAND ----------

params_7 = {
  "n_estimators": 1200,
  "max_depth": 18,
  "random_state": 62
}

log_rf(experimentID, "Seventh Run", params_7, X_train, X_test, y_train, y_test)

# COMMAND ----------

params_8 = {
  "n_estimators": 1100,
  "max_depth": 23,
  "random_state": 62
}

log_rf(experimentID, "Eighth Run", params_8, X_train, X_test, y_train, y_test)

# COMMAND ----------

params_9 = {
  "n_estimators": 800,
  "max_depth": 10,
  "random_state": 62
}

log_rf(experimentID, "Nineth Run", params_9, X_train, X_test, y_train, y_test)

# COMMAND ----------

params_10 = {
  "n_estimators": 500,
  "max_depth": 20,
  "random_state": 62
}

log_rf(experimentID, "Tenth Run", params_10, X_train, X_test, y_train, y_test)
