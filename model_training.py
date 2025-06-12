# Is MLflow server needed? It is if I want to use model registry - registry essentially creates a pointer to that artifact location within the tracking server's db
# Can I use databricks for free hosted server?
# 

import logging
from typing import List, Tuple
import pandas as pd
from aws_utils import AWSUtils
import mlflow
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import Ridge

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
logger: logging.Logger = logging.getLogger(__name__)


def initialize_dataframe(path: str) -> pd.DataFrame:
    """
    Receives path to a file and instantiates DataFrame object
    Args:
        path (str): Path to a file
    Returns:
        pd.DataFrame: DataFrame object generated from file at a given location
    """
    return pd.read_csv(path)

# timeseries cross-validation - explore this more
def backtest(df: pd.DataFrame, model, predictors: List[str], start: int = 3650, step: int = 90):
  all_predictions: List[pd.DateFrame] = [] 

  for i in range(start, df.shape[0], step):
    train: pd.DataFrame = df.iloc[:i, :] # All of the rows until row i - I will not have predictions for first 10 years because I use it for a the first training dataset
    test: pd.DataFrame = df.iloc[i:(i + step), :] # Step size number of examples will be our test data

    model.fit(train[predictors], train["target"])
    preds = model.predict(test[predictors])
    preds: pd.Series = pd.Series(preds, index=test.index)
    
    combined: pd.DataFrame = pd.concat([test["target"], preds], axis=1)
    combined.columns = ["actual", "prediction"]
    combined["difference"] = (combined["prediction"] - combined["actual"]).abs()

    all_predictions.append(combined)

  return pd.concat(all_predictions)


def evals(df: pd.DataFrame) -> Tuple[float]:
   """
    Receives DataFrame that holds both predictions and actual labels and performs evals.
    Args:
        df (pd.DataFrame): DataFrame that holds both predictions and actual labels and performs evals.
    Returns:

   """
   mae: float = mean_absolute_error(df['actual'], df['prediction'])
   r2: float = r2_score(df['actual'], df['prediction'])
   mse: float = mean_squared_error(df['actual'], df['prediction'])
   
   return mae, r2, mse
   


if __name__ == "__main__":
    logging.info("Starting model training pipeline")
    aws_utils = AWSUtils()
    try:
        file_paths: List[str] = aws_utils.download_latest_from_s3("mlops-weather-data", "weather-features-v1/data", 1)
        df: pd.DataFrame = initialize_dataframe(file_paths[0])
        
        # Starting MLflow experiment
        exp = mlflow.set_experiment("weather_tavg_prediction")
        
        with mlflow.start_run(experiment_id=exp.experiment_id):
            # This will act as an ordinary Linear Regression since alpha is 0.0
            alpha: float = 0.0
            ridge_regression: Ridge = Ridge(alpha=alpha, random_state=42)
            predictors: List[str] = ["tavg", "prcp", "wspd"]
            predictions: pd.DataFrame = backtest(df, ridge_regression, predictors)
 
            (mae, r2, mse) = evals(predictions)
            # Logging run related data
            mlflow.log_param("Ridge regression alpha", alpha)
            mlflow.log_metrics({
               "mae": mae,
               "r2_score": r2,
               "mse": mse
            })
            mlflow.log_artifact(file_paths[0]) # Log the dataset
            # Maybe log the plot of actual vs predictions
            mlflow.log_artifact(predictions[["actual", "prediction"]].plot())
            # Tags
            mlflow.set_tags({
                "feature_version": "v1",
                "model_version": "v1",
                "regularization": "l2"
            })    

            # Log model and save it to registry  
            mlflow.sklearn.log_model(sk_model=ridge_regression,
                                     name="tavg-prediction-model",
                                     registered_model_name="sk-learn-ridge-tavg-prediction-model")


    except Exception as e:
        logging.error(f"Error in model training pipeline: {e}", exc_info=True)