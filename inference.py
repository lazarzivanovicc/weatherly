from typing import List, Tuple
from aws_utils import AWSUtils
import logging
import pandas as pd
import mlflow.sklearn

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


if __name__ == "__main__":
    logging.info("Starting inference pipeline")
    aws_utils = AWSUtils()
    try:
        file_paths: List[str] = aws_utils.download_latest_from_s3("mlops-weather-data", "weather-inference-v1/data", 1)
        df: pd.DataFrame = initialize_dataframe(file_paths[0])
        predictors: List[str] = ["tavg", "prcp", "wspd"]
        
        # Needed to setup uri because my registry server is now hosted by DB
        mlflow.set_registry_uri("databricks-uc")
        
        model_name = "workspace.default.sk-learn-ridge-tavg-prediction-model@champ"
        # model_version = "latest" - This was necessary for local usage - now when I use remote tracking server and registry I must resort to alisa @champ from above

        # Load the model from the Model Registry
        model_uri = f"models:/{model_name}"
        model = mlflow.sklearn.load_model(model_uri)

        pred: List[float]  = model.predict(df[predictors])

        df_pred = pd.DataFrame({"time": [df['time'][0]], "prediction": pred})
        df_pred.set_index("time", inplace=True)
        aws_utils.upload_file_to_bucket(df_pred, 'mlops-weather-data', 'weather-inference-results-v1')

    except Exception as e:
        logging.error(f"Error in inference pipeline: {e}", exc_info=True)