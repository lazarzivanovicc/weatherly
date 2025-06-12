from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from aws_utils import AWSUtils
import pandas as pd

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def initialize_dataframe(path: str) -> pd.DataFrame:
    """
    Receives path to a file and instantiates DataFrame object
    Args:
        path (str): Path to a file
    Returns:
        pd.DataFrame: DataFrame object generated from file at a given location
    """
    return pd.read_csv(path)


@app.get("/api/latest-prediction")
def get_latest_prediction():
    aws_utils = AWSUtils()
    try:
        file_paths: List[str] = aws_utils.download_latest_from_s3("mlops-weather-data", "weather-inference-results-v1/data", 1)
        df: pd.DataFrame = initialize_dataframe(file_paths[0])
        pred: float = df['prediction'][0]

        return {"prediction": pred}
    except Exception as e:
        return {"error": f"Error occured while fetching prediction: {e}"}
