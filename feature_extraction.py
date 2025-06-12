# Ova skripta se okida nakon sto data_ingestion.py skripta zavrsi svoj posao
# Ideja je da procita poslednji raw fajl i da odradi feature extraction
# Uredjena verzija feature-a cuva se na bucket-u (ili feature store-u, ali prvo bih isao sa jednostavnom verzijom)
# Ekstrahovanje iz vremenske kolone - koji je dan u nedelji, dan u mesecu, mesec, godina, godisnje doba, vikend ili radni dan
# Ciklicne transformacije za vreme
# Lagged Feature-i - da bi predvideo podatke za sutra, trebaju mi podaci od juce, prekljuce, itd...
# tavg_lag_1(prosecna temperatura juce), prcp_lag_2(padavine pre dva dana npr), broj lagova zavisi od toga koliko nam je istorijskih podataka korisno za predvidjanje
# Rolling statisike (pokretne statistike) - prosek, min/max, sume preko kliznih prozora avg_tavg_last_7_days
# Lokacijski feature-i, kako da ukljucim informaciju o lokaciji, da li kao One Hot Encoding ili da ukljucim koordinate
# Rukovanje nedostajucim vrednostima

# Zadatak 1 - predvidjanje sutrasnje temperature

from typing import List, Tuple
from aws_utils import AWSUtils
import logging
import pandas as pd

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


def clean_data(df: pd.DataFrame, target_column: str, columns_to_drop: List[str], columns_to_keep: List[str]) -> Tuple[pd.DataFrame]:
    """
    Function which is responsible for cleaning raw data and produces clean data which will be used for training model and prediction
    Args:
        df (pd.DataFrame): DataFrame containg raw data
        target_column (str): Name of the column which will be used as target column values
        columns_to_drop (List[str]): List of columns that will be dropped from DataFrame
        columns_to_keep (List[str]): List of columns that will be keept in the DataFrame
    Returns:
        Tuple[pd.DataFrame]: Clean DataFrames which can be used for downstream tasks
    """
    # I will copy the original df so I do not change it directly
    df_cleaned = df.copy()
    # Drop columns
    df_cleaned = df_cleaned.drop(columns_to_drop, axis=1)
    # Set set and conver index
    df_cleaned = df_cleaned.set_index('time')
    df_cleaned.index = pd.to_datetime(df_cleaned.index)
    # Shift data and create target - crucial for timeseries predictions
    df_cleaned["target"] = df_cleaned.shift(-1)[target_column]
    # Keep the predictor and target columns
    df_cleaned = df_cleaned[columns_to_keep]
    # Forward fill columns
    df_cleaned = df_cleaned.ffill()
    # Take the last row and save it to for inference
    df_inference = df_cleaned.tail(1)
    # Drop the last value - it will have NaN for target since we do not have data for tomorrow's tavg
    df_cleaned = df_cleaned[:-1]

    return df_cleaned, df_inference


if __name__ == "__main__":
    logging.info("Starting feature extraction pipeline")
    aws_utils = AWSUtils()
    try:
        file_paths: List[str] = aws_utils.download_latest_from_s3("mlops-weather-data", "raw-weather-data/data", 1)
        df: pd.DataFrame = initialize_dataframe(file_paths[0])
        columns_to_drop: List[str] = ["snow", "wdir", "wpgt", "pres", "tsun"] # These had more than 50% of NaN values when I did EDA
        columns_to_keep: List[str] = ["tavg", "prcp", "wspd", "target"] # Additionaly dropping location, lat, lon, alt implicitly for features v1
        target_column: str = "tavg"
        (clean_df, df_inference) = clean_data(df, target_column, columns_to_drop, columns_to_keep)
        aws_utils.upload_file_to_bucket(clean_df, 'mlops-weather-data', 'weather-features-v1')
        aws_utils.upload_file_to_bucket(df_inference, 'mlops-weather-data', 'weather-inference-v1')
    except Exception as e:
        logging.error(f"Error in feature extraction pipeline: {e}", exc_info=True)