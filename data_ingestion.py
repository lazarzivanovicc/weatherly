from typing import Dict, List
import requests
import logging
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from meteostat import Point, Daily
import os

from aws_utils import AWSUtils

load_dotenv()


START_DATE: datetime = datetime(1985, 1, 1) 
CURRENT_DATE = datetime.now()
END_DATE: datetime = datetime(CURRENT_DATE.year, CURRENT_DATE.month, CURRENT_DATE.day)
LOCATIONS: Dict[str, Point] = {
        "Belgrade": Point(lat=44.7872, lon=20.4573, alt=117)
    }


logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
logger: logging.Logger = logging.getLogger(__name__)


def enrich_dataframe(df: pd.DataFrame, location_name: str, coordinates: Point) -> None:
    """
    Enriches dataframe with additional locational data
    Args:
        df (pd.DataFrame): DataFrame which will be enriched
        location_name (str): name of the location for which we will fetch data
        coordinates (Point): Point object that contains latitude, longitude and altitude of a location
    """
    df["lat"] = coordinates._lat
    df["lon"] = coordinates._lon
    df["alt"] = coordinates._alt
    df["location_name"] = location_name


if __name__ == "__main__":
    logger.info("Data Ingestion Pipeline Started")

    data: pd.DataFrame = pd.DataFrame()
    for location_name, coordinates in LOCATIONS.items():
        logging.info(f"Fetching data for {location_name}")
        try:
            loc_data: pd.DataFrame = Daily(coordinates, START_DATE, END_DATE).fetch()
            enrich_dataframe(loc_data, location_name, coordinates)
            data = pd.concat([data, loc_data], axis=0) # columns: tavg, tmin, tmax, prcp, snow, wdir, wspd, wpget, pres, tsun
        except Exception as e:
            logger.error(f"Error occured while fetching data for location: {location_name}: {e}",
                         exc_info=True)
            # Is it a problem if I continue and there was no data for a given location - feature extraction should think about this?
            continue       
    
    # Upload raw data to S3 - Data Lake
    aws_utils: AWSUtils = AWSUtils()
    try:
        aws_utils.upload_file_to_bucket(data, 'mlops-weather-data', 'raw-weather-data')
    except Exception as e:
        logger.error(f"Error while uploading file to S3 bucket: {e}", exc_info=True)
