from typing import Any, Dict, List
import boto3
import pandas as pd
from botocore.exceptions import ClientError
import os
from dotenv import load_dotenv
import logging
from datetime import datetime

load_dotenv()

class AWSUtils:
    """
    Class that is responsible for interactions with AWS services
    """

    def __init__(self):
        self.session = boto3.Session(os.environ.get("AWS_ACCESS_KEY"),
                                     os.environ.get("AWS_SECRET_ACCESS_KEY"),
                                     region_name="us-east-1")
        self.s3 = self.session.client('s3')
        self.logger = logging.getLogger(__name__)
    

    def upload_file_to_bucket(self,
                              df: pd.DataFrame,
                              bucket_name: str,
                              s3_directory_name: str) -> None:
        """
        Upload the processed DataFrame to S3
        Args:
            df (pd.DataFrame): DataFrame to be uploaded to bucket 
            bucket_name (str): Name of the bucket where file will be uploaded
            s3_directory_name (str): Name of the directory where file will be uploaded
        """ 
        try:
            current_time: datetime = datetime.now()
            output_filename: str = f'data_{current_time.strftime("%Y%m%d_%H%M%S")}.csv'
            # Once I run the workflow I might need to change this to tmp
            output_path: str = f'data/{output_filename}'
            df.to_csv(f"{output_path}")
            self.s3.upload_file(output_path,
                                bucket_name,
                                f'{s3_directory_name}/{output_filename}')
        except ClientError as e:
            self.logger.error(f"Error while uploading {output_filename}: {e}",
                              exc_info=True)
            raise
        finally:
            if output_path and os.path.exists(output_path):
                os.remove(output_path)


    def list_bucket_items(self, bucket_name: str, prefix: str) -> List[Dict[str, Any]] | None:
        """
        List items that are available in the S3 bucket
        Args:
            bucket_name (str): Name of the bucket
            prefix (str): Limits the response to keys that begin with the specified prefix.
        Returns:
            List[Dict[str, Any]] | None: Returns the list of dictionaries that hold information about files found in bucket
            or None if no items were found
        """
        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
            contents: List[Dict[str, Any]] = []

            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        contents.append(obj)

            return contents
        except Exception as e:
            # Should I just raise the error not log it here?
            self.logger.error(f"Error while fetching data from bucket: {e}", exc_info=True)
            raise


    def sort_items_by_date(self, contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sorts the content of s3 folder by 'LastModified' field where the last modifed files will be returned at the begging of the list.
        Args:
            contents (List[Dict[str, Any]]): List of files/folders that will be sorted by the 'LastModified' date
        Returns:
            List[Dict[str, Any]]: Sorted List of files/folders from s3 bucket
        """
        if len(contents) < 2:
            return contents
        else:
            pivot = contents[0]
            older_than_pivot = [content for content in contents[1:] if content.get('LastModified') < pivot.get('LastModified')]
            newer_than_pivot = [content for content in contents[1:] if content.get('LastModified') > pivot.get('LastModified')]

            return self.sort_items_by_date(newer_than_pivot) + [pivot] + self.sort_items_by_date(older_than_pivot)


    def download_latest_from_s3(self, bucket_name: str, prefix: str, number_of_latest: int = 1, download_dir: str = 'data') -> List[str]:
        """
        Downloads the lates number_of_latest files from s3 bucket
        Args:
            bucket_name (str): Name of the bucket
            prefix (str): Limits the response to keys that begin with the specified prefix.
            number_of_latest (int): Number of latest documents that will be downloaded
            download_dir (str): Name of the directory where we will store downloaded data
        Returns:
            List[str]: List of file paths of downloaded files
        """
        try:
            contents: List[Dict[str, Any]] | None = self.list_bucket_items(bucket_name, prefix)
            if contents:
                sorted_contents: List[Dict[str, Any]] = self.sort_items_by_date(contents)
                latest_docs: List[Dict[str, Any]] = sorted_contents[:number_of_latest]
                file_paths: List[str] = []
                
                for doc in latest_docs:
                    doc_key: str = doc.get('Key')
                    if not os.path.exists(download_dir):
                        os.makedirs(download_dir)
                    file_name = os.path.join(download_dir, os.path.basename(doc_key))
                    self.s3.download_file(bucket_name, doc_key, file_name)
                    file_paths.append(file_name)
                
                return file_paths
        except Exception as e:
            self.logger.error(f"Error while downloading from s3: {e}",
                              exc_info=True)
            raise 
