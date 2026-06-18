from us_visa.configuration.mongo_db_connection import MongoDBClient
from us_visa.constants import DATABASE_NAME
from us_visa.exception import USvisaException
from us_visa.logger import logging
from from_root import from_root
import pandas as pd
import sys
import os
from typing import Optional
import numpy as np


class USvisaData:
    """
    This class help to export entire mongo db record as pandas dataframe
    """

    def __init__(self):
        """
        """
        try:
            self.mongo_client = None
            if os.getenv("MONGODB_URL"):
                self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
            else:
                logging.warning("MONGODB_URL environment variable is not set. Will use local CSV fallback.")
        except Exception as e:
            logging.warning(f"Failed to connect to MongoDB: {e}. Will use local CSV fallback.")
        

    def export_collection_as_dataframe(self,collection_name:str,database_name:Optional[str]=None)->pd.DataFrame:
        try:
            """
            export entire collectin as dataframe:
            return pd.DataFrame of collection
            """
            if self.mongo_client is not None:
                if database_name is None:
                    collection = self.mongo_client.database[collection_name]
                else:
                    collection = self.mongo_client[database_name][collection_name]

                df = pd.DataFrame(list(collection.find()))
                if "_id" in df.columns.to_list():
                    df = df.drop(columns=["_id"], axis=1)
                df.replace({"na":np.nan},inplace=True)
                return df
            else:
                csv_path = os.path.join(from_root(), "notebook", "Visadataset.csv")
                logging.info(f"MongoDB Client is not initialized. Reading from local CSV fallback: {csv_path}")
                if not os.path.exists(csv_path):
                    raise FileNotFoundError(f"Local CSV dataset not found at {csv_path}")
                df = pd.read_csv(csv_path)
                df.replace({"na":np.nan},inplace=True)
                return df
        except Exception as e:
            raise USvisaException(e,sys)