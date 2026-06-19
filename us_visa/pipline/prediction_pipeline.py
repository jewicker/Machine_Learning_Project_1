import os
import sys
import glob
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np
import pandas as pd
from us_visa.entity.config_entity import USvisaPredictorConfig
from us_visa.entity.s3_estimator import USvisaEstimator
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import read_yaml_file, load_object
from pandas import DataFrame
from from_root import from_root


class USvisaData:
    def __init__(self,
                continent,
                education_of_employee,
                has_job_experience,
                requires_job_training,
                no_of_employees,
                region_of_employment,
                prevailing_wage,
                unit_of_wage,
                full_time_position,
                company_age
                ):
        """
        Usvisa Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.continent = continent
            self.education_of_employee = education_of_employee
            self.has_job_experience = has_job_experience
            self.requires_job_training = requires_job_training
            self.no_of_employees = no_of_employees
            self.region_of_employment = region_of_employment
            self.prevailing_wage = prevailing_wage
            self.unit_of_wage = unit_of_wage
            self.full_time_position = full_time_position
            self.company_age = company_age


        except Exception as e:
            raise USvisaException(e, sys) from e

    def get_usvisa_input_data_frame(self)-> DataFrame:
        """
        This function returns a DataFrame from USvisaData class input
        """
        try:
            
            usvisa_input_dict = self.get_usvisa_data_as_dict()
            return DataFrame(usvisa_input_dict)
        
        except Exception as e:
            raise USvisaException(e, sys) from e


    def get_usvisa_data_as_dict(self):
        """
        This function returns a dictionary from USvisaData class input 
        """
        logging.info("Entered get_usvisa_data_as_dict method as USvisaData class")

        try:
            input_data = {
                "continent": [self.continent],
                "education_of_employee": [self.education_of_employee],
                "has_job_experience": [self.has_job_experience],
                "requires_job_training": [self.requires_job_training],
                "no_of_employees": [self.no_of_employees],
                "region_of_employment": [self.region_of_employment],
                "prevailing_wage": [self.prevailing_wage],
                "unit_of_wage": [self.unit_of_wage],
                "full_time_position": [self.full_time_position],
                "company_age": [self.company_age],
            }

            logging.info("Created usvisa data dict")

            logging.info("Exited get_usvisa_data_as_dict method as USvisaData class")

            return input_data

        except Exception as e:
            raise USvisaException(e, sys) from e

class USvisaClassifier:
    def __init__(self,prediction_pipeline_config: USvisaPredictorConfig = USvisaPredictorConfig(),) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            # self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise USvisaException(e, sys)

    def get_latest_local_model_path(self) -> str:
        """
        Finds the latest local model.pkl file within the artifact directory.
        """
        try:
            artifact_dir = os.path.join(from_root(), "artifact")
            if not os.path.exists(artifact_dir):
                return None
            
            # Find all model.pkl paths under artifact/
            model_paths = glob.glob(os.path.join(artifact_dir, "*", "model_trainer", "trained_model", "model.pkl"))
            if not model_paths:
                model_paths = glob.glob(os.path.join(artifact_dir, "**", "model.pkl"), recursive=True)
                
            if not model_paths:
                return None
                
            # Get the path of the most recently modified model file
            return max(model_paths, key=os.path.getmtime)
        except Exception as e:
            raise USvisaException(e, sys) from e

    def predict(self, dataframe) -> str:
        """
        This is the method of USvisaClassifier
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of USvisaClassifier class")
            
            aws_keys_configured = os.getenv("AWS_ACCESS_KEY_ID") is not None and os.getenv("AWS_SECRET_ACCESS_KEY") is not None
            
            if aws_keys_configured:
                try:
                    logging.info("AWS credentials found. Attempting to load model from S3.")
                    model = USvisaEstimator(
                        bucket_name=self.prediction_pipeline_config.model_bucket_name,
                        model_path=self.prediction_pipeline_config.model_file_path,
                    )
                    result = model.predict(dataframe)
                    return result
                except Exception as s3_err:
                    logging.warning(f"Failed to load model from S3: {s3_err}. Attempting local fallback.")
            
            logging.info("AWS credentials not configured or S3 load failed. Attempting to load local model.")
            latest_model_path = self.get_latest_local_model_path()
            if latest_model_path is None:
                raise Exception("AWS credentials are not set and no local trained model was found in the 'artifact' directory. Please run the training pipeline first or configure AWS credentials.")
                
            logging.info(f"Loading local model from: {latest_model_path}")
            model = load_object(latest_model_path)
            result = model.predict(dataframe)
            return result
        
        except Exception as e:
            raise USvisaException(e, sys)