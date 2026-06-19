import json
import sys

import pandas as pd
from pandas import DataFrame
from scipy import stats

from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import read_yaml_file, write_yaml_file
from us_visa.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from us_visa.entity.config_entity import DataValidationConfig
from us_visa.constants import SCHEMA_FILE_PATH


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_validation_config: configuration for data validation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config =read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise USvisaException(e,sys)

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        """
        Method Name :   validate_number_of_columns
        Description :   This method validates the number of columns
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info(f"Is required column present: [{status}]")
            return status
        except Exception as e:
            raise USvisaException(e, sys)

    def is_column_exist(self, df: DataFrame) -> bool:
        """
        Method Name :   is_column_exist
        Description :   This method validates the existence of a numerical and categorical columns
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            dataframe_columns = df.columns
            missing_numerical_columns = []
            missing_categorical_columns = []
            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            if len(missing_numerical_columns)>0:
                logging.info(f"Missing numerical column: {missing_numerical_columns}")


            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if len(missing_categorical_columns)>0:
                logging.info(f"Missing categorical column: {missing_categorical_columns}")

            return False if len(missing_categorical_columns)>0 or len(missing_numerical_columns)>0 else True
        except Exception as e:
            raise USvisaException(e, sys) from e

    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys)

    def detect_dataset_drift(self, reference_df: DataFrame, current_df: DataFrame, ) -> bool:
        """
        Method Name :   detect_dataset_drift
        Description :   This method validates if drift is detected
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            # Lightweight drift detection when Evidently is not available or incompatible.
            # Numeric columns: use KS test. Categorical columns: use chi-square test.
            results = {}
            drifted = 0
            total = 0
            for col in reference_df.columns:
                if col not in current_df.columns:
                    continue
                total += 1
                ref = reference_df[col].dropna()
                cur = current_df[col].dropna()
                col_report = {}
                try:
                    if pd.api.types.is_numeric_dtype(ref) and pd.api.types.is_numeric_dtype(cur):
                        # Two-sample Kolmogorov-Smirnov test
                        if len(ref) < 2 or len(cur) < 2:
                            pvalue = 1.0
                        else:
                            _, pvalue = stats.ks_2samp(ref, cur)
                        drift = pvalue < 0.05
                        col_report["method"] = "ks_2samp"
                        col_report["p_value"] = float(pvalue)
                    else:
                        # Chi-square on category counts
                        ref_counts = ref.astype(str).value_counts()
                        cur_counts = cur.astype(str).value_counts()
                        cats = list(set(ref_counts.index) | set(cur_counts.index))
                        ref_vals = [ref_counts.get(c, 0) for c in cats]
                        cur_vals = [cur_counts.get(c, 0) for c in cats]
                        table = [ref_vals, cur_vals]
                        # If any expected cell is zero or too small, fall back to p=1
                        try:
                            chi2, pvalue, _, _ = stats.chi2_contingency(table)
                        except Exception:
                            pvalue = 1.0
                        drift = pvalue < 0.05
                        col_report["method"] = "chi2"
                        col_report["p_value"] = float(pvalue)
                except Exception as e:
                    col_report["error"] = str(e)
                    drift = False

                col_report["drift"] = bool(drift)
                results[col] = col_report
                if drift:
                    drifted += 1

            n_features = total
            n_drifted_features = drifted
            # dataset drift if >=50% of features drifted
            dataset_drift = (n_features > 0) and (n_drifted_features / n_features >= 0.5)

            json_report = {
                "data_drift": {
                    "data": {
                        "metrics": {
                            "n_features": n_features,
                            "n_drifted_features": n_drifted_features,
                            "dataset_drift": bool(dataset_drift),
                        },
                        "details": results,
                    }
                }
            }

            write_yaml_file(file_path=self.data_validation_config.drift_report_file_path, content=json_report)

            logging.info(f"{n_drifted_features}/{n_features} drift detected.")
            return bool(dataset_drift)
        except Exception as e:
            raise USvisaException(e, sys) from e

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Method Name :   initiate_data_validation
        Description :   This method initiates the data validation component for the pipeline
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """

        try:
            validation_error_msg = ""
            logging.info("Starting data validation")
            train_df, test_df = (DataValidation.read_data(file_path=self.data_ingestion_artifact.trained_file_path),
                                 DataValidation.read_data(file_path=self.data_ingestion_artifact.test_file_path))

            status = self.validate_number_of_columns(dataframe=train_df)
            logging.info(f"All required columns present in training dataframe: {status}")
            if not status:
                validation_error_msg += f"Columns are missing in training dataframe."
            status = self.validate_number_of_columns(dataframe=test_df)

            logging.info(f"All required columns present in testing dataframe: {status}")
            if not status:
                validation_error_msg += f"Columns are missing in test dataframe."

            status = self.is_column_exist(df=train_df)

            if not status:
                validation_error_msg += f"Columns are missing in training dataframe."
            status = self.is_column_exist(df=test_df)

            if not status:
                validation_error_msg += f"columns are missing in test dataframe."

            validation_status = len(validation_error_msg) == 0

            if validation_status:
                drift_status = self.detect_dataset_drift(train_df, test_df)
                if drift_status:
                    logging.info(f"Drift detected.")
                    validation_error_msg = "Drift detected"
                else:
                    validation_error_msg = "Drift not detected"
            else:
                logging.info(f"Validation_error: {validation_error_msg}")
                

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise USvisaException(e, sys) from e