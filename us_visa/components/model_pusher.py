import sys
import os

from us_visa.cloud_storage.aws_storage import SimpleStorageService
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from us_visa.entity.config_entity import ModelPusherConfig
from us_visa.entity.s3_estimator import USvisaEstimator


class ModelPusher:
    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        """
        :param model_evaluation_artifact: Output reference of data evaluation artifact stage
        :param model_pusher_config: Configuration for model pusher
        """
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        self.s3 = None
        self.usvisa_estimator = None

        try:
            if os.getenv("AWS_ACCESS_KEY_ID"):
                self.s3 = SimpleStorageService()
                self.usvisa_estimator = USvisaEstimator(bucket_name=model_pusher_config.bucket_name,
                                        model_path=model_pusher_config.s3_model_key_path)
            else:
                logging.warning("AWS_ACCESS_KEY_ID environment variable is not set. Skipping AWS S3 client initialization.")
        except Exception as e:
            logging.warning(f"Failed to initialize AWS S3 storage connection: {e}. Running in offline mode.")

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model pusher
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_model_pusher method of ModelTrainer class")

        try:
            if self.usvisa_estimator is not None:
                logging.info("Uploading artifacts folder to s3 bucket")
                try:
                    self.usvisa_estimator.save_model(from_file=self.model_evaluation_artifact.trained_model_path)
                    logging.info("Uploaded artifacts folder to s3 bucket")
                except Exception as e:
                    logging.warning(f"Failed to upload model to S3: {e}. Proceeding offline.")
            else:
                logging.info("Running in offline mode. Skipping S3 upload.")

            model_pusher_artifact = ModelPusherArtifact(bucket_name=self.model_pusher_config.bucket_name,
                                                        s3_model_path=self.model_pusher_config.s3_model_key_path)

            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            logging.info("Exited initiate_model_pusher method of ModelTrainer class")
            
            return model_pusher_artifact
        except Exception as e:
            raise USvisaException(e, sys) from e