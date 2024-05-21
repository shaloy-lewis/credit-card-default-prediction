import pandas as pd
import numpy as np
import os
import sys

from src.logger.logging import logging
from src.exception.exception import customexception
from src.utils.constants import DATA_PATH, TARGET_FEATURE

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    raw_data_path:str=os.path.join("artifacts","raw.csv")
    train_data_path:str=os.path.join("artifacts","X_train.csv")
    test_data_path:str=os.path.join("artifacts","X_test.csv")
    train_target_path:str=os.path.join("artifacts","y_train.csv")
    test_target_path:str=os.path.join("artifacts","y_test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        

    def initiate_data_ingestion(self):
        logging.info("initiating data ingestion")
        try:
            filepath=Path(DATA_PATH)
            data=pd.read_csv(filepath)
            logging.info("data read successfully")

            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info("raw dataset ingested successfully")
            
            logging.info("initiating train test split")
            
            X=data.drop(columns=TARGET_FEATURE)
            y=data[TARGET_FEATURE]

            X_train,X_test,y_train,y_test=train_test_split(X
                                               , y
                                               , test_size=0.30
                                               , shuffle=True
                                               , stratify=y 
                                               , random_state=42)
            
            logging.info("train test split completed")
            
            X_train.to_csv(self.ingestion_config.train_data_path,index=False)
            X_test.to_csv(self.ingestion_config.test_data_path,index=False)
            y_train.to_csv(self.ingestion_config.train_target_path,index=False)
            y_test.to_csv(self.ingestion_config.test_target_path,index=False)
            
            logging.info("data ingestion completed")
            
            return (             
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.train_target_path,
                self.ingestion_config.test_target_path
            )

        except Exception as e:
            logging.info()
            raise customexception(e,sys)


if __name__=="__main__":
    obj=DataIngestion()

    obj.initiate_data_ingestion()