import pandas as pd
from src.logger.logging import logging
from src.exception.exception import customexception
import os
import sys
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler,OneHotEncoder

from src.utils.utils import save_object

from src.utils.constants import (OUTLIER_COLUMNS,
                                 EDUCATION_CATEGORIES,
                                 NUMERIC_FEATURES_AFTER_PREPROCESSING,
                                 CATEGORICAL_FEATURES,
                                 ORDIANAL_CATEGORICAL_FEATURES,
                                 OUTLIER_CAPPING_LOWER_THRESHOLD,
                                 OUTLIER_CAPPING_UPPER_THRESHOLD)
from src.utils.utils import cap_outliers, preprocess_data

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformation(self):
        try:
            logging.info('Data transformation initiated')
            logging.info('Pipeline Initiated')
            
            numeric_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer()),
                    ("scaler",StandardScaler())
                ]
            )
            ordinal_categorical_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("ordinalencoder",OrdinalEncoder(categories=[EDUCATION_CATEGORIES]))
                ]
            )
            
            categorical_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("onehotencoder",OneHotEncoder(sparse_output=False,drop='if_binary'))
                ]
            )
            
            preprocessor=ColumnTransformer(
                [
                    ("numeric_pipeline",numeric_pipeline,NUMERIC_FEATURES_AFTER_PREPROCESSING),
                    ("catategorical_pipeline",categorical_pipeline,CATEGORICAL_FEATURES),
                    ("ordinal_catategorical_pipeline",ordinal_categorical_pipeline,ORDIANAL_CATEGORICAL_FEATURES)
                ]
            )
            
            return preprocessor
        
        except Exception as e:
            logging.info("Exception occured in get_data_transformation")
            raise customexception(e,sys)
    
    def initiate_data_transformation(self,X_train_path,X_test_path):
        try:
            X_train=pd.read_csv(X_train_path)
            X_test=pd.read_csv(X_test_path)
            
            logging.info("Read train and test data for data transformation")
            logging.info(f'Train Dataframe shape : \n{X_train.shape}')
            logging.info(f'Test Dataframe shape : \n{X_test.shape}')
            
            logging.info('Capping outliers in some numeric columns')            
            X_train, low_perc, high_perc = cap_outliers(X_train
                                            , percentile_low=OUTLIER_CAPPING_LOWER_THRESHOLD
                                            , percentile_high=OUTLIER_CAPPING_UPPER_THRESHOLD
                                            , req_columns=OUTLIER_COLUMNS)
            X_test[OUTLIER_COLUMNS] = X_test[OUTLIER_COLUMNS].clip(lower=low_perc, upper=high_perc, axis=1)
            
            logging.info('Preprocessing data')
            X_train=preprocess_data(X_train)
            X_test=preprocess_data(X_test)
            
            preprocessor = self.get_data_transformation()
            
            logging.info('Applying preprocessing object on training and testing datasets')
            X_train=pd.DataFrame(preprocessor.fit_transform(X_train),columns=preprocessor.get_feature_names_out())
            X_test=pd.DataFrame(preprocessor.transform(X_test),columns=preprocessor.get_feature_names_out())
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            logging.info("pPreprocessing file saved in pickle format")
            
            return (
                X_train,
                X_test
            )
            
        except Exception as e:
            logging.info("Exception occured in initiate_data_transformation")
            raise customexception(e,sys)
