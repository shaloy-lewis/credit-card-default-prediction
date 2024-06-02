import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.logger.logging import logging
from src.exception.exception import customexception


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logging.info('Exception occured in save_object utils')
        raise customexception(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object utils')
        raise customexception(e,sys)

def cap_outliers(df, percentile_low=2.5, percentile_high=97.5, req_columns=[]):
    try:
        numeric_cols = df[req_columns]
        
        low_perc = numeric_cols.quantile(percentile_low / 100)
        high_perc = numeric_cols.quantile(percentile_high / 100)
        
        df[req_columns] = numeric_cols.clip(lower=low_perc, upper=high_perc, axis=1)
        
        return df, low_perc, high_perc
    except Exception as e:
        logging.info('Exception occured in cap_outliers function utils')
        raise customexception(e,sys)

def preprocess_data(df):
    try:
        df['MARRIAGE'] = np.where(df['MARRIAGE']==0, 3,df['MARRIAGE'])
        df['MARRIAGE'] = df['MARRIAGE'].map({1:'married',2:'single',3:'others'})
        df['SEX'] = df['SEX'].map({1:'male',2:'female'})
        
        for i in [0,2,3,4,5,6]:
            df['PAY_{}'.format(i)] = np.where(df['PAY_{}'.format(i)]<0, "bill_paid"
                                            ,(np.where(df['PAY_{}'.format(i)]>0,"bill_payment_delay"
                                                        ,"revolving_credit")))
            
        df['EDUCATION'] = np.where(df['EDUCATION'].isin([0,4,5,6]), 4, df['EDUCATION'])
        df['EDUCATION'] = df['EDUCATION'].map({1:'graduate_school',2:'university',3:'high_school',4:'others'})
        
        df['BILL_AMT_AVG_6M'] = df[['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']].mean(axis=1).values
        df.drop(['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6'],axis=1,inplace=True)
        
        return df
    
    except Exception as e:
        logging.info('Exception occured in preprocess_data function utils')
        raise customexception(e,sys)