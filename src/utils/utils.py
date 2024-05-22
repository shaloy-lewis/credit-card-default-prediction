import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.logger.logging import logging
from src.exception.exception import customexception

from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score

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
    # Select numeric columns
    numeric_cols = df[req_columns]
    
    # Calculate percentiles
    low_perc = numeric_cols.quantile(percentile_low / 100)
    high_perc = numeric_cols.quantile(percentile_high / 100)
    
    # Cap outliers
    df[req_columns] = numeric_cols.clip(lower=low_perc, upper=high_perc, axis=1)
    
    return df, low_perc, high_perc

def preprocess_data(df):
    df['MARRIAGE'] = np.where(df['MARRIAGE']==0, 3,df['MARRIAGE'])
    
    for i in [0,2,3,4,5,6]:
        df['PAY_{}'.format(i)] = np.where(df['PAY_{}'.format(i)]<0, "bill_paid"
                                          ,(np.where(df['PAY_{}'.format(i)]>0,"bill_payment_delay"
                                                     ,"revolving_credit")))
        
    df['EDUCATION'] = np.where(df['EDUCATION'].isin([0,4,5,6]), 4, df['EDUCATION'])
    df['EDUCATION'] = df['EDUCATION'].map({1:'graduate_school',2:'university',3:'high_school',4:'others'})
    
    df['BILL_AMT_AVG_6M'] = df[['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']].mean(axis=1).values
    df.drop(['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6'],axis=1,inplace=True)
    
    return df