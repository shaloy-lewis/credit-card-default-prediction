NUMERIC_FEATURES = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5'
                    , 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

OUTLIER_COLUMNS = NUMERIC_FEATURES.remove('AGE')

ORDIANAL_CATEGORICAL_FEATURES = ['EDUCATION']

CATEGORICAL_FEATURES = ['MARRIAGE', 'SEX', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

TARGET_FEATURE = ['default.payment.next.month']

INDEX_COLUMN = ['ID']

EDUCATION_CATEGORIES = ['others','high_school','university','graduate_school']

NUMERIC_FEATURES_AFTER_PREPROCESSING=[x for x in NUMERIC_FEATURES if x not in ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']]
NUMERIC_FEATURES_AFTER_PREPROCESSING.append('BILL_AMT_AVG_6M')

OUTLIER_CAPPING_LOWER_THRESHOLD=2.5
OUTLIER_CAPPING_UPPER_THRESHOLD=97.5

# Paraeters for hyperparamter tuning
PARAM_GRID={
                'iterations': [50, 100, 200, 400, 800],
                'n_estimators':[50, 100, 200, 400, 800],
                'depth': [None, 3, 5, 7, 9]
            }
