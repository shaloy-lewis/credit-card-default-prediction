<h1 align="center"> Credit Card Default Prediction</h1>

![--](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Overview
- This repository hosts a **CatBoost classifier** model, served via **FastAPI**, that predicts the probability of an individual defaulting on their credit card bills in the following month. The prediction is based on their demographics, credit data, payment history, and bill statements.
- The app is deployed on streamlit. Try it out <a href="https://credit-card-default-prediction-shaloy-lewis.streamlit.app/"> here </a>
- Dataset obtained from <a href="https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset"> Kaggle </a>

![--](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Getting Started

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

### A. Run with Docker
1. Clone the repository
```
git clone https://github.com/shaloy-lewis/credit-card-default-prediction.git
cd credit-card-default-prediction
```
2. Build and run the Docker container
```
docker-compose build
docker-compose up
```
3. Access the application
```
http://localhost:8080
```

### B. Run Locally Without Docker
1. Clone the repository
```
git clone https://github.com/shaloy-lewis/credit-card-default-prediction.git
cd credit-card-default-prediction
```
2. Create and activate virtualenv
```
pip install virtualenv
python3.12 -m venv venv
```
*For windows*
```
venv/Scripts/activate.bat
```
*For linux*
```
source venv/bin/activate
```
3. Install all the required packages and dependencies
```
pip install -r requirements.txt
```
5. Run the server
```
uvicorn api:app --reload --port 8080 --host 0.0.0.0
```
6. Access the application
```
http://localhost:8080
```
![--](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Getting Predictions
```
curl -X 'POST' \
  'http://localhost:[hostname]/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "LIMIT_BAL": 1000000,
  "AGE": 29,
  "BILL_AMT1": 4000,
  "BILL_AMT2": 4000,
  "BILL_AMT3": 4000,
  "BILL_AMT4": 4000,
  "BILL_AMT5": 4000,
  "BILL_AMT6": 4000,
  "PAY_AMT1": 1500,
  "PAY_AMT2": 1500,
  "PAY_AMT3": 1500,
  "PAY_AMT4": 1500,
  "PAY_AMT5": 1500,
  "PAY_AMT6": 1500,
  "EDUCATION": "graduate_school",
  "MARRIAGE": "married",
  "SEX": "female",
  "PAY_0": "bill_payment_delay",
  "PAY_2": "revolving_credit",
  "PAY_3": "bill_paid",
  "PAY_4": "bill_paid",
  "PAY_5": "bill_paid",
  "PAY_6": "bill_paid"
}'
```
Change the hostname with the hostname given on your environment

![--](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Sample Response
```
{
  "probability_of_default": 0.44088,
  "instance_feature_importance": {
    "PAY_4": 1.4777271702957786,
    "PAY_5": 0.17918025605815618,
    "PAY_AMT1": 0.11037106966416492,
    "PAY_AMT2": 0.10314596053208577,
    "PAY_6": 0.09610677884716073,
    "PAY_0": 0.037930420274472784,
    "PAY_AMT4": 0.036291074732687174,
    "MARRIAGE": 0.022433707766544284,
    "SEX": -0.000060132484535163944,
    "PAY_AMT5": -0.006883730662743441,
    "PAY_2": -0.011844956313233505,
    "BILL_AMT_AVG_6M": -0.020733354285267975,
    "PAY_AMT3": -0.022018604419026016,
    "PAY_AMT6": -0.029683160243983285,
    "AGE": -0.031806950578487064,
    "PAY_3": -0.041603067703169974,
    "EDUCATION": -0.1327952939910248,
    "LIMIT_BAL": -0.32958023693049865
  },
  "global_feature_importance": {
    "catategorical_pipeline__PAY_0_bill_payment_delay": 17.055813474118786,
    "numeric_pipeline__BILL_AMT_AVG_6M": 12.431711310501463,
    "numeric_pipeline__LIMIT_BAL": 8.657984820125861,
    "numeric_pipeline__PAY_AMT1": 6.799267441131673,
    "numeric_pipeline__PAY_AMT2": 6.5969902083702285,
    "numeric_pipeline__PAY_AMT3": 6.374757261276668,
    "numeric_pipeline__PAY_AMT4": 6.2085932285442,
    "numeric_pipeline__AGE": 6.056640768191354,
    "numeric_pipeline__PAY_AMT6": 4.4245035885795385,
    "numeric_pipeline__PAY_AMT5": 4.4217923003135375,
    "ordinal_catategorical_pipeline__EDUCATION": 2.799525641836794,
    "catategorical_pipeline__PAY_2_revolving_credit": 1.9141984681564181,
    "catategorical_pipeline__PAY_3_bill_payment_delay": 1.908993187707039,
    "catategorical_pipeline__PAY_2_bill_paid": 1.8650499567765817,
    "catategorical_pipeline__PAY_4_bill_payment_delay": 1.6767491229516107,
    "catategorical_pipeline__PAY_2_bill_payment_delay": 1.6614536149658123,
    "catategorical_pipeline__PAY_5_bill_payment_delay": 1.5890964424818403,
    "catategorical_pipeline__PAY_6_bill_payment_delay": 1.4241302826970939,
    "catategorical_pipeline__PAY_0_revolving_credit": 1.3932961919018094,
    "catategorical_pipeline__SEX_male": 0.8011364436947948,
    "catategorical_pipeline__PAY_3_bill_paid": 0.6957549188414223,
    "catategorical_pipeline__MARRIAGE_single": 0.684984822417616,
    "catategorical_pipeline__PAY_0_bill_paid": 0.6311391989500784,
    "catategorical_pipeline__PAY_3_revolving_credit": 0.33467411966255955,
    "catategorical_pipeline__PAY_4_revolving_credit": 0.30801128686171775,
    "catategorical_pipeline__PAY_4_bill_paid": 0.2912699550985375,
    "catategorical_pipeline__MARRIAGE_married": 0.2569278760791494,
    "catategorical_pipeline__PAY_5_bill_paid": 0.20975005121378454,
    "catategorical_pipeline__PAY_6_revolving_credit": 0.16247980662535025,
    "catategorical_pipeline__PAY_6_bill_paid": 0.12493276841134475,
    "catategorical_pipeline__PAY_5_revolving_credit": 0.1203529543645942,
    "catategorical_pipeline__MARRIAGE_others": 0.11803848715071436
  }
}
```
