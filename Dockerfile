FROM python:3.12.4-slim

# Install dependencies
COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt --log /app/pip_install.log

# Copy application files
COPY ./api.py /app/api.py
COPY ./init_setup.sh /app/init_setup.sh
COPY ./src /app/src  
COPY ./artifacts/model.pkl /app/artifacts/model.pkl
COPY ./artifacts/preprocessor.pkl /app/artifacts/preprocessor.pkl
COPY ./artifacts/outlier_threshold.json /app/artifacts/outlier_threshold.json

WORKDIR /app

# Set up the command to run the application
CMD ["sh", "init_setup.sh"]
