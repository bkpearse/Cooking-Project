# Instead of creating an image from scratch, we use this image which has python installed.
FROM python:3.10.6-buster

# Install required packages
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY project_cook project_cook
COPY setup.py setup.py
RUN pip install .


COPY Makefile Makefile
RUN make reset_local_files

# local
CMD uvicorn project_cook.api.api_file:api --host 0.0.0.0

# deploy to gcp
#CMD uvicorn project_cook.api.api_file:api --host 0.0.0.0 --port $PORT
