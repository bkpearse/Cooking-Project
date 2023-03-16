FROM python:3.8.12-buster

WORKDIR /prod

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY project_cook project_cook
COPY setup.py setup.py
RUN pip install .

COPY Makefile Makefile
RUN make reset_local_files

# local
CMD uvicorn api_folder.api_file:api --host 0.0.0.0

# deploy to gcp
#CMD uvicorn api_folder.api_file:api --host 0.0.0.0 --port $PORT
