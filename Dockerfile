FROM python:3.10.6

WORKDIR /prod

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY project_cook project_cook
COPY setup.py setup.py
COPY new_index new_index
COPY notebooks/model.h5 notebooks/model.h5
COPY notebooks/images notebooks/images
RUN pip install .

# COPY Makefile Makefile
# RUN make reset_local_files

# local
CMD uvicorn project_cook.api_folder.api_file:api --host 0.0.0.0

# deploy to gcp
#CMD uvicorn project_cook.api_folder.api_file:api --host 0.0.0.0 --port $PORT
