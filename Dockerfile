# Instead of creating an image from scratch, we use this image which has python installed.
<<<<<<< HEAD
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
=======
FROM python:3.8.6-buster


# COPY allows you to select the folders and files to include in your docker image
# Here, we will include our api_folder and the requiremenets.txt file
COPY api_folder /api_folder
COPY project_cook /project_cook
COPY requirements.txt /requirements.txt

# RUN allows you to run terminal commands when your image gets created
# Here, we upgrade pip and install the libraries in our requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt



# CMD controls the functionality of the container
# Here, we use uvicorn to control the web server ports

# local
CMD uvicorn api_folder.api_file:api --host 0.0.0.0

# deploy to gcp
# CMD uvicorn api_folder.api_file:api --host 0.0.0.0 --port 8080
>>>>>>> main
