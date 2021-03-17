#Use python as base image
FROM python:3.9-slim

#Use working directiory /app
WORKDIR /app

#Copy all the content of current directory to /app
ADD . /app

#Install required packages
RUN pip install --trusted-host pypi.python.org -r requirements.txt

#Open port
EXPOSE 5000

#Set env var
ENV NAME OpentoAll

# Run python program
CMD ["python","app.py"]
