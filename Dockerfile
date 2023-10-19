# Use an official Python runtime as the base image
FROM python:3.8
# FROM ubuntu:latest

# Set the working directory within the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt /app/

# Install the project dependencies
RUN pip install --no-cache-dir -r requirements.txt

#Create a folder to store the weights.
RUN mkdir weights

# Copy the rest of the application code into the container
COPY . /app

# Specify any environment variables that your application requires
# ENV MY_ENV_VARIABLE=value


# Set the command to run your application
# CMD ["python3", "test/save_weights.py"]
CMD ["python3", "save_weights.py"]
