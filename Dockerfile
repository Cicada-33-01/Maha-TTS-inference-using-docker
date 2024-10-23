# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Run setup.py to download dependencies
RUN python setup.py

# Download the dataset by running download.py
RUN python download.py

# Run the inference script
CMD ["python", "inference.py"]
