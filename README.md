# Text-to-Speech Inference with Docker

This project uses a pre-trained TTS (Text-to-Speech) model to generate audio from text input based on a reference speaker. The setup is containerized using Docker to handle all dependencies seamlessly.

## Prerequisites

Before getting started, make sure you have [Docker](https://www.docker.com/get-started) installed on your machine.

## Steps to Run the Code

### 1. Build the Docker Image

The Dockerfile is configured to install all necessary dependencies and set up the environment. To build the Docker image, follow these steps:

1. Open a terminal and navigate to the project directory (where the Dockerfile is located).
2. Run the following command to build the Docker image:

   ```bash
   docker build -t python-input-app .

2. Run the following command to run the inference code that will prompt user for input text to be synthesized.

   ```bash
   docker run -it python-input-app
