# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory to /app
WORKDIR /app

# Clone the project from GitHub
RUN git clone https://github.com/Satcomx00-x00/AI-Guess-Price-BTC.git .


RUN python -m pip install --upgrade pip

COPY * ./
# Install any necessary dependencies
RUN pip install -r requirements.txt --force --allow-multiple

# Copy the prediction script to the working directory


# Set the entrypoint to the prediction script
# RUN two python scripts
ENTRYPOINT ["python", "predict.py"]
