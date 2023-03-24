FROM python:3.9

# Set the working directory to /app
WORKDIR /app

# Clone the project from GitHub
RUN git clone https://github.com/Satcomx00-x00/AI-Guess-Price-BTC.git .

# Install any necessary dependencies
RUN pip install -r requirements.txt

# Copy the prediction script to the working directory
COPY AI-Guess-Price-BTC/* .

# Set the entrypoint to the prediction script
ENTRYPOINT ["python", "predict.py"]
