# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory to /app
WORKDIR /app

# Clone the project from GitHub
RUN git clone https://github.com/Satcomx00-x00/AI-Guess-Price-BTC.git .


RUN python -m pip install --upgrade pip

COPY * ./
# Install any necessary dependencies
RUN pip install -r requirements.txt

# python: can't open file '/home/container/predict.py': [Errno 2] No such file or directory^
COPY predict.py ./

# Set the entrypoint to the prediction script
# RUN two python scripts
ENTRYPOINT ["python", "predict.py"]
