FROM python:3.9

# Set the working directory to /app
WORKDIR /app

# Clone the project from GitHub
RUN git clone https://github.com/yourusername/yourproject.git .

# Install any necessary dependencies
RUN pip install -r requirements.txt

# Copy the prediction script to the working directory
COPY predict.py .

# Set the entrypoint to the prediction script
ENTRYPOINT ["python", "predict.py"]
