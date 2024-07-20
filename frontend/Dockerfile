# Use the official Python 3.10 image as the base image
FROM python:3.10

# Set the working directory to /app/frontend
WORKDIR /app/frontend

# Copy the requirements.txt file to the working directory
COPY frontend/requirements.txt /app/frontend/requirements.txt

# Install the Python dependencies specified in requirements.txt
RUN python -m pip install --no-cache-dir --timeout 300000 -r /app/frontend/requirements.txt

# Copy the entire frontend directory into the working directory
COPY frontend /app/frontend

# Expose port 8501 for the Streamlit application
EXPOSE 8501

# Command to run the Streamlit application on port 8501
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
