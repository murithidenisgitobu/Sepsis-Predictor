# Use the official Python 3.10.11 image as the base image
FROM python:3.10.11

# Copy the requirements.txt file to the /tmp directory in the container
COPY requirements.txt /tmp/requirements.txt

# Install the Python dependencies specified in requirements.txt
# The --timeout option is set to 300000 seconds to prevent timeouts during installation
RUN python -m pip install --timeout 300000 -r /tmp/requirements.txt

# Copy the entire current directory (.) to the /app directory in the container
COPY . /app

# Set the working directory to /app
WORKDIR /app

# Expose port 80 for the application
EXPOSE 80

# Command to run the application using Uvicorn, specifying the host and port
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "80"]
