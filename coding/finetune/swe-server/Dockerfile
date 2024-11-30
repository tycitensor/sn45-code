# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install firejail
RUN apt-get update && apt-get install -y firejail

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 3000 available to the world outside this container
EXPOSE 3000

# Run server.py with firejail when the container launches
CMD ["firejail", "--net=host", "--whitelist=/app", "--private", "--netfilter=/etc/firejail/netfilter", "python", "server.py"]

# Create netfilter file to restrict network access
RUN echo "whitelist host.docker.internal:21000" > /etc/firejail/netfilter