FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create the necessary Streamlit configuration directories and files
RUN mkdir -p ~/.streamlit

# Create and populate the config.toml file to bypass the email prompt
RUN bash -c 'echo "[server]" >> ~/.streamlit/config.toml'
RUN bash -c 'echo "headless = true" >> ~/.streamlit/config.toml'

# Expose the port that Streamlit runs on
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
