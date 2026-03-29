# Use an official lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files into the container
COPY . .

# Expose the Hugging Face Space port
EXPOSE 7860

# Command to run the Environment Web Server when the container starts
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
