# Use Python 3.10 slim image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

COPY requirements.txt .
# Install dependencies, including PyTorch for model loading
RUN pip install --no-cache-dir -r requirements.txt
COPY . . 
# Expose the port FastAPI will run on
ENV PORT=8000

# Ensure user has a valid home directory
ENV HOME=/app

# Start the FastAPI app
CMD ["uvicorn", "main:app","--host", "0.0.0.0", "--port", "8000"]

