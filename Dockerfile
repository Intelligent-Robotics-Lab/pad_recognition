# Docker file from the initial project -> not updated yet for the new architecture and libaries

# Use python 3.10, copy the code in, install dependencies, and run demo_script.py,

FROM python:3.10-slim

WORKDIR /app
COPY . /app

# After initial testing realized system dependencies needed to be added
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install python dependencies
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python"]
CMD ["demo_script.py"]