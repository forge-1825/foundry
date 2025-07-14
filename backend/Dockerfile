FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    curl \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    libpoppler-dev \
    docker.io \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies directly with pip (faster than Poetry for Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install specific versions of key packages
RUN pip install --no-cache-dir \
    numpy==1.23.5 \
    spacy \
    nltk==3.8.1 \
    scikit-learn==1.2.2 \
    sentence-transformers==2.2.2 \
    transformers==4.30.2 \
    torch==2.0.1 \
    PyMuPDF==1.22.5 \
    pytesseract \
    Pillow \
    beautifulsoup4 \
    selenium \
    webdriver-manager \
    tqdm \
    requests \
    openai

# Download spaCy and NLTK resources
RUN python3 -m spacy download en_core_web_sm && \
    python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Copy application code
COPY . /app/

# Expose port
EXPOSE 7433

# Start command
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7433", "--reload"]
