FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    swig \
    git \
    cmake \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone and install FAISS
RUN git clone https://github.com/facebookresearch/faiss.git && \
    cd faiss && \
    cmake -B build . -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=ON -DFAISS_OPT_LEVEL=generic && \
    make -C build -j faiss && \
    make -C build install && \
    cd python && \
    python setup.py install && \
    cd ../.. && \
    rm -rf faiss

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Command to run the application
CMD ["python", "app.py"] 