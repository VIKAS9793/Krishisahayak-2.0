# ==============================================================================
# Stage 1: Builder
# This stage installs dependencies, including any needed for compiling wheels.
# The resulting packages will be copied to the final stage.
# ==============================================================================
FROM python:3.11-slim-bookworm AS builder

# Set environment variables for a clean, consistent build
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PIP_NO_CACHE_DIR 1
ENV PIP_DISABLE_PIP_VERSION_CHECK 1

# Set the working directory
WORKDIR /app

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the dependency definition file first to leverage layer caching
COPY pyproject.toml .

# Install dependencies. We install all optional groups to ensure all tools
# are available for building any necessary components.
# This creates a virtual environment inside the builder stage.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip
RUN pip install ".[dev,test,deploy]"

# ==============================================================================
# Stage 2: Final Production Image
# This stage creates the final, minimal image for production.
# ==============================================================================
FROM python:3.11-slim-bookworm AS final

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user to run the application for enhanced security
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /home/appuser/app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy the application source code
COPY src/ .

# Change ownership of the app directory to the non-root user
RUN chown -R appuser:appuser /home/appuser/app

# Switch to the non-root user
USER appuser

# Set the PATH to include the virtual environment's bin directory
ENV PATH="/opt/venv/bin:$PATH"

# Expose the port the app runs on
EXPOSE 8000

# The container will run the command specified in `docker run` or `docker-compose`.
# For example, to run the API: `docker run -p 8000:8000 <image> uvicorn krishi_sahayak.api.app:app --host 0.0.0.0 --port 8000`
CMD ["uvicorn", "krishi_sahayak.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
