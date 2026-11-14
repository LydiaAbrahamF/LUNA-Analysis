# Reproducible environment for the LUNA16 nodule analysis.
# Python 3.11 is deliberate: PyRadiomics 3.0.1 does not build on Python 3.12/3.13.
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

# C/C++ compiler: PyRadiomics compiles its C extensions at install time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-docker.txt .

# Order matters. PyRadiomics must be compiled against the same numpy that runs at
# runtime, so numpy and its build helpers go in first, then PyRadiomics without
# build isolation, then everything else (numpy is pinned identically).
RUN pip install --upgrade pip setuptools wheel versioneer numpy==2.4.6 \
    && pip install --no-build-isolation pyradiomics==3.0.1 \
    && pip install -r requirements-docker.txt

COPY pytest.ini .
COPY src ./src
COPY pipeline ./pipeline
COPY tests ./tests

# Default: run the tests (they include a synthetic end-to-end pipeline run).
# Real data: mount it at /app/data and run `bash pipeline/run_all.sh`.
CMD ["python", "-m", "pytest", "-q"]
