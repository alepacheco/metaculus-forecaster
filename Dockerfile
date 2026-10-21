FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY forecaster ./forecaster
COPY tests ./tests
CMD ["python", "-m", "forecaster", "run"]
