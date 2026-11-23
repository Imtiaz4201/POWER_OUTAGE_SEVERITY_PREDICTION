# Container image for AWS App Runner / ECS (Elastic Beanstalk can use the Procfile instead).
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY application.py predictor.py ./
COPY models ./models
COPY templates ./templates
COPY static ./static
ENV PORT=8000
EXPOSE 8000
CMD ["gunicorn", "--bind", ":8000", "--workers", "1", "--threads", "4", "--timeout", "60", "application:application"]
