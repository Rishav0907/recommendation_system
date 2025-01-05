FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt
RUN pip3 install requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]