FROM python:3.10-slim-bullseye

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD ["python", "app.py"]

EXPOSE 8080