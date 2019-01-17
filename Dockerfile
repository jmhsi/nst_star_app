FROM python:3.6-slim-stretch

RUN apt-get update && apt-get install -y \
    python3-dev \
    gcc \
    wget
#    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY deployment deployment/
WORKDIR "/deployment"

RUN chmod +x download_models.sh

RUN sh download_models.sh 

EXPOSE 8008

CMD ["python", "nst_star_app.py", "serve"]