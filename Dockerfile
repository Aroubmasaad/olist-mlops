# Använd en liten Python-version som bas för containern.
FROM python:3.11-slim

# Skapa och använd /app som arbetsmapp i containern.
WORKDIR /app

# Kopiera runtime-kraven först.
COPY requirements/runtime.txt requirements/runtime.txt

# Installera de Python-paket som API:t behöver.
RUN pip install --no-cache-dir -r requirements/runtime.txt

# Kopiera endast det som behövs för API och inference.
COPY app/ app/
COPY src/ src/
COPY config/ config/

COPY models/preprocessor.joblib models/preprocessor.joblib
COPY models/feature_list.json models/feature_list.json

# API:t använder port 8000.
EXPOSE 8000

# Starta FastAPI med Uvicorn.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
