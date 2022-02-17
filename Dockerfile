FROM python:3.8.12-slim

RUN pip --no-cache-dir install pipenv

WORKDIR /app

COPY . /app

RUN pipenv install --system --deploy 

EXPOSE 9696

ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:9696", "predict:app"]
