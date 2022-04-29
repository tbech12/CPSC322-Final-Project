FROM continuumio/anaconda3:2021.11
ADD . /code
WORKDIR /code
ENTRYPOINT ["python", "genre_app.py"]