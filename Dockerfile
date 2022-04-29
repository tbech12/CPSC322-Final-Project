FROM continuumio/anaconda3:2021.11
ADD . /code
WORKDIR /code
RUN pip install pip pipenv --upgrade
RUN pipenv install --skip-lock --system --dev
ENTRYPOINT ["python", "genre_app.py"]