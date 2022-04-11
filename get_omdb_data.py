import requests
import csv
import os
import json
from mysklearn.mypytable import MyPyTable

OMDB_API_URL = "http://www.omdbapi.com/?apikey=60a1d5e4&"
FIELDS = ['title', 'year', 'rated', 'release_date', 'runtime',
          'genre', 'director', 'writer', 'actors', 'plot',
          'language', 'country', 'awards', 'poster',
          'ratings', 'metascore', 'imdbrating', 'imdbvotes',
          'imdbid', 'type', 'dvd', 'boxoffice', 'production',
          'website', 'response']

def get_data():
    file_exists = os.path.exists('input_data/omdb_data.csv')
    if not(file_exists):
        download_from_start()
    else:
        download_from_given_place()

def get_list_of_titles():
    file_name = ("input_data/movie_names.csv")
    table = MyPyTable().load_from_file(file_name)
    return table

def download_from_start():
    table = get_list_of_titles()
    movie_result = []
    movie_list = []
    for movie in table.data:
        movie_list.append(movie[0])
    for count, movie in enumerate(movie_list): #Only allowed 1000 calls a day
        r_params = {"t":movie}
        r = requests.get(url = OMDB_API_URL, params=r_params)
        if r.status_code == 200:
            r_data = r.json()
            if "False" in r_data:
                print(r_data)
            try:
                r_data = json.dumps(r_data)
                r_data = json.loads(r_data)
                if ['False', 'Movie not found!']!= list(r_data.values()):
                    movie_result.append(list(r_data.values()))
            except:
                print("Could not find", movie)
        elif r.status_code == 401:
            r_data = r.json()
            if r_data["Error"] == 'Request limit reached!':
                break;
    with open('input_data/omdb_data.csv', 'w') as file:
        write = csv.writer(file)
        write.writerow(FIELDS)
        write.writerows(movie_result)
    file.close()
    with open('input_data/left_off_at.txt', 'w') as file:
        file.write(str(count))
    file.close()

def download_from_given_place():
    table = get_list_of_titles()
    movie_result = []
    movie_list = []
    with open('input_data/left_off_at.txt', 'r') as file:
        previous_count = int(file.readline())
    file.close()
    for movie in table.data[previous_count:]:
        movie_list.append(movie[0])
    for count, movie in enumerate(movie_list): #Only allowed 1000 calls a day
        r_params = {"t":movie}
        r = requests.get(url = OMDB_API_URL, params=r_params)
        if r.status_code == 200:
            print(count)
            r_data = r.json()
            print(r_data)
            try:
                r_data = json.dumps(r_data)
                r_data = json.loads(r_data)
                if ['False', 'Movie not found!']!= list(r_data.values()):
                    movie_result.append(list(r_data.values()))
            except:
                print("Could not find", movie)
        elif r.status_code == 401:
            r_data = r.json()
            if r_data["Error"] == 'Request limit reached!':
                break;
    with open('input_data/omdb_data.csv', 'a') as file:
        write = csv.writer(file)
        write.writerows(movie_result)
    with open('input_data/left_off_at.txt', 'w') as file:
        file.write(str(count + previous_count))
    file.close()

if __name__ == "get_data":
    get_data()