import requests
import csv
import os
from mysklearn.mypytable import MyPyTable

OMDB_API_URL = "http://www.omdbapi.com/?apikey=60a1d5e4&"
FIELDS = ["Title", "Year", "Rated", "Released", "Runtime",
          "Genre", "Director", "Writer", "Actors", "Plot",
          "Language", "Country", "Awards", "Poster", "Ratings",
          "Metascore", "imdbRating", "imdbVotes", "imdbID",
          "Type", "DVD", "BoxOffice", "Production",
          "Website", "Response"]

def get_data():
    file_exists = os.path.exists('input_data/omdb_data.csv')
    if not(file_exists):
        table = get_list_of_titles()
        movie_result = []
        movie_list = []
        for movie in table.data:
            movie_list.append(movie[0])
        for count, movie in enumerate(movie_list): #Only allowed 1000 calls a day
            r_params = {"t":movie}
            r = requests.get(url = OMDB_API_URL, params=r_params)
            print(r)
            if r.status_code == 200:
                print(count)
                r_data = r.json()
                try:
                    for data in r_data["Search"]:
                        movie_result.append(list(data.values()))
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

def get_list_of_titles():
    file_name = ("input_data/movie_names.csv")
    table = MyPyTable().load_from_file(file_name)
    return table

if __name__ == "get_data":
    get_data()