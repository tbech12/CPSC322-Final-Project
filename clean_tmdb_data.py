import json
from mysklearn.mypytable import MyPyTable

def clean_data(clean=True):
    if clean:
        tmdb_table_movies = MyPyTable()
        tmdb_table_movies.load_from_file("input_data/tmdb_5000_movies.csv")
        for count, row in enumerate(tmdb_table_movies.data):
            for row_data, col in zip(row, tmdb_table_movies.column_names):
                if col == 'genres':
                    genre = []
                    data_copy = list(eval(row_data))
                    for val in data_copy:
                        genre.append(val['name'])
                    tmdb_table_movies.data[count][1] = genre
                if col == 'keywords':
                    keywords = []
                    data_copy = list(eval(row_data))
                    for val in data_copy:
                        keywords.append(val['name'])
                    tmdb_table_movies.data[count][tmdb_table_movies.column_names.index(col)] = keywords
                if col == 'production_companies':
                    companies = []
                    data_copy = list(eval(row_data))
                    for val in data_copy:
                        companies.append(val['name'])
                    tmdb_table_movies.data[count][tmdb_table_movies.column_names.index(col)] = companies
                if col == 'production_countries':
                    companies = []
                    data_copy = list(eval(row_data))
                    for val in data_copy:
                        companies.append(val['name'])
                    tmdb_table_movies.data[count][tmdb_table_movies.column_names.index(col)] = companies
                if col == 'spoken_languages':
                    lang = []
                    data_copy = list(eval(row_data))
                    for val in data_copy:
                        lang.append(val['name'])
                    tmdb_table_movies.data[count][tmdb_table_movies.column_names.index(col)] = lang
        tmdb_table_movies.save_to_file("input_data/tmdb_5000_movies.csv")



if __name__ == "clean_data":
    clean_data()