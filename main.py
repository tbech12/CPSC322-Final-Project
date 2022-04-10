import get_omdb_data as omdb
from mysklearn.mypytable import MyPyTable

def main():
    omdb_table = MyPyTable()
    omdb_table.load_from_file("input_data/omdb_data.csv")
    tmdb_table_credits = MyPyTable()
    tmdb_table_credits.load_from_file("input_data/tmdb_5000_credits.csv")
    tmdb_table_movies = MyPyTable()
    tmdb_table_movies.load_from_file("input_data/tmdb_5000_movies.csv")

    movie_table = omdb_table.perform_full_outer_join(tmdb_table_credits, ['title'])
    movie_table = movie_table.perform_full_outer_join(tmdb_table_movies, ['title'])



if __name__ == "__main__":
    omdb.get_data()
    main()