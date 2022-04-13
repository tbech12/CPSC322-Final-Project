import os
import get_omdb_data as omdb
import clean_tmdb_data as tmdb
import prediction_class as pc
import mysklearn.myutils as myutils
from mysklearn.mypytable import MyPyTable
from mysklearn.myclassifiers import MyKNeighborsClassifier, MyDummyClassifier, MyNaiveBayesClassifier, MyDecisionTreeClassifier

def clean_movie_data():
    omdb_table = MyPyTable()
    omdb_table.load_from_file("input_data/omdb_data.csv")
    tmdb_table_movies = MyPyTable()
    tmdb_table_movies.load_from_file("input_data/tmdb_5000_movies.csv")
    movie_table = MyPyTable()
    movie_table = omdb_table.perform_full_outer_join(tmdb_table_movies, ['title', 'genres', 'runtime', 'release_date', 'rated', 'imdbrating'])
    columns_to_remove = set(['title', 'genres', 'runtime', 'release_date', 'rated', 'imdbrating']) ^ set(movie_table.column_names)
    for col in columns_to_remove:
        movie_table.drop_col(col)
    movie_table = myutils.convert_to_list_to_date(movie_table, "release_date")
    movie_table.save_to_file("input_data/cleaned_movie_data.csv")
    return movie_table

def main():
    file_exists = os.path.exists('input_data/cleaned_movie_data.csv')
    movie_table = MyPyTable()
    if not(file_exists):
        movie_table = clean_movie_data()
    else:
        movie_table.load_from_file("input_data/cleaned_movie_data.csv")
    knn, dummy, naive, tree = pc.set_up(movie_table)


if __name__ == "__main__":
    omdb.get_data()
    tmdb.clean_data(False) #Set True for first run
    main()