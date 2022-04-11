import get_omdb_data as omdb
import clean_tmdb_data as tmdb
import prediction_class as pc
import mysklearn.myutils as myutils
from mysklearn.mypytable import MyPyTable
from mysklearn.myclassifiers import MyKNeighborsClassifier, MyDummyClassifier, MyNaiveBayesClassifier, MyDecisionTreeClassifier


def main():
    omdb_table = MyPyTable()
    omdb_table.load_from_file("input_data/omdb_data.csv")
    tmdb_table_movies = MyPyTable()
    tmdb_table_movies.load_from_file("input_data/tmdb_5000_movies.csv")
    print(tmdb_table_movies.column_names)
    print(omdb_table.column_names)
    movie_table = omdb_table.perform_full_outer_join(tmdb_table_movies, ['title', 'genres', 'runtime', 'release_date'])
    movie_table = myutils.convert_to_list_and_get_first(movie_table, 'genres')
    movie_table = myutils.convert_to_list_to_int(movie_table, 'runtime')
    movie_table = myutils.convert_to_list_to_date(movie_table, 'release_date')
    knn, dummy, naive, tree = pc.set_up(movie_table)


if __name__ == "__main__":
    omdb.get_data()
    tmdb.clean_data(False) #Set True for first run
    main()