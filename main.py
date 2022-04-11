import get_omdb_data as omdb
import clean_tmdb_data as tmdb
import prediction_class as pc
from mysklearn.mypytable import MyPyTable
from mysklearn.myclassifiers import MyKNeighborsClassifier, MyDummyClassifier, MyNaiveBayesClassifier, MyDecisionTreeClassifier


def main():
    omdb_table = MyPyTable()
    omdb_table.load_from_file("input_data/omdb_data.csv")
    tmdb_table_movies = MyPyTable()
    tmdb_table_movies.load_from_file("input_data/tmdb_5000_movies.csv")

    movie_table = omdb_table.perform_full_outer_join(tmdb_table_movies, ['title'])
    knn, dummy, naive, tree = pc.set_up(movie_table)


if __name__ == "__main__":
    omdb.get_data()
    tmdb.clean_data(False) #Set True for first run
    main()