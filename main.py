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
    movie_table = omdb_table.perform_full_outer_join(tmdb_table_movies, ['title', 'genres', 'runtime', 'release_date'])
    movie_table.drop_col("website")
    movie_table.drop_col("dvd")
    movie_table.drop_col("original_title")
    movie_table.drop_col("id")
    movie_table.drop_col("tagline")
    for column in movie_table.column_names:
        movie_table.replace_missing_values_with_column_most_common(column)

    for column in movie_table.column_names:
        try:
            test_list = movie_table.get_column(column)
            res = True
            for ele in test_list:
                if not isinstance(ele, type(test_list[0])):
                    res = False
                    break
        except:
            pass
        if res == False:
            if column == 'title':
                movie_table.convert_row_to_same_type(column, type(test_list[0]))
            elif column == 'year':
                movie_table.convert_row_to_same_type(column, type(test_list[0]))
            elif column == 'imdbvotes':
                movie_table.convert_row_to_same_type(column, float)
            elif column == 'imdbrating':
                movie_table.convert_row_to_same_type(column, float)
            elif column == 'metascore':
                input("press to go")
                movie_table.convert_row_to_same_type(column, float)
                input("u done")
            else:
                movie_table.convert_row_to_same_type(column, type(ele))
    movie_table.replace_na_with_most_common()
    print(movie_table.column_names)
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