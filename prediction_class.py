import contextlib
from typing_extensions import runtime

from sklearn.utils import shuffle
import mysklearn.myutils as myutils
import mysklearn.myevaluation as myevaluation
from mysklearn.myclassifiers import MyKNeighborsClassifier, MyDummyClassifier, MyNaiveBayesClassifier, MyDecisionTreeClassifier
from mysklearn.mypytable import MyPyTable

def set_up(movie_table: MyPyTable):
    knn = MyKNeighborsClassifier()
    dummy = MyDummyClassifier()
    naive = MyNaiveBayesClassifier()
    tree = MyDecisionTreeClassifier()

    movie_table.data = movie_table.data[:2000]

    movie_table = myutils.convert_to_list(movie_table, 'genres')
    print(movie_table.column_names)
    columns_used = ['title', 'genres', 'runtime', 'release_date', 'rated', 'imdbrating']
    for x in columns_used:
        print(movie_table.column_names.index(x), end=" ")
    print()
    titles = [str(value) for value in movie_table.get_column("title")]
    genres = movie_table.get_column("genres")
    genres = myutils.get_genre_sum(genres)
    runtimes = movie_table.get_column("runtime")
    release_dates = movie_table.get_column("release_date")
    rated = movie_table.get_column("rated")
    print(rated)
    rated = myutils.get_genre_rating(rated)
    print(rated)
    imdbrating = movie_table.get_column("imdbrating")

    print(genres[:10])
    print(runtimes[:10])
    print(release_dates[:10])
    print(rated[:10])
    print(imdbrating[:10])

    return
    genres_strings = []
    for val in genres:
        if len(val) == 0:
            genres_strings.append("Action")
        else:
            genres_strings.append(val[0])

    x_train = []
    y_train = titles

    for val in range(len(movie_table.data)):
        tmp = []
        tmp.append(genres_strings[val])
        tmp.append(rated[val])
        tmp.append(imdbrating[val])
        tmp.append(runtimes[val])
        tmp.append(release_dates[val])
        x_train.append(tmp)

    split_X_train, split_X_test, split_y_train, split_y_test = myevaluation.train_test_split(x_train, y_train)
    #knn_train, knn_test = myutils.normalize_2(split_X_train, split_X_test)

    strat_knn = []
    strat_nb = []
    strat_dummy = []
    strat_tree = []
    strat_true = split_y_test

    print("Entering Evaluation")
    ''' knn = MyKNeighborsClassifier()
    knn.fit(split_X_train, split_y_train) '''

    nb = MyNaiveBayesClassifier()
    nb.fit(split_X_train, split_y_train)

    dummy = MyDummyClassifier()
    dummy.fit(split_X_train, split_y_train)

    tree = MyDecisionTreeClassifier()
    #tree.fit(split_X_train, split_y_train)


    strat_nb = nb.predict(split_X_test)
    strat_dummy = dummy.predict(split_X_test)
    #strat_tree = tree.predict(split_X_test)

    ''' for i in range(len(split_X_test)):
        prediction = knn.predict(split_X_test)
        only_prediction = prediction[0]
        strat_knn.append(only_prediction) '''

    print(strat_knn)
    print(strat_nb)
    print(strat_dummy)
    print(strat_tree)

    knn_accuracy = myevaluation.accuracy_score(strat_true, strat_knn)
    nb_accuracy = myevaluation.accuracy_score(strat_true, strat_nb)
    dummy_accuracy = myevaluation.accuracy_score(strat_true, strat_dummy)
    tree_accuracy = myevaluation.accuracy_score(strat_true, strat_tree)
    myutils.print_accuracys(knn_accuracy, nb_accuracy, dummy_accuracy, tree_accuracy)

    for i in range(len(strat_nb)):
        if strat_nb[i] == split_y_test[i]:
            nb_accuracy += 1
    print("nb Accuracy: ", (nb_accuracy / len(split_y_test)))

    return knn, dummy, naive, tree
