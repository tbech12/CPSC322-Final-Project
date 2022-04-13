import contextlib
from typing_extensions import runtime
from more_itertools import only
from regex import P

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
    print(movie_table.column_names)
    movie_table.data = movie_table.data[:500]
    ''' columns_used = ["title", "genres", "metascore", "imdbrating", "boxoffice", "runtime", "year", "rated"]
    for x in columns_used:
        print(movie_table.column_names.index(x), end=" ")
    print() '''
    titles = [str(value) for value in movie_table.get_column("title")]
    genres = movie_table.get_column("genres")
    metascore = movie_table.get_column("metascore")
    imdbrating = movie_table.get_column("imdbrating")
    boxoffices = movie_table.get_column("boxoffice")
    boxoffices_high_low = myutils.find_low_mid_high(boxoffices, 10000000.0, 500000000.0)
    runtimes = movie_table.get_column("runtime")
    years = movie_table.get_column("year")
    release_dates = movie_table.get_column("release_date")
    rated = movie_table.get_column("rated")

    x_train = []
    y_train = titles
    for val in range(len(movie_table.data)):
        tmp = []
        tmp.append(genres[val])
        tmp.append(metascore[val])
        tmp.append(imdbrating[val])
        tmp.append(boxoffices_high_low[val])
        tmp.append(runtimes[val])
        tmp.append(years[val])
        tmp.append(release_dates[val])
        tmp.append(rated[val])
        x_train.append(tmp)


    nb = MyNaiveBayesClassifier()
    nb.fit(x_train, y_train)
    tree = MyDecisionTreeClassifier()
    tree.fit(x_train, y_train)
    dummy = MyDummyClassifier()
    dummy.fit(x_train, y_train)
    random_forest = None

    strat_train_folds, strat_test_folds = myevaluation.stratified_kfold_cross_validation(x_train, y_train, n_splits=20)
    strat_nb = []
    strat_dummy = []
    strat_tree = []
    strat_true = []

    print("Enter Eval")
    for train, test in zip(strat_train_folds, strat_test_folds):
        X_train = [[movie_table.data[i][5], movie_table.data[i][15], movie_table.data[i][16],
                    movie_table.data[i][21], movie_table.data[i][4], movie_table.data[i][1], movie_table.data[i][2]] for i in train]
        X_test = [[movie_table.data[i][5], movie_table.data[i][15], movie_table.data[i][16],
                    movie_table.data[i][21], movie_table.data[i][4], movie_table.data[i][1], movie_table.data[i][2]] for i in test]
        #y_train = [titles[i] for i in train]
        y_test = [titles[i] for i in test]
        strat_true.extend(y_test)
        nb_predicted = nb.predict(X_test)
        strat_nb.extend(nb_predicted)
        dummy_predicted = dummy.predict(X_test)
        strat_dummy.extend(dummy_predicted)

    split_X_train, split_X_test, split_y_train, split_y_test = myevaluation.train_test_split(x_train, y_train)

    strat_tree = tree.predict(split_X_test)

    '''print(strat_nb)
    print(strat_dummy)
    print(strat_tree) '''

    nb_accuracy = myevaluation.accuracy_score(strat_true, strat_nb)
    dummy_accuracy = myevaluation.accuracy_score(strat_true, strat_dummy)
    tree_accuracy = myevaluation.accuracy_score(split_y_test, strat_tree)
    myutils.print_accuracys_2(nb_accuracy, dummy_accuracy, tree_accuracy)

    return dummy, naive, tree, random_forest
