from os import pread
import mysklearn.myutils as myutils
import mysklearn.myevaluation as myevaluation
from mysklearn.myclassifiers import MyKNeighborsClassifier, MyDummyClassifier, MyNaiveBayesClassifier, MyDecisionTreeClassifier
from mysklearn.mypytable import MyPyTable

def set_up(movie_table: MyPyTable):
    knn = MyKNeighborsClassifier()
    dummy = MyDummyClassifier()
    naive = MyNaiveBayesClassifier()
    tree = MyDecisionTreeClassifier()
    #columns_used = ["title", "genres", "ratings", "metascore", "imdbrating", "boxoffice", "runtime", "year", "release_date"]
    columns_used = ["genres", "runtime","release_date"]
    for x in columns_used:
        print(movie_table.column_names.index(x), end=" ")
    print()
    titles = [str(value) for value in movie_table.get_column("title")]
    genres = movie_table.get_column("genres")
    metascore = movie_table.get_column("metascore")
    imdbrating =movie_table.get_column("imdbrating")
    boxoffices = movie_table.get_column("boxoffice")
    runtimes = movie_table.get_column("runtime")
    years = movie_table.get_column("year")
    release_dates = movie_table.get_column("release_date")

    movie = [str(value) for value in movie_table.get_column("title")]

    x_train = []
    y_train = movie
    for val in range(len(movie_table.data)):
        tmp = []
        try:
            tmp.append(genres[val][0])
        except IndexError:
            tmp.append("NA")
        tmp.append(runtimes[val])
        tmp.append(release_dates[val])
        x_train.append(tmp)

    strat_train_folds, strat_test_folds = myevaluation.stratified_kfold_cross_validation(x_train, y_train, n_splits=10)
    strat_knn = []
    strat_nb = []
    strat_dummy = []
    strat_tree = []
    strat_true = []

    for train, test in zip(strat_train_folds[:25], strat_test_folds[:25]):
        ''' X_train = [[str(movie_table.data[i][0]), movie_table.data[i][5], movie_table.data[i][16],
                    movie_table.data[i][21], movie_table.data[i][4], movie_table.data[i][1],
                    movie_table.data[i][3]] for i in train] '''
        X_train = [[str(movie_table.data[i][0]), movie_table.data[i][5], movie_table.data[i][4], movie_table.data[i][3]] for i in train]
        X_test = [[str(movie_table.data[i][0]), movie_table.data[i][5], movie_table.data[i][4], movie_table.data[i][3]] for i in test]
        y_train = [movie[i] for i in train]
        y_test = [movie[i] for i in test]
        strat_true.extend(y_test)
        knn = MyKNeighborsClassifier()
        knn.fit(X_train, y_train)
        for tests in X_test:
            strat_knn.append(knn.predict(tests)[0])
        nb = MyNaiveBayesClassifier()
        nb.fit(X_train, y_train)
        nb_predicted = nb.predict(X_test)
        strat_nb.extend(nb_predicted)
        dummy = MyDummyClassifier()
        dummy.fit(X_train, y_train)
        dummy_predicted = dummy.predict(X_test)
        strat_dummy.extend(dummy_predicted)
        tree = MyDecisionTreeClassifier()
        tree.fit(X_train, y_train)
        strat_tree.extend(tree.predict(X_test))


    knn_accuracy = myevaluation.accuracy_score(strat_true, strat_knn)
    nb_accuracy = myevaluation.accuracy_score(strat_true, strat_nb)
    dummy_accuracy = myevaluation.accuracy_score(strat_true, strat_dummy)
    tree_accuracy = myevaluation.accuracy_score(strat_true, strat_tree)
    myutils.print_accuracys(knn_accuracy, nb_accuracy, dummy_accuracy, tree_accuracy)

    return knn, dummy, naive, tree
