import contextlib
import mysklearn.myutils as myutils
import mysklearn.myevaluation as myevaluation
from mysklearn.myclassifiers import MyKNeighborsClassifier, MyDummyClassifier, MyNaiveBayesClassifier, MyDecisionTreeClassifier
from mysklearn.mypytable import MyPyTable

def set_up(movie_table: MyPyTable):
    knn = MyKNeighborsClassifier()
    dummy = MyDummyClassifier()
    naive = MyNaiveBayesClassifier()
    tree = MyDecisionTreeClassifier()
    movie_table = myutils.convert_to_list(movie_table, 'genres')
    print(movie_table.column_names)
    columns_used = ["genres", "metascore", "imdbrating", "boxoffice", "runtime", "release_date"]
    for x in columns_used:
        print(movie_table.column_names.index(x), end=" ")
    print()
    titles = [str(value) for value in movie_table.get_column("title")]
    genres = movie_table.get_column("genres")
    genres_strings = []
    for val in genres:
        if len(val) == 0:
            genres_strings.append("Action")
        else:
            genres_strings.append(val[0])
    metascore = movie_table.get_column("metascore")
    imdbrating = movie_table.get_column("imdbrating")
    boxoffices = movie_table.get_column("boxoffice")
    runtimes = movie_table.get_column("runtime")
    release_dates = movie_table.get_column("release_date")

    movie = [str(value) for value in movie_table.get_column("title")]

    x_train = []
    y_train = movie
    for val in range(len(movie_table.data)):
        tmp = []
        #if genres[val] != []:
        tmp.append(genres[val])
        #else:
        #    tmp.append("Action")
        tmp.append(metascore[val])
        tmp.append(imdbrating[val])
        tmp.append(boxoffices[val])
        tmp.append(runtimes[val])
        tmp.append(release_dates[val])
        x_train.append(tmp)

    strat_train_folds, strat_test_folds = myevaluation.stratified_kfold_cross_validation(x_train, y_train, n_splits=10)
    strat_knn = []
    strat_nb = []
    strat_dummy = []
    strat_tree = []
    strat_true = []

    for train, test in zip(strat_train_folds, strat_test_folds):
        '''genres_strings[i], movie_table.data[i][15],
                    movie_table.data[i][16], movie_table.data[i][20],
                    movie_table.data[i][4], movie_table.data[i][3]'''
        X_train = [[movie_table.data[i][1:]] for i in train]
        X_test = [[movie_table.data[i][1:]] for i in test]
        y_train = [movie[i] for i in train]
        y_test = [movie[i] for i in test]
        strat_true.extend(y_test)
        knn = MyKNeighborsClassifier()
        knn.fit(X_train, y_train)
        for tests in X_test:
            strat_knn.append(knn.predict(tests)[0])
        ''' nb = MyNaiveBayesClassifier()
        nb.fit(X_train, y_train)
        nb_predicted = nb.predict(X_test)
        strat_nb.extend(nb_predicted) '''
        dummy = MyDummyClassifier()
        dummy.fit(X_train, y_train)
        dummy_predicted = dummy.predict(X_test)
        strat_dummy.extend(dummy_predicted)
        ''' tree = MyDecisionTreeClassifier()
        tree.fit(X_train, y_train)
        strat_tree.extend(tree.predict(X_test)) '''
        break
    print(X_test)
    print(tree.tree)
    print(strat_knn)

    knn_accuracy = myevaluation.accuracy_score(strat_true, strat_knn)
    nb_accuracy = myevaluation.accuracy_score(strat_true, strat_nb)
    dummy_accuracy = myevaluation.accuracy_score(strat_true, strat_dummy)
    tree_accuracy = myevaluation.accuracy_score(strat_true, strat_tree)
    myutils.print_accuracys(knn_accuracy, nb_accuracy, dummy_accuracy, tree_accuracy)

    #"Harry Potter and the Sorcerer's Stone", 'Adventure', 152.0, 20011116
    ''' with open("decision_rules.txt", "w") as output_data:
        with contextlib.redirect_stdout(output_data):
            tree.print_decision_rules(columns_used)
        output_data.close() '''

    return knn, dummy, naive, tree
