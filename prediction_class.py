import mysklearn.myutils as myutils
import mysklearn.myevaluation as myevaluation
from mysklearn.myclassifiers import MyKNeighborsClassifier, MyDummyClassifier, MyNaiveBayesClassifier, MyDecisionTreeClassifier
from mysklearn.mypytable import MyPyTable

def set_up(movie_table: MyPyTable):
    knn = MyKNeighborsClassifier()
    dummy = MyDummyClassifier()
    naive = MyNaiveBayesClassifier()
    tree = MyDecisionTreeClassifier()
    columns_used = ["title", "genres", "ratings", "metascore", "imdbrating", "boxoffice", "runtime", "year", "release_date"]
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
        tmp.append(titles[val])
        tmp.append(genres[val])
        tmp.append(metascore[val])
        tmp.append(imdbrating[val])
        tmp.append(boxoffices[val])
        tmp.append(runtimes[val])
        tmp.append(years[val])
        tmp.append(release_dates[val])
        x_train.append(tmp)

    strat_train_folds, strat_test_folds = myevaluation.stratified_kfold_cross_validation(x_train, y_train, n_splits=10)
    strat_knn = []
    strat_nb = []
    strat_dummy = []
    strat_tree = []
    strat_true = []

    print(len(tmp))

    for train, test in zip(strat_train_folds, strat_test_folds):
        X_train = [[str(movie_table.data[i][0]), movie_table.data[i][5], movie_table.data[i][16],
                    movie_table.data[i][21], movie_table.data[i][4], movie_table.data[i][1],
                    movie_table.data[i][3]] for i in train]
        X_test = [[str(movie_table.data[i][0]), movie_table.data[i][5], movie_table.data[i][16],
                   movie_table.data[i][21], movie_table.data[i][4], movie_table.data[i][1],
                   movie_table.data[i][3]] for i in test]
        y_train = [movie[i] for i in train]
        y_test = [movie[i] for i in test]
        strat_true.extend(y_test)
        knn = MyKNeighborsClassifier()
        knn.fit(X_train, y_train)
        for tests in X_test[:25]:
            #print(tests)
            print(knn.predict(tests))
            for test in tests:
                print(knn.predict(test)[0])
                #strat_knn.append(knn.predict(test)[0])
        '''nb = MyNaiveBayesClassifier()
        nb.fit(X_train, y_train)
        strat_nb.extend(nb.predict(X_test))
        dummy = MyDummyClassifier()
        dummy.fit(X_train, y_train)
        strat_dummy.extend(dummy.predict(X_test))
        tree = MyDecisionTreeClassifier()
        tree.fit(X_train_2, y_train)
        strat_tree.extend(tree.predict(X_test_2))'''

    return knn, dummy, naive, tree
