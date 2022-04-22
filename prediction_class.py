from sklearn.utils import shuffle
import mysklearn.myutils as myutils
import mysklearn.myevaluation as myevaluation
from mysklearn.myclassifiers import MyKNeighborsClassifier, MyDummyClassifier, MyNaiveBayesClassifier, MyDecisionTreeClassifier
from mysklearn.mypytable import MyPyTable

def set_up(movie_table: MyPyTable, title=True):
    if title is True:
        return predict_titles(movie_table)
    else:
       return  predict_genre(movie_table)

def predict_titles(movie_table: MyPyTable):
    dummy = MyDummyClassifier()
    naive = MyNaiveBayesClassifier()
    tree = MyDecisionTreeClassifier()
    #print(movie_table.column_names)
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
    boxoffices_high_low = myutils.find_low_mid_high(boxoffices, 1000000.0, 35000000.0)
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

    nb_precision = []
    nb_recall = []
    nb_f1 = []

    dummy_precision = []
    dummy_recall = []
    dummy_f1 = []

    tree_precision = []
    tree_recall = []
    tree_f1 = []
    for title in titles:
        nb_precision.append(myevaluation.binary_precision_score(strat_true, strat_nb, pos_label=title))
        nb_recall.append(myevaluation.binary_recall_score(strat_true, strat_nb, pos_label=title))
        nb_f1.append(myevaluation.binary_f1_score(strat_true, strat_nb, pos_label=title))

        dummy_precision.append(myevaluation.binary_precision_score(strat_true, strat_dummy, pos_label=title))
        dummy_recall.append(myevaluation.binary_recall_score(strat_true, strat_dummy, pos_label=title))
        dummy_f1.append(myevaluation.binary_f1_score(strat_true, strat_dummy, pos_label=title))

        tree_precision.append(myevaluation.binary_precision_score(split_y_test, strat_tree, pos_label=title))
        tree_recall.append(myevaluation.binary_recall_score(split_y_test, strat_tree, pos_label=title))
        tree_f1.append(myevaluation.binary_f1_score(split_y_test, strat_tree, pos_label=title))

    nb_eval = [sum(nb_precision) / len(nb_precision), sum(nb_recall) / len(nb_recall), sum(nb_f1) / len(nb_f1)]
    dummy_eval = [sum(dummy_precision) / len(dummy_precision), sum(dummy_recall) / len(dummy_recall), sum(dummy_f1) / len(dummy_f1)]
    tree_eval = [sum(tree_precision) / len(tree_precision), sum(tree_recall) / len(tree_recall), sum(tree_f1) / len(tree_f1)]

    myutils.print_precision_recall_f1_2(nb_eval, dummy_eval, tree_eval)


    return dummy, naive, tree, random_forest


def predict_genre(movie_table: MyPyTable):
    dummy = MyDummyClassifier()
    naive = MyNaiveBayesClassifier()
    tree = MyDecisionTreeClassifier()
    random_forest = None
    #print(movie_table.column_names)
    movie_table.data = movie_table.data
    ''' columns_used = ["metascore", "imdbrating", "boxoffice", "runtime", "year", "rated"]
    for x in columns_used:
        print(movie_table.column_names.index(x), end=" ")
    print() '''
    #titles = [str(value) for value in movie_table.get_column("title")]
    genre_1, genre_2, genre_3 = myutils.seperate_genre(movie_table.get_column("genres"))
    genres_unique =  ['Documentary', 'Musical', 'Music', 'Short', 'Sport', 'War', 'Foreign',
     'Mystery', 'Biography', 'History', 'Horror', 'Romance', 'Talk-Show', 'TV Movie',
     'Western', 'Family', 'Comedy', 'Animation', 'Sci-Fi', 'Science Fiction',
     'Drama', 'Crime', 'Thriller', 'Fantasy', 'Adventure', 'Action', 'NA', '']
    genre_1, _ = myutils.even_out_lists(genre_1, genres_unique)
    genre_2, _ = myutils.even_out_lists(genre_2, genres_unique)
    genre_3, _ = myutils.even_out_lists(genre_3, genres_unique)
    metascore = movie_table.get_column("metascore")
    imdbrating = movie_table.get_column("imdbrating")
    boxoffices = movie_table.get_column("boxoffice")
    boxoffices_high_low = myutils.find_low_mid_high(boxoffices, 1000000.0, 35000000.0)
    runtimes = movie_table.get_column("runtime")
    years = movie_table.get_column("year")
    release_dates = movie_table.get_column("release_date")
    rated = movie_table.get_column("rated")

    for genres, count in zip([genre_1, genre_2, genre_3], ["first", "second", "third"]):
        print(f'\nFor {count} genre category')
        x_train = []
        y_train = genres
        for val in range(len(genres)):
            tmp = []
            tmp.append(metascore[val])
            tmp.append(imdbrating[val])
            tmp.append(boxoffices_high_low[val])
            tmp.append(runtimes[val])
            tmp.append(years[val])
            tmp.append(release_dates[val])
            tmp.append(rated[val])
            x_train.append(tmp)

        strat_train_folds, strat_test_folds = myevaluation.stratified_kfold_cross_validation(x_train, y_train, n_splits=2)
        strat_nb = []
        strat_dummy = []
        strat_tree = []
        strat_true = []

        for train, test in zip(strat_train_folds, strat_test_folds):
            X_train = [[movie_table.data[i][15], movie_table.data[i][16], movie_table.data[i][21],
                        movie_table.data[i][4], movie_table.data[i][1], movie_table.data[i][2]] for i in train]
            X_test = [[movie_table.data[i][15], movie_table.data[i][16], movie_table.data[i][21],
                    movie_table.data[i][4], movie_table.data[i][1], movie_table.data[i][2]] for i in test]
            y_train = [genres[i] for i in train]
            y_test = [genres[i] for i in test]
            strat_true.extend(y_test)
            nb = MyNaiveBayesClassifier()
            nb.fit(X_test, y_train)
            strat_nb.extend(nb.predict(X_test))
            dummy = MyDummyClassifier()
            dummy.fit(X_test, y_train)
            strat_dummy.extend(dummy.predict(X_test))
            tree = MyDecisionTreeClassifier()
            tree.fit(X_test, y_train)
            strat_tree.extend(tree.predict(X_test))

        ''' print(strat_nb)
        print(strat_dummy)
        print(strat_tree) '''

        nb_accuracy = myevaluation.accuracy_score(strat_true, strat_nb)
        dummy_accuracy = myevaluation.accuracy_score(strat_true, strat_dummy)
        tree_accuracy = myevaluation.accuracy_score(strat_true, strat_tree)
        myutils.print_accuracys_2(nb_accuracy, dummy_accuracy, tree_accuracy)

        nb_precision = []
        nb_recall = []
        nb_f1 = []

        dummy_precision = []
        dummy_recall = []
        dummy_f1 = []

        tree_precision = []
        tree_recall = []
        tree_f1 = []
        for genre in genres:
            nb_precision.append(myevaluation.binary_precision_score(strat_true, strat_nb, pos_label=genre))
            nb_recall.append(myevaluation.binary_recall_score(strat_true, strat_nb, pos_label=genre))
            nb_f1.append(myevaluation.binary_f1_score(strat_true, strat_nb, pos_label=genre))

            dummy_precision.append(myevaluation.binary_precision_score(strat_true, strat_dummy, pos_label=genre))
            dummy_recall.append(myevaluation.binary_recall_score(strat_true, strat_dummy, pos_label=genre))
            dummy_f1.append(myevaluation.binary_f1_score(strat_true, strat_dummy, pos_label=genre))

            tree_precision.append(myevaluation.binary_precision_score(strat_true, strat_tree, pos_label=genre))
            tree_recall.append(myevaluation.binary_recall_score(strat_true, strat_tree, pos_label=genre))
            tree_f1.append(myevaluation.binary_f1_score(strat_true, strat_tree, pos_label=genre))

        nb_eval = [sum(nb_precision) / len(nb_precision), sum(nb_recall) / len(nb_recall), sum(nb_f1) / len(nb_f1)]
        dummy_eval = [sum(dummy_precision) / len(dummy_precision), sum(dummy_recall) / len(dummy_recall), sum(dummy_f1) / len(dummy_f1)]
        tree_eval = [sum(tree_precision) / len(tree_precision), sum(tree_recall) / len(tree_recall), sum(tree_f1) / len(tree_f1)]

        myutils.print_precision_recall_f1_2(nb_eval, dummy_eval, tree_eval)

    return dummy, naive, tree, random_forest
