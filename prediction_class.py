"""
Programmer: Troy and James
Class: CptS 322-01, Spring 2022
Programming Final Project
"""
import random
import mysklearn.myutils as myutils
import mysklearn.myevaluation as myevaluation
from mysklearn.myclassifiers import MyKNeighborsClassifier, MyDummyClassifier, MyNaiveBayesClassifier, MyDecisionTreeClassifier, MyRandomForestClassifier
from mysklearn.mypytable import MyPyTable

def set_up(movie_table: MyPyTable):
    """Sets up and compares the Kfold and test train split

    Args:
        movie_table (MyPyTable): movie data

    Returns:
        classifiers: 4 seperate classifiers
    """
    predict_genre_stratified(movie_table)
    predict_genre_test_train(movie_table)
    return get_best(movie_table)

def predict_genre_stratified(movie_table: MyPyTable):
    """predicts and get stats around using kfold

    Args:
        movie_table (MyPyTable): _description_
    """
    print("USING STRATIFIED K FOLD")
    #create the 4 classifiers
    dummy = MyDummyClassifier()
    naive = MyNaiveBayesClassifier()
    tree = MyDecisionTreeClassifier()
    forest = MyRandomForestClassifier(100, 2, 2)
    movie_table.data = movie_table.data[:250] #shorten amount of data because this is a test
    genre_1, genre_2, genre_3 = myutils.seperate_genre(movie_table.get_column("genres")) #get genres
    genres_unique =  ['Documentary', 'Musical', 'Music', 'Short', 'Sport', 'War', 'Foreign',
     'Mystery', 'Biography', 'History', 'Horror', 'Romance', 'Talk-Show', 'TV Movie',
     'Western', 'Family', 'Comedy', 'Animation', 'Sci-Fi', 'Science Fiction',
     'Drama', 'Crime', 'Thriller', 'Fantasy', 'Adventure', 'Action', 'NA', '']
    genre_1_even, _ = myutils.even_out_lists(genre_1, genres_unique) #even out
    genre_2_even, _ = myutils.even_out_lists(genre_2, genres_unique) #even out
    genre_3_even, _ = myutils.even_out_lists(genre_3, genres_unique) #even out
    genre = movie_table.get_column("genres")  #get genres
    metascore = movie_table.get_column("metascore") #metascore
    imdbrating = movie_table.get_column("imdbrating") #imdbrating
    boxoffices = movie_table.get_column("boxoffice") #box office
    boxoffices_high_low = myutils.find_low_mid_high(boxoffices, 1000000.0, 35000000.0) #convert to string
    runtimes = movie_table.get_column("runtime") #get run time
    years = movie_table.get_column("year") #get year
    release_dates = movie_table.get_column("release_date") #ger release date
    rated = movie_table.get_column("rated") #get rated

    genres = genre_1_even

    x_train = []
    y_train = genres #set y train
    #build xtrain
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
    strat_forest = []
    strat_ensemble = []
    strat_true = []

    for train, test in zip(strat_train_folds, strat_test_folds):
        X_train = [[movie_table.data[i][15], movie_table.data[i][16], myutils.low_or_high(movie_table.data[i][21], 1000000.0, 35000000.0),
                    movie_table.data[i][4], movie_table.data[i][1], movie_table.data[i][3], movie_table.data[i][2]] for i in train]
        X_test =  [[movie_table.data[i][15], movie_table.data[i][16], myutils.low_or_high(movie_table.data[i][21], 1000000.0, 35000000.0),
                    movie_table.data[i][4], movie_table.data[i][1], movie_table.data[i][3], movie_table.data[i][2]] for i in test]
        y_train = [genres[i] for i in train]
        y_test = [genres[i] for i in test]
        strat_true.extend(y_test)
        nb = MyNaiveBayesClassifier()
        nb.fit(X_train, y_train) #fit
        strat_nb.extend(nb.predict(X_test)) #predict
        dummy = MyDummyClassifier()
        dummy.fit(X_train, y_train)#fit
        strat_dummy.extend(dummy.predict(X_test))#predict
        tree = MyDecisionTreeClassifier()
        tree.fit(X_train, y_train)#fit
        strat_tree.extend(tree.predict(X_test))#predict
        forest = MyRandomForestClassifier(200, 2, 2)
        forest.fit(X_train, y_train)#fit
        strat_forest.extend(forest.predict(X_test))#predict
        strat_ensemble = myutils.ensemble_helper(dummy, nb, tree, forest, X_test)#predict

    #compute and print accuracys
    nb_accuracy = myevaluation.accuracy_score(strat_true, strat_nb)
    dummy_accuracy = myevaluation.accuracy_score(strat_true, strat_dummy)
    tree_accuracy = myevaluation.accuracy_score(strat_true, strat_tree)
    forest_accuracy = myevaluation.accuracy_score(strat_true, strat_forest)
    ensemble_accuracy = myevaluation.accuracy_score(strat_true, strat_ensemble)
    myutils.print_accuracys_2(nb_accuracy, dummy_accuracy, tree_accuracy, forest_accuracy, ensemble_accuracy)

    nb_precision = []
    nb_recall = []
    nb_f1 = []

    dummy_precision = []
    dummy_recall = []
    dummy_f1 = []

    tree_precision = []
    tree_recall = []
    tree_f1 = []

    forest_precision = []
    forest_recall = []
    forest_f1 = []

    ensemble_precision = []
    ensemble_recall = []
    ensemble_f1 = []

    #get precision. recall,  and f1
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

        forest_precision.append(myevaluation.binary_precision_score(strat_true, strat_forest, pos_label=genre))
        forest_recall.append(myevaluation.binary_recall_score(strat_true, strat_forest, pos_label=genre))
        forest_f1.append(myevaluation.binary_f1_score(strat_true, strat_forest, pos_label=genre))

        ensemble_precision.append(myevaluation.binary_precision_score(strat_true, strat_ensemble, pos_label=genre))
        ensemble_recall.append(myevaluation.binary_recall_score(strat_true, strat_ensemble, pos_label=genre))
        ensemble_f1.append(myevaluation.binary_f1_score(strat_true, strat_ensemble, pos_label=genre))

    #compute the precision. recall, and f1
    nb_eval = [sum(nb_precision) / len(nb_precision), sum(nb_recall) / len(nb_recall), sum(nb_f1) / len(nb_f1)]
    dummy_eval = [sum(dummy_precision) / len(dummy_precision), sum(dummy_recall) / len(dummy_recall), sum(dummy_f1) / len(dummy_f1)]
    tree_eval = [sum(tree_precision) / len(tree_precision), sum(tree_recall) / len(tree_recall), sum(tree_f1) / len(tree_f1)]
    forest_eval = [sum(forest_precision) / len(forest_precision), sum(forest_recall) / len(forest_recall), sum(forest_f1) / len(tree_f1)]
    ensemble_eval = [sum(ensemble_precision) / len(ensemble_precision), sum(ensemble_recall) / len(ensemble_recall), sum(ensemble_f1) / len(tree_f1)]

    myutils.print_precision_recall_f1_2(nb_eval, dummy_eval, tree_eval, forest_eval, ensemble_eval)

def predict_genre_test_train(movie_table: MyPyTable):
    """predicts and computes stats around test train split

    Args:
        movie_table (MyPyTable): _description_
    """
    print("\n\nUSING TEST TRAIN SPLIT")
    #create the 4 classifiers
    dummy = MyDummyClassifier()
    naive = MyNaiveBayesClassifier()
    tree = MyDecisionTreeClassifier()
    forest = MyRandomForestClassifier(100, 2, 2)
    movie_table.data = movie_table.data[:250] #shorten amount of data because this is a test
    genre_1, genre_2, genre_3 = myutils.seperate_genre(movie_table.get_column("genres")) #get genres
    genres_unique =  ['Documentary', 'Musical', 'Music', 'Short', 'Sport', 'War', 'Foreign',
     'Mystery', 'Biography', 'History', 'Horror', 'Romance', 'Talk-Show', 'TV Movie',
     'Western', 'Family', 'Comedy', 'Animation', 'Sci-Fi', 'Science Fiction',
     'Drama', 'Crime', 'Thriller', 'Fantasy', 'Adventure', 'Action', 'NA', '']
    genre_1_even, _ = myutils.even_out_lists(genre_1, genres_unique) #even out
    genre_2_even, _ = myutils.even_out_lists(genre_2, genres_unique) #even out
    genre_3_even, _ = myutils.even_out_lists(genre_3, genres_unique) #even out
    genre = movie_table.get_column("genres")  #get genres
    metascore = movie_table.get_column("metascore") #metascore
    imdbrating = movie_table.get_column("imdbrating") #imdbrating
    boxoffices = movie_table.get_column("boxoffice") #box office
    boxoffices_high_low = myutils.find_low_mid_high(boxoffices, 1000000.0, 35000000.0) #convert to string
    runtimes = movie_table.get_column("runtime") #get run time
    years = movie_table.get_column("year") #get year
    release_dates = movie_table.get_column("release_date") #ger release date
    rated = movie_table.get_column("rated") #get rated

    genres = genre_1_even

    x_train = []
    y_train = genres #set y train
    #build xtrain
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

    #split test train split 33 66
    X_train, X_test, split_y_train, split_y_test = myevaluation.train_test_split(x_train, y_train, random_state=25)

    strat_nb = []
    strat_dummy = []
    strat_tree = []
    strat_forest = []
    strat_ensemble = []
    strat_true = []

    #predict and fitting
    strat_true.extend(split_y_test)
    nb = MyNaiveBayesClassifier()
    nb.fit(X_train, y_train) #fit
    strat_nb.extend(nb.predict(X_test))#predict
    dummy = MyDummyClassifier()
    dummy.fit(X_train, y_train)#fit
    strat_dummy.extend(dummy.predict(X_test))#predict
    tree = MyDecisionTreeClassifier()
    tree.fit(X_train, y_train)#fit
    strat_tree.extend(tree.predict(X_test))#predict
    forest = MyRandomForestClassifier(200, 2, 2)
    forest.fit(X_train, y_train)#fit
    strat_forest.extend(forest.predict(X_test))#predict
    strat_ensemble = myutils.ensemble_helper(dummy, nb, tree, forest, X_test)#predict

    #compute accuracys
    nb_accuracy = myevaluation.accuracy_score(strat_true, strat_nb)
    dummy_accuracy = myevaluation.accuracy_score(strat_true, strat_dummy)
    tree_accuracy = myevaluation.accuracy_score(strat_true, strat_tree)
    forest_accuracy = myevaluation.accuracy_score(strat_true, strat_forest)
    ensemble_accuracy = myevaluation.accuracy_score(strat_true, strat_ensemble)
    myutils.print_accuracys_2(nb_accuracy, dummy_accuracy, tree_accuracy, forest_accuracy, ensemble_accuracy)

    nb_precision = []
    nb_recall = []
    nb_f1 = []

    dummy_precision = []
    dummy_recall = []
    dummy_f1 = []

    tree_precision = []
    tree_recall = []
    tree_f1 = []

    forest_precision = []
    forest_recall = []
    forest_f1 = []

    ensemble_precision = []
    ensemble_recall = []
    ensemble_f1 = []

    #collect precision, recall, and f1
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

        forest_precision.append(myevaluation.binary_precision_score(strat_true, strat_forest, pos_label=genre))
        forest_recall.append(myevaluation.binary_recall_score(strat_true, strat_forest, pos_label=genre))
        forest_f1.append(myevaluation.binary_f1_score(strat_true, strat_forest, pos_label=genre))

        ensemble_precision.append(myevaluation.binary_precision_score(strat_true, strat_ensemble, pos_label=genre))
        ensemble_recall.append(myevaluation.binary_recall_score(strat_true, strat_ensemble, pos_label=genre))
        ensemble_f1.append(myevaluation.binary_f1_score(strat_true, strat_ensemble, pos_label=genre))

    #compute precision, recall, and f1
    nb_eval = [sum(nb_precision) / len(nb_precision), sum(nb_recall) / len(nb_recall), sum(nb_f1) / len(nb_f1)]
    dummy_eval = [sum(dummy_precision) / len(dummy_precision), sum(dummy_recall) / len(dummy_recall), sum(dummy_f1) / len(dummy_f1)]
    tree_eval = [sum(tree_precision) / len(tree_precision), sum(tree_recall) / len(tree_recall), sum(tree_f1) / len(tree_f1)]
    forest_eval = [sum(forest_precision) / len(forest_precision), sum(forest_recall) / len(forest_recall), sum(forest_f1) / len(tree_f1)]
    ensemble_eval = [sum(ensemble_precision) / len(ensemble_precision), sum(ensemble_recall) / len(ensemble_recall), sum(ensemble_f1) / len(tree_f1)]

    myutils.print_precision_recall_f1_2(nb_eval, dummy_eval, tree_eval, forest_eval, ensemble_eval)

def get_best(movie_table: MyPyTable):
    """gets the best of the classifiers

    Args:
        movie_table (MyPyTable): movie data

    Returns:
        dummy, naive, tree, forest: the for best classifiers
    """
    #create classifiers
    dummy = MyDummyClassifier()
    naive = MyNaiveBayesClassifier()
    tree = MyDecisionTreeClassifier()
    forest = MyRandomForestClassifier(200, 2, 2)
    genre_1, genre_2, genre_3 = myutils.seperate_genre(movie_table.get_column("genres")) #get genres
    genres_unique =  ['Documentary', 'Musical', 'Music', 'Short', 'Sport', 'War', 'Foreign',
     'Mystery', 'Biography', 'History', 'Horror', 'Romance', 'Talk-Show', 'TV Movie',
     'Western', 'Family', 'Comedy', 'Animation', 'Sci-Fi', 'Science Fiction',
     'Drama', 'Crime', 'Thriller', 'Fantasy', 'Adventure', 'Action', 'NA', '']
    genre_1_even, _ = myutils.even_out_lists(genre_1, genres_unique) #even out
    genre_2_even, _ = myutils.even_out_lists(genre_2, genres_unique) #even out
    genre_3_even, _ = myutils.even_out_lists(genre_3, genres_unique) #even out
    genre = movie_table.get_column("genres")  #get genres
    metascore = movie_table.get_column("metascore") #metascore
    imdbrating = movie_table.get_column("imdbrating") #imdbrating
    boxoffices = movie_table.get_column("boxoffice") #box office
    boxoffices_high_low = myutils.find_low_mid_high(boxoffices, 1000000.0, 35000000.0) #convert to string
    runtimes = movie_table.get_column("runtime") #get run time
    years = movie_table.get_column("year") #get year
    release_dates = movie_table.get_column("release_date") #ger release date
    rated = movie_table.get_column("rated") #get rated

    genres = genre_1_even

    x_train = []
    y_train = genres #ytrain
    #build but x train
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

    naive = MyNaiveBayesClassifier()
    naive.fit(x_train, y_train) #fit
    dummy = MyDummyClassifier()
    dummy.fit(x_train, y_train)  #fit
    tree = MyDecisionTreeClassifier()
    tree.fit(x_train, y_train) #fit
    forest = MyRandomForestClassifier(200, 2, 2)
    forest.fit(x_train, y_train) #fit

    return dummy, naive, tree, forest

