"""
Programmer: Troy Bechtold
Class: CptS 322-01, Spring 2022
Programming Assignment #5
3/14/22

Description: my elvation file for classifiers
"""
import math
import numpy as np
from mysklearn import myutils

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    if random_state is not None:
        np.random.seed(random_state) #set seed
    if shuffle is True:
        myutils.randomize_in_place(X,y) #ranomdize in place
    num_instances = len(X) #get length
    if isinstance(test_size, float):
        test_size = math.ceil(num_instances * test_size) #create test size
    split_index = num_instances - test_size #find split index
    X_train = X[:split_index] #make xtrain
    X_test = X[split_index:] #make test
    y_train = y[:split_index] #make ytrain
    y_test = y[split_index:] #make test
    return X_train, X_test, y_train, y_test #return lists

def kfold_cross_validation(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    if random_state is not None:
        np.random.seed(random_state) #set random seed
    n = len(X) #get length
    X_train_folds = []
    X_test_folds = []
    sample_size = []
    folds = n % n_splits #get folds size
    for i in range(n_splits): #for i to nsplits
        if i >= folds:
            sample_size.append(n // n_splits) #append size
        else:
            sample_size.append((n // n_splits) + 1) #append siz
    if shuffle is True: #shuffle
        myutils.randomize_in_place(X) #randomize in space
        for i in range(n_splits): #for i to nsplits
            indices = list(range(len(X))) #get list full of length
            range_size = (sample_size[i]) #get size
            start_index = sum(sample_size[n] for n in range(i)) #get sum of sample
            test_fold = list(range(start_index, start_index + range_size)) #get list of range
            X_test_folds.append(test_fold) #add the fold
            del indices[start_index: start_index + range_size] #delete incdices
            X_train_folds.append(indices) #set indeices to x folds
            myutils.randomize_in_place(X_train_folds, X_test_folds) #randamize x train and x test
        if random_state is not None: #if state is not none
            for _ in range(random_state): #randomize random state amount of times
                myutils.randomize_in_place(X_train_folds, X_test_folds) #randomize in place
    else:
        for i in range(n_splits):
            indices = list(range(len(X))) #get indicies
            range_size = sample_size[i] #get range size
            start_index = sum(sample_size[n] for n in range(i)) #get sum of sample size
            test_fold = list(range(start_index, start_index + range_size)) #get list of ranges
            X_test_folds.append(test_fold) #append test fold
            del indices[start_index: start_index + range_size] #delete indices
            X_train_folds.append(indices) #add indices
    return X_train_folds, X_test_folds #return x_train and x_tests

def stratified_kfold_cross_validation(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    if random_state is not None:
        np.random.seed(random_state) #set seed
    if shuffle is True:
        myutils.randomize_in_place(X,y) #shuff
    # creates arrays for folds
    total_folds = [[] for _ in range(n_splits)]
    X_train_folds = [[] for _ in range(n_splits)]
    X_test_folds = [[] for _ in range(n_splits)]
    grouped_folds = myutils.group(X, y, n_splits)
    index = 0
    for row in grouped_folds: #for each row
        for col in row: #for each col
            total_folds[index].append(col) #add the col
            index = (index + 1) % n_splits #get index
    index = 0
    for i in range(n_splits): #for i to nsplits
        for j, row in enumerate(total_folds):
            if i != j:
                for col in row: #for the col
                    X_train_folds[index].append(col) #add col
            else:
                X_test_folds[index] = row #set to row
        index += 1
    return X_train_folds, X_test_folds #return folds

def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results
    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
    """
    if random_state is not None:
        np.random.seed(random_state) #set seed
    if n_samples is None:
        n_samples = len(X[0]) + 1 #set len
    X_sample = []
    X_out_of_bag = []
    y_sample = []
    y_out_of_bag = []
    for _ in range(n_samples):
        random_value = np.random.randint(0,len(X)) #gets random value
        value = X[random_value]
        X_sample.append(value)#adds random value to x
        if y is not None:
            y_sample.append(y[random_value])#adds random values
    X_out_of_bag = [x for x in X if x not in X_sample]
    if y is not None:
        #creates y out of bag
        y_out_of_bag = [y_data for y_data in y if y_data not in y_sample]
    else:
        y_out_of_bag = None #set to noe
        y_sample = None #set to none
    return X_sample, X_out_of_bag, y_sample, y_out_of_bag
    #returns samples and out of bag

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix = [[0 for x in labels] for y in labels]# labels for matrix

    a_map = {key: i for i, key in enumerate(labels)}# creates map
    for pre, true in zip(y_pred, y_true):# create matrix
        matrix[a_map[true]][a_map[pre]] += 1
    return matrix# return matrix

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    accuracy = 0.0
    for true_y, predicted_y in zip(y_true, y_pred): #traverse through true and pre
        if true_y == predicted_y:
            accuracy += 1 #add up accuracy
    if normalize is True:
        accuracy = accuracy / len(y_true) #normalizes
    return accuracy #returns

def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    if labels is None:
        labels = list(set(y_true))
    if pos_label is None:
        pos_label = labels[0]
    true_pos = 0.0
    fp_and_tp = 0.0
    false_pos = 0.0
    for true_y, predicted_y in zip(y_true, y_pred): #traverse through true and pre
        if true_y in labels and predicted_y in labels:
            if true_y == predicted_y and predicted_y == pos_label:
                true_pos += 1 #add up accuracy
            if true_y != predicted_y and predicted_y == pos_label:
                false_pos += 1 #add up accuracy
    fp_and_tp = true_pos + false_pos
    if true_pos == 0 and fp_and_tp == 0:
        precision = 0.0
    else:
        precision = true_pos/fp_and_tp
    return precision

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    if labels is None:
        labels = list(set(y_true))
    if pos_label is None:
        pos_label = labels[0]
    true_pos = 0.0
    fn_and_tp = 0.0
    false_neg = 0.0
    for true_y, predicted_y in zip(y_true, y_pred): #traverse through true and pre
        if true_y in labels and predicted_y in labels:
            if true_y == predicted_y and predicted_y == pos_label:
                true_pos += 1 #add up accuracy
            if predicted_y not in (true_y, pos_label):
                false_neg += 1 #add up accuracy
    fn_and_tp = true_pos + false_neg
    if true_pos == 0 and fn_and_tp == 0:
        recall = 0.0
    else:
        recall = true_pos/fn_and_tp
    return recall

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    if labels is None:
        labels = list(set(y_true)) #get labels
    if pos_label is None:
        pos_label = labels[0] #set to first
    #get precision
    precision = binary_precision_score(y_true, y_pred, labels, pos_label)
    #get recall
    recall = binary_recall_score(y_true, y_pred, labels, pos_label)
    if precision + recall == 0: #if zero then f_one is zero
        f_one = 0.0
    else: #else use above formula
        f_one = 2 * (precision * recall) / (precision + recall)
    return f_one
