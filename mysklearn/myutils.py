"""
Programmer: Troy Bechtold
Class: CptS 322-01, Spring 2022
Programming Assignment #4 and #5 and #6 and #7

Description: utils for classifers and pa4 and pa5 and pa6 notebook
"""
import random
import math
import ast
import numpy as np
from datetime import datetime
from tabulate import tabulate

def compute_euclidean_distance(v1, v2):
    """ Generates the Euclidean Distance

        Args:
            v1: tuple of floats
            v2: tuple of floats

        Returns:
            The distance betweeen the points
    """
    if len(v1) == len(v2):
        #gets distance
        dist = (sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))])) ** (1/2)
    return dist

def compute_slope(col1_data, col2_data):
    """ Computes the slope and intercept

        Args:
            col1_data: list of x vals
            col2_data: list of y vals

        Returns:
            Slope and intercept
    """
    mean1 = sum(col1_data) / len(col1_data)
    mean2 = sum(col2_data) / len(col2_data)
    #get m
    m = sum([(col1_data[i] - mean1) * (col2_data[i] - mean2)
             for i in range(len(col1_data))]) / sum([(col1_data[i] - mean1) ** 2
                for i in range(len(col1_data))])
    b = mean2 - m * mean1 #find b
    return m, b

def itemgetter(*items):
    """
        Function for itemgetter that is usually used with the operator library
    """
    if len(items) == 1:
        item = items[0]
        def get_object(obj): #gets object
            return obj[item] #retrns it
    else:
        def get_object(obj):
            return tuple(obj[item] for item in items) #returns intem
    return get_object

def discretizer(y):
    """ discretize function for simple linear regression

        Args:
            y_train: list of lists - tuples

        Returns:
            a high or low value

    """
    if y >= 100:
        return "High"
    return "Low" #if y < 100:

def discretizer_two(y):
    """ discretize function for simple linear regression

        Args:
            y_train: list of lists - tuples

        Returns:
            a high or low value
    """
    if y >= 100:
        return "High"
    if 50 <= y < 100:
        return "Mid"
    return "Low"#if y < 50:


def normalize(x_train, x_test):
    """ Normalizes the set of data

    Args:
        x_train: list of lists - tuples
        y_train: list of lists - tuples

    Returns:
        Normalized data set

    """
    the_min_x1 = x_train[0][0] #get max
    the_min_x2 = x_train[0][1] #get min
    for row in x_train:
        if row[0] < the_min_x1:
            the_min_x1 = row[0] #get min
        if row[1] < the_min_x2:
            the_min_x2 = row[1] #get max

    for row in x_train:
        row[0] = (row[0] - the_min_x1) #find difference between min
        row[1] = (row[1] - the_min_x2) #find difference between max

    the_max_x1 = x_train[0][0] #get max
    the_max_x2 = x_train[0][1] #get min

    for row in x_train:
        if row[0] > the_max_x1:
            the_max_x1 = row[0] #get max
        if row[1] > the_max_x2:
            the_max_x2 = row[1] #get min

    for row in x_train:
        row[0] /= the_max_x1 #divide
        row[1] /= the_max_x2 #divide

    x_test[0][0] = (x_test[0][0] - the_min_x1) / the_max_x1
    x_test[0][1] = (x_test[0][1] - the_min_x2) / the_max_x2

    return x_train, x_test #retrun x train and test

def generate_random_instances(n, table):
    '''
        gets random instances from the table

        Args:
            n: number of instances
            table: mypytable

        Returns:
            radnom set of intances
    '''
    random.seed(0)
    table_len = len(table.data) #get table length
    random_instances = []
    for _ in range(n):
        index = random.randint(0, table_len - 1) #get random index
        random_instances.append(table.data[index]) #get data
    return random_instances

def get_vals(table, column_index, is_two_dim):
    '''
        gets vals from the table

        Args:
            column index: index for the column
            table: mypytable
            istwodim: if table is two d

        Returns:
            vals from column
    '''
    vals = []
    final_vals = []
    for row in table:
        vals.append(row[column_index]) #gets val
    if is_two_dim is True:
        final_vals = [[val] for val in vals] #put in a 2d list
        return final_vals
    return vals

def replace_na(table, cols):
    """
        Try to convert each na value in the table to a int.
    """
    for col in cols:
        result = []
        for row in range(len(table.data)): #loop through all rows
            column = table.column_names.index(col)
            if isinstance(table.data[row][column], str):
                try:
                    float_val = float(''.join(i for i in table.data[row][column] if i.isdigit()))
                    result.append(float_val)
                    table.data[row][column] = float_val
                except:
                    table.data[row][column] = float(round(sum(result)/len(result), 1))
            else:
                result.append(table.data[row][column])
    return table

def convert_to_int(values):
    """Try to convert each value in the table to a int type (int).

    Notes:
        Leave values as is that cannot be converted to int.
    """
    result = []
    for val in values:
        if isinstance(val, str):
            try:
                result.append(int(''.join(i for i in val if i.isdigit())))
            except:
                result.append(max(set(result), key=result.count))
        else:
            result.append(val)
    return result

def convert_to_list_to_int(table, col):
    """Try to convert each value in the table to a list type (list).

    Notes:
        Leave values as is that cannot be converted to list.
    """
    result = []
    for row in range(len(table.data)): #loop through all rows
        column = table.column_names.index(col)
        if isinstance(table.data[row][column], str):
            try:
                int_val = (float(''.join(i for i in table.data[row][column] if i.isdigit())))
                result.append(int_val)
                table.data[row][column] = int_val
            except:
                table.data[row][column] = max(set(result), key=result.count)
        else:
            result.append(table.data[row][column])
    return table

def to_integer(dt_time):
    """Helper function

    Args:
        dt_time (datetime): time

    Returns:
        int: datetime to int
    """
    return 10000*dt_time.year + 100*dt_time.month + dt_time.day

def convert_to_list_to_date(table, col):
    """Try to convert each value in the table to a list type (list).

    Notes:
        Leave values as is that cannot be converted to list.
    """
    result = []
    for row in range(len(table.data)): #loop through all rows
        column = table.column_names.index(col)
        if isinstance(table.data[row][column], str) and len(table.data[row][column]) > 3:
            try:
                table.data[row][column] = to_integer(datetime.strptime(table.data[row][column],"%Y-%m-%d"))
                result.append(table.data[row][column])
            except:
                table.data[row][column] = to_integer(datetime.strptime(table.data[row][column],"%d %b %Y"))
                result.append(table.data[row][column])
        else:
            table.data[row][column] = max(set(result), key=result.count)
    return table

def convert_to_list(table, col):
    """Try to convert each value in the table to a list type (list).

    Notes:
        Leave values as is that cannot be converted to list.
    """
    for row in range(len(table.data)): #loop through all rows
        column = table.column_names.index(col)
        if not(isinstance(table.data[row][column], list)):
            try:
                list_value = ast.literal_eval(table.data[row][column]) #try to convert the type
                table.data[row][column] = list_value #saves to data
            except ValueError:
                try:
                    list_value = table.data[row][column].strip().split(', ') #try to convert the type
                    table.data[row][column] = list_value
                except ValueError:
                    pass
                #print(table.data[row], " could not be converted to a numeric type")
                pass #does error but continue on processing
            except IndexError:
                pass
    return table

def convert_to_list_and_get_first(table, col):
    """Try to convert each value in the table to a list type (list).

    Notes:
        Leave values as is that cannot be converted to list.
    """
    for row in range(len(table.data)): #loop through all rows
        column = table.column_names.index(col)
        if not(isinstance(table.data[row][column], list)):
            try:
                list_value = ast.literal_eval(table.data[row][column]) #try to convert the type
                table.data[row][column] = list_value[0] #saves to data
            except ValueError:
                try:
                    list_value = table.data[row][column].strip().split(', ') #try to convert the type
                    table.data[row][column] = list_value[0]
                except ValueError:
                    pass
                #print(table.data[row], " could not be converted to a numeric type")
                pass #does error but continue on processing
            except IndexError:
                pass
    return table

def get_mpg_rating(mpg_list):
    '''
        gets mpg rating

        Args:
            mpg_list: mypytable column

        Returns:
            ratings for the mpg column
    '''
    mpg_ratings = []
    for row in mpg_list:
        #compares mpg in a range and set a rating
        if row <= 14.0:
            mpg_ratings.append(1)
        elif row == 14.0:
            mpg_ratings.append(2)
        elif 14.0 < row <= 16.0:
            mpg_ratings.append(3)
        elif 16.0 < row <= 19.0:
            mpg_ratings.append(4)
        elif 19.0 < row <= 23.0:
            mpg_ratings.append(5)
        elif 23.0 < row <= 26.0:
            mpg_ratings.append(6)
        elif 26.0 < row <= 30.0:
            mpg_ratings.append(7)
        elif 30.0 < row <= 36.0:
            mpg_ratings.append(8)
        elif 36.0 < row <= 44.0:
            mpg_ratings.append(9)
        elif row > 44.0:
            mpg_ratings.append(10)
    return mpg_ratings

def get_mpg_rating2(mpg_list):
    '''
        gets mpg rating

        Args:
            mpg_list: mypytable

        Returns:
            ratings for the mpg column
    '''
    mpg_ratings = []
    for row in mpg_list:
        for col in row:
            #compares mpg in a range and set a rating
            if col <= 14.0:
                mpg_ratings.append(1)
            elif col == 14.0:
                mpg_ratings.append(2)
            elif 14.0 < col <= 16.0:
                mpg_ratings.append(3)
            elif 16.0 < col <= 19.0:
                mpg_ratings.append(4)
            elif 19.0 < col <= 23.0:
                mpg_ratings.append(5)
            elif 23.0 < col <= 26.0:
                mpg_ratings.append(6)
            elif 26.0 < col <= 30.0:
                mpg_ratings.append(7)
            elif 30.0 < col <= 36.0:
                mpg_ratings.append(8)
            elif 36.0 < col <= 44.0:
                mpg_ratings.append(9)
            elif col > 44.0:
                mpg_ratings.append(10)
    return mpg_ratings


def print_instance(instance_title, random_vals, predicted, actual):
    '''
        prints intances and class acutal and accuracy from prediction

        Args:
            instance_title: title for the print out
            random_vals: random values
            predicted: what was predicted
            actual: acutal val
    '''
    print("===========================================")
    print(instance_title)
    print("===========================================")
    count = 0
    for index, row in enumerate(random_vals):
        print("instance:", row)
        print("class:", predicted[index], " actual:", actual[index])
        if actual[index] == predicted[index]:
            count += 1
    print("accuracy:", count/len(random_vals))

def randomize_in_place(alist, parallel_list=None):
    """randomized the list even if parrallel

    Args:
        alist (list): list to shuffle
        parallel_list (list, optional): 2nd list equal to alist in lenght.
        Defaults to None.
    """
    for i, _ in enumerate(alist):
        # generate a random index to swap this value at i with
        rand_index = np.random.randint(0, len(alist)) # rand int in [0, len(alist))
        # do the swap
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] =\
                parallel_list[rand_index], parallel_list[i]

def group(X, y, n_splits):
    """ Helper function for stratified_kfold_cross_validation

    Args:
        X: list of lists
        y_train: list
        n_splits: float

    Returns:
        grouped_folds: a list of lists with the folds grouped
    """
    if X is None:
        return []
    y_labels = []
    count_list = []
    for label in y:
        if label not in y_labels: #label not in y label
            y_labels.append(label) #add to ylabe
            count_list.append(1) #add 1
        elif label in y_labels:
            index = y_labels.index(label) #get index
            count_list[index] += 1 #add 1
    x_train = []
    updated_y_labels = []
    for y_val in y:
        if y_val not in updated_y_labels: #not in updated y_label
            updated_y_labels.append(y_val)
            for i, _ in enumerate(y_labels): #traverse through
                for j ,_ in enumerate(y): #traver
                    if y[j] == y_labels[i]: #if y in ylables
                        x_train.append(j) #add to xtrain
        break
    grouped_folds = [[] for _ in range(n_splits)] #group
    element_count = 0
    row_count = -1
    for i in range(len(x_train)):
        if element_count % ((len(x_train) + 1) / 2) == 0:
            row_count += 1 #add 1
        grouped_folds[row_count].append(x_train[element_count]) #add to group
        element_count += 1 #add 1
    return grouped_folds #return folds

def get_column(table, col_identifier, include_missing_values=True):
    """Extracts a column from the table data as a list.

    Args:
        col_identifier(str or int): string for a column name or int
            for a column index
        include_missing_values(bool): True if missing values ("NA")
            should be included in the column, False otherwise.

    Returns:
        tuple of int: rows, cols in the table

    Notes:
        Raise ValueError on invalid col_identifier
    """
    col_index = table.column_names.index(col_identifier)
    col = []
    if include_missing_values is False:
        for row in table.data:
            if row[col_index] != "NA":
                col.append(row[col_index])
    else:
        for row in table.data:
            col.append(row[col_index])
    return col

def convert_to_2d(table, column_name):
    """creates a 2d table

    Args:
        table (mypytable): my table
        column_name (string): name of column

    Returns:
        list: 2d list
    """
    index = table.column_names.index(column_name)
    arr = []
    for val in table.data:
        curr_row = []
        curr_row.append(val[index])
        arr.append(curr_row)
    return arr

def print_step1(k, ratio, acc, err, acc2, err2):
    """prints the acurracy and other information

    Args:
        k (string): k value
        ratio (string): ratio
        acc (string): accuracy
        err (string): error rate
        acc2 (string): accuracy
        err2 (string): error rate
    """
    print("===========================================")
    print("Step 1: Predictive Accuracy")
    print("===========================================")
    print("Subsample (k=" + k + ", " + ratio + " Train/Test)")
    print("k Nearest Neighbors Classifier: accuracy = " + acc + ", error rate = " + err)
    print("Dummy Classifier: accuracy = " + acc2 + ", error rate = " + err2)

def print_matrix(knn_matrix, dummy_matrix):
    """Prints out the matrix created by the confusion matrix

    Args:
        lin_matrix (list): matrix for knn
        knn_matrix (list): matrix for dummy
    """
    print("=" * 80)
    print("Step 5: Confusion Matrices")
    print("=" * 80)
    print("kNN Classifer (Stratified 10-Fold Cross Validation Results):")
    print(tabulate(knn_matrix))
    print("\nDummy Classifer (Stratified 10-Fold Cross Validation Results):")
    print(tabulate(dummy_matrix))

def get_mpg_class(mpg):
    """Calcuates mpg based of given mpg
    Args:
        mpg (int): mpg

    Returns:
        int: class label
    """
    value = 0
    if mpg >= 45:
        value = 10
    elif 37 <= mpg < 45:
        value = 9
    elif 31 <= mpg < 37:
        value = 8
    elif 27 <= mpg < 31:
        value = 7
    elif 24 <= mpg < 27:
        value = 6
    elif 20 <= mpg < 24:
        value = 5
    elif 17 <= mpg < 20:
        value = 4
    elif 15 <= mpg < 17:
        value = 3
    elif 14 <= mpg < 15:
        value = 2
    else:
        value = 1
    return value

def compute_euclidean_distance2(v1, v2):
    """computes euclideans distances, but with str

    Args:
        v1 (list): a list of strings
        v2 (list): a list of strings

    Returns:
        int: distance between points
    """
    dist = -1
    if len(v1) == len(v2):
        dist = -1
        if v1 == v2:
            dist =  0
        else:
            dist = 1
    else:
        dist = 1
    return dist

def print_accuracys(knn_accuracy, nb_accuracy, dummy_accuracy, tree_accuracy = None):
    """print Accuracys

    Args:
        knn_accuracy (float): Accuracy to KNN
        nb_accuracy (float): Accuracy to NB
        dummy_accuracy (float): Accuracy to DUMMY
    """
    print()
    print("=" * 80)
    print("Step 2: Accuracy and error rate")
    print("=" * 80)
    print("Stratified 10-fold Cross Validation")
    print("KNN: accuracy = ", round(knn_accuracy, 2),
          ", error rate = ", round(1-knn_accuracy, 2), sep='')
    print("Naive Bayes: accuracy = ", round(nb_accuracy, 2),
          ", error rate = ", round(1-nb_accuracy, 2), sep='')
    print("Dummy: accuracy = ", round(dummy_accuracy, 2),
          ", error rate = ", round(1-dummy_accuracy, 2), sep='')
    if tree_accuracy is not None:
        print("Decision Tree: accuracy = ", round(tree_accuracy, 2),
          ", error rate = ", round(1-tree_accuracy, 2), sep='')

def print_precision_recall_f1(knn, nb_class, dummy, tree=None):
    """print Precision recall and f1

    Args:
        knn_accuracy (list): Precision recall and f1 to KNN
        nb_accuracy (list): Precision recall and f1 to NB
        dummy_accuracy (list): Precision recall and f1 to DUMMY
    """
    print()
    print("=" * 80)
    print("Step 2: Precision, recall, and F1 measure")
    print("=" * 80)
    print("Stratified 10-fold Cross Validation")
    print("KNN: precision = ", round(knn[0], 2), ", recall = ",
          round(knn[1], 2), ", F1 measure = ", round(knn[2], 2), sep='')
    print("Naive Bayes: precision = ", round(nb_class[0], 2), ", recall = ",
          round(nb_class[1], 2), ", F1 measure = ", round(nb_class[2], 2), sep='')
    print("Dummy: precision = ", round(dummy[0], 2), ", recall = ",
          round(dummy[1], 2), ", F1 measure = ", round(dummy[2], 2), sep='')
    if tree is not None:
        print("Decision Tree: precision = ", round(tree[0], 2), ", recall = ",
          round(tree[1], 2), ", F1 measure = ", round(tree[2], 2), sep='')

def print_precision_recall_f1_helper(precision, recall, f1):
    """print Precision recall and f1
    """
    print()
    print("=" * 80)
    print("Step 1: Precision, recall, and F1 measure")
    print("=" * 80)
    print("Base Line: precision = ", round(precision, 2), ", recall = ",
          round(recall, 2), ", F1 measure = ", round(f1, 2), sep='')

def print_three_matrix(knn_matrix, nb_matrix, dummy_matrix):
    """Prints out the matrix created by the confusion matrix

    Args:
        knn_matrix (list): matrix for knn
        nb_matrix (list): matrix for nb
        dummy_matrix (list): matrix for dummy
    """
    print("=" * 80)
    print("Step 3: Confusion Matrices")
    print("=" * 80)
    print("kNN Classifer (Stratified 10-Fold Cross Validation Results):")
    print(tabulate(knn_matrix))
    print("\nNaive Bayes (Stratified 10-Fold Cross Validation Results):")
    print(tabulate(nb_matrix))
    print("\nDummy Classifer (Stratified 10-Fold Cross Validation Results):")
    print(tabulate(dummy_matrix))

def print_four_matrix(knn_matrix, nb_matrix, dummy_matrix, tree_matrix):
    """Prints out the matrix created by the confusion matrix

    Args:
        knn_matrix (list): matrix for knn
        nb_matrix (list): matrix for nb
        dummy_matrix (list): matrix for dummy
    """
    print("=" * 80)
    print("Step 3: Confusion Matrices")
    print("=" * 80)
    print("kNN Classifer (Stratified 10-Fold Cross Validation Results):")
    print(tabulate(knn_matrix))
    print("\nNaive Bayes (Stratified 10-Fold Cross Validation Results):")
    print(tabulate(nb_matrix))
    print("\nDummy Classifer (Stratified 10-Fold Cross Validation Results):")
    print(tabulate(dummy_matrix))
    print("\nDecision Tree Classifer (Stratified 10-Fold Cross Validation Results):")
    print(tabulate(tree_matrix))

def print_a_matrix(matrix):
    """Prints out the matrix created by the confusion matrix

    Args:
        matrix (list): matrix
    """
    print("=" * 80)
    print("Step 1: Confusion Matrices")
    print("=" * 80)
    print("Classifer (Stratified 10-Fold Cross Validation Results):")
    print(tabulate(matrix))

def matrix_helper(matrix, header, divider):
    """Helper function for Matrixs

    Args:
        matrix (list of lists): Matrix created by confusion matrix
        header (list): list of strings for header
        divider (list): list of ==== or --- for divider
    """
    for i, row in enumerate(matrix):# Append values to the nb_matrix
        total = sum(row)# Get sum
        if total != 0:# Set recognition
            recognition = 100.0 * row[i] / total
        else:
            recognition = 0.0
        if i == 0:# Insert
            row.insert(0, "Win")
        else:
            row.insert(0, "Lost")
        row.append(total)# Append total
        row.append(round(recognition, 1))# recognition
    matrix.insert(0, divider) # Insert the divider/header
    matrix.insert(0, header)

def x_train_helper(X_train, data_array):
    """
    Helper function for Naive Bayes

    Args:
        X_train (list): data to train on
        data_array (list): data is kept in array
    """
    for row in X_train:
        for i, j in enumerate(row):
            if i >= len(data_array): #if the len is less than i
                data_array.append({}) #append empty dic
            if j not in data_array[i]: #if not in
                data_array[i][j] = 0 #set to zero

def print_helper(tree, stack, attribute_names, class_name):
    """Helper function to figure out rules for decision

    Args:
        tree (list): tree built by tdidt
        stack (list): list to build
        attribute_names (list): list of attribute names
        class_name (str): class name
    """
    if tree[0] == "Attribute":
        for val in tree[2:]:
            add = [(tree[1], val[1])]
            print_helper(val[2], stack + add, attribute_names, class_name)
    elif tree[0] == "Leaf":
        print(get_rule(stack, tree[1], attribute_names, class_name))

def get_rule(stack, value, attribute_names, class_name):
    """_summary_

    Args:
        stack (list): stack to build
        value (str): value from tree
        attribute_names (str): string of attribute name
        class_name (str): class name

    Returns:
        str: rule that function finds
    """
    if attribute_names is None:
        name = lambda att: att
    else:
        name = lambda att: attribute_names[int(att[3:])]
    rule = "IF "
    for val in stack:
        rule += str(name(val[0])) + " == " + str(val[1]) + " "
        if val == stack[-1]:
            rule += "THEN "
        else:
            rule += "AND "
    rule += str(class_name) + " = " + str(value)
    return rule

def get_attribute_domains(X_train, header):
    """get attribute domains

    Args:
        X_train (list): data to train on
        header (list): header of data

    Returns:
        dict: attribute doamins of xtrain
    """
    attribute_domains = {}
    for i, j in enumerate(header):
        attribute_domains[j] = []
        for x in X_train:
            if x[i] not in attribute_domains[j]:
                attribute_domains[j].append(x[i])
    for key, val in attribute_domains.items():
        attribute_domains[key] = sorted(val)
    return attribute_domains

def partition_for_entropy(instances, index):
    """partition of entropy

    Args:
        instances (list): instances in the tree
        index (int): index

    Returns:
        partition: list of insttances
    """
    partitions = []
    unique = []
    for instance in instances:
        if instance[index] in unique:
            partition_index = unique.index(instance[index])
            partitions[partition_index].append(instance)
        else:
            unique.append(instance[index])
            partitions.append([instance])
    return partitions

def select_attribute(instances, available_attributes):
    """Finds a attribute with min entropy

    Args:
        instances (list): all instatnces
        available_attributes (list): avaibile attributes to split on

    Returns:
        str: attribute with minimal entropy to maximize gain
    """
    attribute_entropies = []
    for attribute in available_attributes:
        entropies = []
        denominators = []
        index = int(attribute[-1])
        partitions = partition_for_entropy(instances, index)
        for partition in partitions:
            unique_classifiers = []
            classifiers_counts = []
            value_entropy = 0
            for instance in partition:
                if instance[-1] in unique_classifiers:
                    classifier_index = unique_classifiers.index(instance[-1])
                    classifiers_counts[classifier_index] += 1
                else:
                    unique_classifiers.append(instance[-1])
                    classifiers_counts.append(1)
            denominator = len(partition)
            for count in classifiers_counts:
                if count == 0:
                    value_entropy = 0
                    break
                value_entropy -= count/denominator * math.log(count/denominator,2)
            entropies.append(value_entropy)
            denominators.append(denominator/len(instances))
        total_entropy = 0
        for entropy, demonitor in zip(entropies, denominators):
            total_entropy += entropy * demonitor
        attribute_entropies.append(total_entropy)
    min_entropy = min(attribute_entropies)
    att_index = attribute_entropies.index(min_entropy)
    return available_attributes[att_index]

def partition_instances(instances, split_attribute, attribute_domains, header):
    """partitions the instances for tdidt

    Args:
        instances (list): _description_
        split_attribute (str): domain split on
        attribute_domains (list): attibutes domain
        header (list): header of tree

    Returns:
        dict: dictionary of attribtute value to istance
    """
    attribute_domain = attribute_domains[split_attribute]
    attribute_index = header.index(split_attribute)
    partitions = {}
    for attribute_value in attribute_domain:
        partitions[attribute_value] = []
        for instance in instances:
            if instance[attribute_index] == attribute_value:
                partitions[attribute_value].append(instance)
    return partitions

def all_same_class(instances):
    """checks that the intances all are the same

    Args:
        instances (list): a set of all isntances

    Returns:
        Boolean: True or false
    """
    first_label = instances[0][-1]
    for instance in instances:
        if instance[-1] != first_label:
            return False
    return True

def compute_partition_stats(current_instances, index):
    """computes partitions stats

    Args:
        current_instances (list): list of current isntatnces
        index (int): index

    Returns:
        dict: final stats calcuated on partition
    """
    stats = {}
    for instance in current_instances:
        if instance[index] in stats:
            stats[instance[index]] += 1
        else:
            stats[instance[index]] = 1
    final_stats = []
    for key, val in stats.items():
        final_stats.append([key, val])
    return final_stats

def tdidt(current_instances, available_attributes, attribute_domains, header):
    """Returns the tree

    Args:
        current_instances (List of Lists): current instances
        available_attributes (List): avaible attributes
        attribute_domains (Dictionary): set of attributes
        header (List): header to tree

    Returns:
        list: The built up tree
    """
    # basic approach (uses recursion!!):
    # select an attribute to split on
    split_attribute = select_attribute(current_instances, available_attributes)
    #print("splitting on:", split_attribute)
    # remove split attribute from available attributes
    # because, we can't split on the same attribute twice in a branch
    available_attributes.remove(split_attribute) # Python is pass by object reference!!
    tree = ["Attribute", split_attribute]

    # group data by attribute domains (creates pairwise disjoint partitions)
    partitions = partition_instances(current_instances, split_attribute, attribute_domains, header)
    #print("partitions:", partitions)
    # for each partition, repeat unless one of the following occurs (base case)
    for attribute_value, partition in partitions.items():
        values_subtree = ["Value", attribute_value]
        # append your leaf nodes to this list appropriately
        #    CASE 1: all class labels of the partition are the same => make a leaf node
        if len(partition) > 0 and all_same_class(partition):
            # make a leaf node
            # look at class label and make an occurance of the most occuring class label????
            values_subtree.append(["Leaf", partition[0][-1], len(partition), len(current_instances)])
        #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(partition) > 0 and len(available_attributes) == 0:
            # majority vote leaf node
            partition_stats = compute_partition_stats(partition, -1)
            partition_stats.sort(key=lambda x:x[1])
            values_subtree.append(["Leaf", partition_stats[-1][0], len(partition), len(current_instances)])
        #    CASE 3: no more instances to partition (empty partition) => backtrack
        #    and replace attribute node with majority vote leaf node        elif len(partition) == 0:
        elif len(partition) == 0:
            # back trackand reapla e attribute node
            # with a majority vote leaf node over all
            partition_stats = compute_partition_stats(current_instances, -1)
            partition_stats.sort(key=lambda x:x[1])
            leaf = ["Leaf", partition_stats[-1][0], len(partition), len(current_instances)]
            return leaf
        else: # all base cases are false, recurse!!
            subtree = tdidt(partition, available_attributes.copy(), attribute_domains, header)
            values_subtree.append(subtree)
        tree.append(values_subtree)
    return tree


def tdidt_predict(header, tree, instance):
    """Uses tree created by fit to predict

    Args:
        header (List): List of headers
        tree (List): Tree
        instance (Instances): list of Instances

    Returns:
        predicted value: The path of a tree returns the leaf node
    """
    info_type = tree[0]
    if info_type == "Attribute":
        attribute_index = header.index(tree[1])
        instance_value = instance[attribute_index]
        for i in range(2, len(tree)):
            value_list = tree[i]
            if value_list[1] == instance_value:
                return tdidt_predict(header, value_list[2], instance)
    return tree[1]


def visual_tree(tree, dot_fname, pdf_fname, attribute_names):
    """Bonus for PA7

    Args:
        tree (list): tree created by tdidt
        dot_fname (str): dot file name
        pdf_fname (str): file to save to
        attribute_names (list): attribute names
    """
    if dot_fname and pdf_fname and attribute_names:
        for subtree in tree:
            print(subtree)
