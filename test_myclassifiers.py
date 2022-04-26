"""
Programmer: Troy Bechtold
Class: CptS 322-01, Spring 2022
Programming Assignment #4 and #5 and #6 and #7

Description: tests for classifers and pa4 and pa5 and pa6 and pa7
"""

import numpy as np
import random
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from mysklearn.myclassifiers import MyNaiveBayesClassifier, MyDecisionTreeClassifier, MyRandomForestClassifier
from mysklearn import myevaluation

# note: order is actual/received student value, expected/solution
#naive bayes data

inclass_example_col_names = ["att1", "att2"]
X_train_inclass_example = [
    [1, 5], # yes
    [2, 6], # yes
    [1, 5], # no
    [1, 5], # no
    [1, 6], # yes
    [2, 6], # no
    [1, 5], # yes
    [1, 6] # yes
]
y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]
class_test = [
    [1, 5]
    ]
class_true = ["yes"]
class_priors_solutions = {'no': 0.375, 'yes': 0.625}
class_posteriors_solutions = {'no': [{1: 0.6666666666666666, 2: 0.3333333333333333}, {5: 0.6666666666666666, 6: 0.3333333333333333}],
                              'yes': [{1: 0.8, 2: 0.2}, {5: 0.4, 6: 0.6}]}

iphone_col_names = ["standing", "job_status", "credit_rating", "buys_iphone"]
iphone_table = [
    [1, 3, "fair", "no"],
    [1, 3, "excellent", "no"],
    [2, 3, "fair", "yes"],
    [2, 2, "fair", "yes"],
    [2, 1, "fair", "yes"],
    [2, 1, "excellent", "no"],
    [2, 1, "excellent", "yes"],
    [1, 2, "fair", "no"],
    [1, 1, "fair", "yes"],
    [2, 2, "fair", "yes"],
    [1, 2, "excellent", "yes"],
    [2, 2, "excellent", "yes"],
    [2, 3, "fair", "yes"],
    [2, 2, "excellent", "no"],
    [2, 3, "fair", "yes"]
]
iphone_table_priors = {'no': 0.3333333333333333, 'yes': 0.6666666666666666}
iphone_table_posteriors = {'no': [{1: 0.6, 2: 0.4}, {1: 0.2, 2: 0.4, 3: 0.4}, {'excellent': 0.6, 'fair': 0.4}],
                           'yes': [{1: 0.2, 2: 0.8}, {1: 0.3, 2: 0.4, 3: 0.3}, {'excellent': 0.3, 'fair': 0.7}]}
iphone_test = [
    [2, 2, "fair"],
    [1, 1, "excellent"]
    ]
iphone_true = ["yes", "no"]

iphone_tree = ['Attribute', 'att0',
                ['Value', 1,
                    ['Attribute', 'att1',
                        ['Value', 1,
                            ['Leaf', 'yes', 1, 5]],
                        ['Value', 2,
                            ['Attribute', 'att2',
                                ['Value', 'excellent',
                                    ['Leaf', 'yes', 1, 2]
                                ],
                                ['Value', 'fair',
                                    ['Leaf', 'no', 1, 2]
                                ]
                            ]
                        ],
                        ['Value', 3,
                            ['Leaf', 'no', 2, 5]
                        ]
                    ]
                ],
                ['Value', 2,
                    ['Attribute', 'att2',
                        ['Value', 'excellent',
                            ['Leaf', 'yes', 0, 4]
                        ],
                        ['Value', 'fair',
                            ['Leaf', 'yes', 6, 10]
                        ]
                    ]
                ]
              ]

# Bramer 3.2 train dataset
train_col_names = ["day", "season", "wind", "rain", "class"]
train_table = [
    ["weekday", "spring", "none", "none", "on time"],
    ["weekday", "winter", "none", "slight", "on time"],
    ["weekday", "winter", "none", "slight", "on time"],
    ["weekday", "winter", "high", "heavy", "late"],
    ["saturday", "summer", "normal", "none", "on time"],
    ["weekday", "autumn", "normal", "none", "very late"],
    ["holiday", "summer", "high", "slight", "on time"],
    ["sunday", "summer", "normal", "none", "on time"],
    ["weekday", "winter", "high", "heavy", "very late"],
    ["weekday", "summer", "none", "slight", "on time"],
    ["saturday", "spring", "high", "heavy", "cancelled"],
    ["weekday", "summer", "high", "slight", "on time"],
    ["saturday", "winter", "normal", "none", "late"],
    ["weekday", "summer", "high", "none", "on time"],
    ["weekday", "winter", "normal", "heavy", "very late"],
    ["saturday", "autumn", "high", "slight", "on time"],
    ["weekday", "autumn", "none", "heavy", "on time"],
    ["holiday", "spring", "normal", "slight", "on time"],
    ["weekday", "spring", "normal", "none", "on time"],
    ["weekday", "spring", "normal", "slight", "on time"]
]

train_priors = {'cancelled': 0.05, 'late': 0.1, 'on time': 0.7, 'very late': 0.15}

train_posteriors = {
    'on time': [{'weekday': 0.6428571428571429, 'saturday': 0.14285714285714285, 'holiday': 0.14285714285714285, 'sunday': 0.07142857142857142},
                {'spring': 0.2857142857142857, 'winter': 0.14285714285714285, 'summer': 0.42857142857142855, 'autumn': 0.14285714285714285},
                {'none': 0.35714285714285715, 'high': 0.2857142857142857, 'normal': 0.35714285714285715},
                {'none': 0.35714285714285715, 'slight': 0.5714285714285714, 'heavy': 0.07142857142857142}],
    'late': [{'weekday': 0.5, 'saturday': 0.5, 'holiday': 0.0, 'sunday': 0.0},
             {'spring': 0.0, 'winter': 1.0, 'summer': 0.0, 'autumn': 0.0},
             {'none': 0.0, 'high': 0.5, 'normal': 0.5},
             {'none': 0.5, 'slight': 0.0, 'heavy': 0.5}],
    'very late': [{'weekday': 1.0, 'saturday': 0.0, 'holiday': 0.0, 'sunday': 0.0},
                  {'spring': 0.0, 'winter': 0.6666666666666666, 'summer': 0.0, 'autumn': 0.3333333333333333},
                  {'none': 0.0, 'high': 0.3333333333333333, 'normal': 0.6666666666666666},
                  {'none': 0.3333333333333333, 'slight': 0.0, 'heavy': 0.6666666666666666}],
    'cancelled': [{'weekday': 0.0, 'saturday': 1.0, 'holiday': 0.0, 'sunday': 0.0},
                  {'spring': 1.0, 'winter': 0.0, 'summer': 0.0,'autumn': 0.0},
                  {'none': 0.0, 'high': 1.0, 'normal': 0.0},
                  {'none': 0.0, 'slight': 0.0, 'heavy': 1.0}]
}

train_test = [
    ["weekday", "winter", "high", "heavy"],
    ["weekday", "summer", "high", "heavy"],
    ["sunday", "summer", "normal", "sligh"]
]
train_true = ["very late", "on time", "on time"]

# interview dataset
interview_header = ["level", "lang", "tweets", "phd", "interviewed_well"]
interview_table = [
        ["Senior", "Java", "no", "no", "False"],
        ["Senior", "Java", "no", "yes", "False"],
        ["Mid", "Python", "no", "no", "True"],
        ["Junior", "Python", "no", "no", "True"],
        ["Junior", "R", "yes", "no", "True"],
        ["Junior", "R", "yes", "yes", "False"],
        ["Mid", "R", "yes", "yes", "True"],
        ["Senior", "Python", "no", "no", "False"],
        ["Senior", "R", "yes", "no", "True"],
        ["Junior", "Python", "yes", "no", "True"],
        ["Senior", "Python", "yes", "yes", "True"],
        ["Mid", "Python", "no", "yes", "True"],
        ["Mid", "Java", "yes", "no", "True"],
        ["Junior", "Python", "no", "yes", "False"]
    ]

# note: this tree uses the generic "att#" attribute labels because fit() does not and should not accept attribute names
# note: the attribute values are sorted alphabetically
interview_tree = \
        ["Attribute", "att0",
            ["Value", "Junior",
                ["Attribute", "att3",
                    ["Value", "no",
                        ["Leaf", "True", 3, 5]
                    ],
                    ["Value", "yes",
                        ["Leaf", "False", 2, 5]
                    ]
                ]
            ],
            ["Value", "Mid",
                ["Leaf", "True", 4, 14]
            ],
            ["Value", "Senior",
                ["Attribute", "att2",
                    ["Value", "no",
                        ["Leaf", "False", 3, 5]
                    ],
                    ["Value", "yes",
                        ["Leaf", "True", 2, 5]
                    ]
                ]
            ]
        ]

# bramer degrees dataset
degrees_header = ["SoftEng", "ARIN", "HCI", "CSA", "Project", "Class"]
degrees_table = [
        ["A", "B", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "A", "FIRST"],
        ["A", "A", "A", "B", "B", "SECOND"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "A", "B", "B", "A", "FIRST"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
        ["A", "A", "A", "A", "A", "FIRST"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "A", "B", "SECOND"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["A", "A", "B", "A", "B", "FIRST"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["A", "A", "B", "B", "B", "SECOND"],
        ["B", "B", "B", "B", "B", "SECOND"],
        ["A", "A", "B", "A", "A", "FIRST"],
        ["B", "B", "B", "A", "A", "SECOND"],
        ["B", "B", "A", "A", "B", "SECOND"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["B", "A", "B", "A", "B", "SECOND"],
        ["A", "B", "B", "B", "A", "FIRST"],
        ["A", "B", "A", "B", "B", "SECOND"],
        ["B", "A", "B", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
    ]

degrees_tree = ['Attribute', 'att0',
                    ['Value', 'A',
                        ['Attribute', 'att4',
                            ['Value', 'A',
                                ['Leaf', 'FIRST', 5, 14]
                            ],
                            ['Value', 'B',
                                ['Attribute', 'att3',
                                    ['Value', 'A',
                                        ['Attribute', 'att1',
                                            ['Value', 'A',
                                                ['Leaf', 'FIRST', 1, 2]
                                            ],
                                            ['Value', 'B',
                                                ['Leaf', 'SECOND', 1, 2]
                                            ]
                                        ]
                                    ],
                                    ['Value', 'B',
                                        ['Leaf', 'SECOND', 7, 9]
                                    ]
                                ]
                            ]
                        ]
                    ],
                    ['Value', 'B',
                        ['Leaf', 'SECOND', 12, 26]
                    ]
                ]

tree_actual_1 = ['Attribute', 'att0',
                    ['Value', 'Junior',
                        ['Leaf', 'True', 2, 13]
                    ],
                    ['Value', 'Mid',
                        ['Attribute', 'att1',
                            ['Value', 'Java',
                                ['Leaf', 'False', 1,4]
                            ],
                            ['Value', 'Python',
                                ['Leaf','False', 0, 2]
                            ],
                            ['Value', 'R',
                                ['Leaf', 'False', 1, 4]
                            ]
                        ]
                    ],
                    ['Value', 'Senior',
                        ['Attribute', 'att1',
                            ['Value', 'Java',
                                ['Leaf', 'True', 2, 7]
                            ],
                            ['Value', 'Python',
                                ['Leaf', 'False', 3, 7]
                            ],
                            ['Value', 'R',
                                ['Leaf', 'True', 2, 7]
                            ]
                        ]
                    ]
                ]

tree_actual_2 = ['Attribute', 'att0',
                    ['Value', 'Junior',
                        ['Leaf', 'True', 0, 7]
                    ],
                    ['Value', 'Mid',
                        ['Leaf', 'True', 2, 13]
                    ],
                    ['Value', 'Senior',
                        ['Attribute', 'att1',
                            ['Value', 'Java',
                                ['Leaf', 'False', 0, 2]
                            ],
                            ['Value', 'Python',
                                ['Leaf', 'False', 1, 4]
                            ],
                            ['Value', 'R',
                                ['Leaf', 'True', 1, 4]
                            ]
                        ]
                    ]
                ]


def test_train_test_split():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    X_1 = [[0, 1],
       [2, 3],
       [4, 5],
       [6, 7],
       [8, 9]]
    y_1 = [0, 1, 2, 3, 4]
    # then put repeat values in
    X_2 = [[0, 1],
       [2, 3],
       [5, 6],
       [6, 7],
       [0, 1]]
    y_2 = [2, 3, 3, 2, 2]
    test_sizes = [0.33, 0.25, 4, 3, 2, 1]
    for X, y in zip([X_1, X_2], [y_1, y_2]):
        for test_size in test_sizes:
            X_train_solution, X_test_solution, y_train_solution, y_test_solution =\
                train_test_split(X, y, test_size=test_size, random_state=0, shuffle=False)
            X_train, X_test, y_train, y_test = myevaluation.train_test_split(X, y, test_size=test_size, random_state=0, shuffle=False)

            assert np.array_equal(X_train, X_train_solution) # order matters with np.array_equal()
            assert np.array_equal(X_test, X_test_solution)
            assert np.array_equal(y_train, y_train_solution)
            assert np.array_equal(y_test, y_test_solution)

    # if get here, should have base algorithm implemented just fine
    # now test random_state and shuffle
    test_size = 2
    X_train0_notshuffled, X_test0_notshuffled, y_train0_notshuffled, y_test0_notshuffled =\
        myevaluation.train_test_split(X_1, y_1, test_size=test_size, random_state=0, shuffle=False)
    X_train0_shuffled, X_test0_shuffled, y_train0_shuffled, y_test0_shuffled =\
        myevaluation.train_test_split(X_1, y_1, test_size=test_size, random_state=0, shuffle=True)
    # make sure shuffle keeps X and y parallel
    for i, _ in enumerate(X_train0_shuffled):
        assert y_1[X_1.index(X_train0_shuffled[i])] == y_train0_shuffled[i]
    # same random_state but with shuffle= False vs True should produce diff folds
    assert not np.array_equal(X_train0_notshuffled, X_train0_shuffled)
    assert not np.array_equal(y_train0_notshuffled, y_train0_shuffled)
    assert not np.array_equal(X_test0_notshuffled, X_test0_shuffled)
    assert not np.array_equal(y_test0_notshuffled, y_test0_shuffled)
    X_train1_shuffled, X_test1_shuffled, y_train1_shuffled, y_test1_shuffled =\
        myevaluation.train_test_split(X_1, y_1, test_size=test_size, random_state=1, shuffle=True)
    # diff random_state should produce diff folds when shuffle=True
    assert not np.array_equal(X_train0_shuffled, X_train1_shuffled)
    assert not np.array_equal(y_train0_shuffled, y_train1_shuffled)
    assert not np.array_equal(X_test0_shuffled, X_test1_shuffled)
    assert not np.array_equal(y_test0_shuffled, y_test1_shuffled)

# test utility function
def check_folds(n, n_splits, train_folds, test_folds, train_folds_solution, test_folds_solution):
    """Utility function

    n(int): number of samples in dataset
    """
    all_test_indices = []
    all_train_indices = []
    all_train_indices_solution = []
    all_test_indices_solution = []
    for i in range(n_splits):
        # make sure all indices are accounted for in each split
        all_indices_in_fold = train_folds[i] + test_folds[i]
        assert len(all_indices_in_fold) == n
        for index in range(n):
            assert index in all_indices_in_fold
        all_test_indices.extend(test_folds[i])
        all_train_indices.extend(train_folds[i])
        all_train_indices_solution.extend(train_folds_solution[i])
        all_test_indices_solution.extend(test_folds_solution[i])

    # make sure all indices are in a test set
    assert len(all_test_indices) == n
    for index in range(n):
        assert index in all_indices_in_fold
    # make sure fold test on appropriate number of indices
    all_test_indices.sort()
    all_test_indices_solution.sort()
    assert all_test_indices == all_test_indices_solution

    # make sure fold train on appropriate number of indices
    all_train_indices.sort()
    all_train_indices_solution.sort()
    assert all_train_indices == all_train_indices_solution

def test_kfold_cross_validation():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

    Notes:
        The order does not need to match sklearn's split() so long as the implementation is correct
    """
    X = [[0, 1], [2, 3], [4, 5], [6, 7]]
    y = [1, 2, 3, 4]

    n_splits = 2
    for tset in [X, y]:
        train_folds, test_folds = myevaluation.kfold_cross_validation(tset, n_splits=n_splits)
        standard_kf = KFold(n_splits=n_splits)
        train_folds_solution = []
        test_folds_solution = []
        # convert all solution numpy arrays to lists
        for train_fold_solution, test_fold_solution in list(standard_kf.split(tset)):
            train_folds_solution.append(list(train_fold_solution))
            test_folds_solution.append(list(test_fold_solution))
        check_folds(len(tset), n_splits, train_folds, test_folds, train_folds_solution, test_folds_solution)

    # more complicated dataset
    table = [
        [3, 2, "no"],
        [6, 6, "yes"],
        [4, 1, "no"],
        [4, 4, "no"],
        [1, 2, "yes"],
        [2, 0, "no"],
        [0, 3, "yes"],
        [1, 6, "yes"]
    ]
    # n_splits = 2, ..., 8 (LOOCV)
    for n_splits in range(2, len(table) + 1):
        train_folds, test_folds = myevaluation.kfold_cross_validation(table, n_splits=n_splits)
        standard_kf = KFold(n_splits=n_splits)
        train_folds_solution = []
        test_folds_solution = []
        # convert all solution numpy arrays to lists
        for train_fold_solution, test_fold_solution in list(standard_kf.split(table)):
            train_folds_solution.append(list(train_fold_solution))
            test_folds_solution.append(list(test_fold_solution))
        check_folds(len(table), n_splits, train_folds, test_folds, train_folds_solution, test_folds_solution)

    # if get here, should have base algorithm implemented just fine
    # now test random_state and shuffle
    train_folds0_notshuffled, test_folds0_notshuffled = myevaluation.kfold_cross_validation(X, n_splits=2, random_state=0, shuffle=False)
    train_folds0_shuffled, test_folds0_shuffled = myevaluation.kfold_cross_validation(X, n_splits=2, random_state=0, shuffle=True)
    # same random_state but with shuffle= False vs True should produce diff folds
    for i, _ in enumerate(train_folds0_notshuffled):
        assert not np.array_equal(train_folds0_notshuffled[i], train_folds0_shuffled[i])
        assert not np.array_equal(test_folds0_notshuffled[i], test_folds0_shuffled[i])
    train_folds1_shuffled, test_folds1_shuffled = myevaluation.kfold_cross_validation(X, n_splits=2, random_state=1, shuffle=True)
    # diff random_state should produce diff folds when shuffle=True
    for i, _ in enumerate(train_folds0_shuffled):
        assert not np.array_equal(train_folds0_shuffled[i], train_folds1_shuffled[i])
        assert not np.array_equal(test_folds0_shuffled[i], test_folds1_shuffled[i])

# test utility function
def get_min_label_counts(y, label, n_splits):
    """Utility function
    """
    label_counts = sum([1 for yval in y if yval == label])
    min_test_label_count = label_counts // n_splits
    min_train_label_count = (n_splits - 1) * min_test_label_count
    return min_train_label_count, min_test_label_count

def test_stratified_kfold_cross_validation():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold

    Notes:
        This test does not test shuffle or random_state
        The order does not need to match sklearn's split() so long as the implementation is correct
    """
    # note: this test case does test order against sklearn's
    X = [[0, 1], [2, 3], [4, 5], [6, 4]]
    y = [0, 0, 1, 1]

    n_splits = 2
    train_folds, test_folds = myevaluation.stratified_kfold_cross_validation(X, y, n_splits=n_splits)
    stratified_kf = StratifiedKFold(n_splits=n_splits)
    train_folds_solution = []
    test_folds_solution = []
    # convert all solution numpy arrays to lists
    for train_fold_solution, test_fold_solution in list(stratified_kf.split(X, y)):
        train_folds_solution.append(list(train_fold_solution))
        test_folds_solution.append(list(test_fold_solution))
    # sklearn solution and order:
    # i=0: TRAIN: [1 3] TEST: [0 2]
    # i=1: TRAIN: [0 2] TEST: [1 3]
    check_folds(len(y), n_splits, train_folds, test_folds, train_folds_solution, test_folds_solution)
    for i in range(n_splits):
        # since the actual result could have folds in diff order, make sure this train and test set is in the solution somewhere
        # sort the train and test sets of the fold so the indices can be in any order within a set
        # make sure at least minimum count of each label in each split
        for label in [0, 1]:
            train_yes_labels = [y[j] for j in train_folds[i] if y[j] == label]
            test_yes_labels = [y[j] for j in test_folds[i] if y[j] == label]
            min_train_label_count, min_test_label_count = get_min_label_counts(y, label, n_splits)
            assert len(train_yes_labels) >= min_train_label_count
            assert len(test_yes_labels) >= min_test_label_count

    # note: this test case does not test order against sklearn's solution
    table = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]
    ]
    table_y = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    for n_splits in range(2, 5):
        train_folds, test_folds = myevaluation.stratified_kfold_cross_validation(table, table_y, n_splits=n_splits)
        stratified_kf = StratifiedKFold(n_splits=n_splits)
        train_folds_solution = []
        test_folds_solution = []
        # convert all solution numpy arrays to lists
        for train_fold_solution, test_fold_solution in list(stratified_kf.split(table, table_y)):
            train_folds_solution.append(list(train_fold_solution))
            test_folds_solution.append(list(test_fold_solution))
        check_folds(len(table), n_splits, train_folds, test_folds, train_folds_solution, test_folds_solution)

        for i in range(n_splits):
            # make sure at least minimum count of each label in each split
            for label in ["yes", "no"]:
                train_yes_labels = [table_y[j] for j in train_folds[i] if table_y[j] == label]
                test_yes_labels = [table_y[j] for j in test_folds[i] if table_y[j] == label]
                min_train_label_count, min_test_label_count = get_min_label_counts(table_y, label, n_splits)
                assert len(train_yes_labels) >= min_train_label_count
                assert len(test_yes_labels) >= min_test_label_count

    # if get here, should have base algorithm implemented just fine
    # now test random_state and shuffle
    train_folds0_notshuffled, test_folds0_notshuffled = \
        myevaluation.stratified_kfold_cross_validation(X, y, n_splits=2, random_state=0, shuffle=False)
    train_folds0_shuffled, test_folds0_shuffled = myevaluation.stratified_kfold_cross_validation(X, y, n_splits=2, random_state=0, shuffle=True)
    # same random_state but with shuffle= False vs True should produce diff folds
    for i, _ in enumerate(train_folds0_notshuffled):
        assert not np.array_equal(train_folds0_notshuffled[i], train_folds0_shuffled[i])
        assert not np.array_equal(test_folds0_notshuffled[i], test_folds0_shuffled[i])
    train_folds1_shuffled, test_folds1_shuffled = myevaluation.stratified_kfold_cross_validation(X, y, n_splits=2, random_state=1, shuffle=True)
    # diff random_state should produce diff folds when shuffle=True
    for i, _ in enumerate(train_folds0_shuffled):
        assert not np.array_equal(train_folds0_shuffled[i], train_folds1_shuffled[i])
        assert not np.array_equal(test_folds0_shuffled[i], test_folds1_shuffled[i])

# test utility function
def check_same_lists_regardless_of_order(list1, list2):
    """Utility function
    """
    assert len(list1) == len(list2) # same length
    for item in list1:
        assert item in list2
        list2.remove(item)
    assert len(list2) == 0
    return True

def test_bootstrap_sample():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html

    Notes:
        This test does not test shuffle or random_state
    """
    X = [[1., 0.], [2., 1.], [0., 0.]]
    y = [0, 1, 2]
    # X_sample, y_sample = resample(X, y, random_state=0) #n_samples = None means length of first dimension
    X_sample_solution = [[1., 0.], [2., 1.], [1., 0.]]
    X_out_of_bag_solution = [[0., 0.]]
    y_sample_solution = [0, 1, 0]
    y_out_of_bag_solution = [2]

    X_sample, X_out_of_bag, y_sample, y_out_of_bag = myevaluation.bootstrap_sample(X, y, random_state=0)
    check_same_lists_regardless_of_order(X_sample, X_sample_solution)
    check_same_lists_regardless_of_order(y_sample, y_sample_solution)
    check_same_lists_regardless_of_order(X_out_of_bag, X_out_of_bag_solution)
    check_same_lists_regardless_of_order(y_out_of_bag, y_out_of_bag_solution)

    # another example adapted from
    # https://machinelearningmastery.com/a-gentle-introduction-to-the-bootstrap-method/
    X = [[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]]
    X_sample_solution = [[0.6], [0.4], [0.5], [0.1]]
    X_out_of_bag_solution = [[0.2], [0.3]]
    X_sample, X_out_of_bag, y_sample, y_out_of_bag = myevaluation.bootstrap_sample(X, y=None, n_samples=4, random_state=1)
    check_same_lists_regardless_of_order(X_sample, X_sample_solution)
    assert y_sample is None
    check_same_lists_regardless_of_order(X_out_of_bag, X_out_of_bag_solution)
    assert y_out_of_bag is None

def test_confusion_matrix():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 0, 2, 2, 0, 2]

    matrix_solution = [[2, 0, 0],
                [0, 0, 1],
                [1, 0, 2]]
    matrix = myevaluation.confusion_matrix(y_true, y_pred, [0, 1, 2])
    assert np.array_equal(matrix, matrix_solution)

    y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
    y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
    matrix = myevaluation.confusion_matrix(y_true, y_pred, ["ant", "bird", "cat"])
    assert np.array_equal(matrix, matrix_solution)

    y_true = [0, 1, 0, 1]
    y_pred = [1, 1, 1, 0]

    matrix_solution = [[0, 2],[1, 1]]
    matrix = myevaluation.confusion_matrix(y_true, y_pred, [0, 1])
    assert np.array_equal(matrix, matrix_solution)

def test_accuracy_score():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    y_pred = [0, 2, 1, 3]
    y_true = [0, 1, 2, 3]

    # normalize=True
    score = myevaluation.accuracy_score(y_true, y_pred, normalize=True)
    score_sol =  accuracy_score(y_true, y_pred, normalize=True) # 0.5
    assert np.isclose(score, score_sol)

    # normalize=False
    score = myevaluation.accuracy_score(y_true, y_pred, normalize=False)
    score_sol =  accuracy_score(y_true, y_pred, normalize=False) # 2
    assert np.isclose(score, score_sol)

def test_naive_bayes_classifier_fit():
    """Tests for naive bayes classifiers fit function
    """
    test_fit = MyNaiveBayesClassifier()
    test_fit.fit(X_train_inclass_example, y_train_inclass_example)

    assert class_priors_solutions == test_fit.priors
    assert class_posteriors_solutions == test_fit.posteriors

    x_train = [x[0:3] for x in iphone_table]
    y_train = [x[-1] for x in iphone_table]
    test_fit.fit(x_train, y_train)

    assert iphone_table_priors == test_fit.priors
    assert iphone_table_posteriors == test_fit.posteriors

    x_train = [x[0:4] for x in train_table]
    y_train = [x[-1] for x in train_table]
    test_fit.fit(x_train, y_train)

    assert train_priors == test_fit.priors
    assert train_posteriors == test_fit.posteriors

def test_naive_bayes_classifier_predict():
    """Tests for naive bayes classifiers predict function
    """
    test_predict = MyNaiveBayesClassifier()
    test_predict.fit(X_train_inclass_example, y_train_inclass_example)
    class_predicted = test_predict.predict(class_test)
    assert class_true == class_predicted

    x_train = [x[0:3] for x in iphone_table]
    y_train = [x[-1] for x in iphone_table]
    test_predict.fit(x_train, y_train)
    iphone_predicted = test_predict.predict(iphone_test)

    assert iphone_true == iphone_predicted

    x_train = [x[0:4] for x in train_table]
    y_train = [x[-1] for x in train_table]
    test_predict.fit(x_train, y_train)
    train_predicted = test_predict.predict(train_test)

    assert train_true == train_predicted

def test_decision_tree_classifier_fit():
    """Tests for decision tree classifiers fit function
    """
    X_train = []
    y_train = []
    X_train.append(interview_header)
    del X_train[0][-1]
    for row in interview_table:
        info = []
        for col in row:
            info.append(col)
        X_train.append(info)
    for row in interview_table:
        y_train.append(row[-1])
    test_fit = MyDecisionTreeClassifier()
    test_fit.fit(X_train, y_train)
    assert interview_tree == test_fit.tree

    X_train = []
    y_train = []
    X_train.append(degrees_header)
    del X_train[0][-1]
    for row in degrees_table:
        info = []
        for col in row:
            info.append(col)
        X_train.append(info)
    for row in degrees_table:
        y_train.append(row[-1])
    test_fit = MyDecisionTreeClassifier()
    test_fit.fit(X_train, y_train)
    assert test_fit.tree == degrees_tree

    X_train = []
    y_train = []
    X_train.append(iphone_col_names)
    del X_train[0][-1]
    for row in iphone_table:
        info = []
        for col in row:
            info.append(col)
        X_train.append(info)
    for row in iphone_table:
        y_train.append(row[-1])
    test_fit = MyDecisionTreeClassifier()
    test_fit.fit(X_train, y_train)
    print(test_fit.tree)
    assert test_fit.tree == iphone_tree

def test_decision_tree_classifier_predict():
    """Tests for decision tree classifiers predict function
    """
    X_train = []
    y_train = []
    X_test = [["Junior", "R", "yes", "no"], ["Junior", "Python", "no", "yes"], ["Senior", "Java", "no", "no", "False"]]
    actual = ["True", "True", "False"]
    X_train.append(interview_header)
    del X_train[0][-1]
    for row in interview_table:
        info = []
        for col in row:
            info.append(col)
        X_train.append(info)
    for row in interview_table:
        y_train.append(row[-1])
    test_predict = MyDecisionTreeClassifier()
    test_predict.fit(X_train, y_train)
    predicted = test_predict.predict(X_test)
    assert predicted == actual

    X_train = []
    y_train = []
    X_test = [["B", "B", "B", "B", "B"], ["A", "A", "A", "A", "A"], ["A", "A", "A", "A", "B"]]
    actual = ['SECOND', 'FIRST', 'FIRST']
    X_train.append(degrees_header)
    del X_train[0][-1]
    for row in degrees_table:
        info = []
        for col in row:
            info.append(col)
        X_train.append(info)
    for row in degrees_table:
        y_train.append(row[-1])
    test_predict = MyDecisionTreeClassifier()
    test_predict.fit(X_train, y_train)
    predicted2 = test_predict.predict(X_test)
    assert predicted2 == actual

    X_train = []
    y_train = []
    X_test = [
                [2, 2, "fair"],
                [1, 1, "excellent"]
            ]
    actual = ["yes", "yes"]
    X_train.append(iphone_col_names)
    del X_train[0][-1]
    for row in iphone_table:
        info = []
        for col in row:
            info.append(col)
        X_train.append(info)
    for row in iphone_table:
        y_train.append(row[-1])
    test_predict = MyDecisionTreeClassifier()
    test_predict.fit(X_train, y_train)
    predicted3 = test_predict.predict(X_test)
    assert predicted3 == actual

def test_MyRandomForestClassifier_fit():
    random.seed(1)
    # Interview DataSet

    # Create X_train and y_train
    X_train = []
    y_train = []
    # Append the header
    X_train.append(["level", "lang", "tweets", "phd", "interviewed_well"])
    # Delete the classifier
    del X_train[0][-1]
    # Get X_train
    for row in range(len(interview_table)):
        tmp = []
        for col in range(len(interview_table[0]) - 1):
            tmp.append(interview_table[row][col])
        X_train.append(tmp)

    # Get y_train
    for row in range(len(interview_table)):
        y_train.append(interview_table[row][-1])
    # Create a MyDecisionTreeClassifier object
    #print(X_train)
    test_fit = MyRandomForestClassifier(100, 2, 2)
    # Call fit

    test_fit.fit(X_train, y_train)
    # Test
    #print("working")
    #print(test_fit.forest)

    assert(test_fit.forest[0]['atts']) == ['att0', 'att1']
    assert(test_fit.forest[0]['tree'].tree) == tree_actual_1

    assert(test_fit.forest[1]['atts']) == ['att0', 'att1']
    assert(test_fit.forest[1]['tree'].tree) == tree_actual_2

def test_MyRandomForestClassifier_predict():
    random.seed(1)
    # Interview DataSet

    # Create X_train and y_train
    X_train = []
    y_train = []
    X_test = [["Junior", "R", "yes", "no"], ["Junior", "Python", "no", "yes"], ["Senior", "Java", "no", "no", "False"]]
    # Append the header
    X_train.append(["level", "lang", "tweets", "phd", "interviewed_well"])
    # Delete the classifier
    del X_train[0][-1]
    # Get X_train
    for row in range(len(interview_table)):
        tmp = []
        for col in range(len(interview_table[0]) - 1):
            tmp.append(interview_table[row][col])
        X_train.append(tmp)

    # Get y_train
    for row in range(len(interview_table)):
        y_train.append(interview_table[row][-1])
    # Create a MyDecisionTreeClassifier object
    #print(X_train)
    test_fit = MyRandomForestClassifier(100, 2, 2)
    # Call fit
    actual = ['True', 'True', 'True']
    test_fit.fit(X_train, y_train)
    predicted = test_fit.predict(X_test)
    assert predicted == actual