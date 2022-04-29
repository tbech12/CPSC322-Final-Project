"""
Programmer: Troy Bechtold
Class: CptS 322-01, Spring 2022
Programming Assignment #4

Description: First set of classifiers
"""
import copy
from mysklearn import myutils
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor


class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).

    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data

    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer, regressor=None):
        """Initializer for MySimpleLinearClassifier.

        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        self.regressor = regressor

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        self.regressor = myutils.compute_slope(X_train, y_train) #compute slope

    def predict(self, X_test):
        """Makes predictions for test samples in X_test by applying discretizer
            to the numeric predictions from regressor.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        predictions = []
        if self.regressor is not None:
            lin_regressor = MySimpleLinearRegressor(self.regressor[0], self.regressor[1]) #make a linear regressor obj
            predictions.append(lin_regressor) #add to prediction
            y_predicted = lin_regressor.predict(X_test) #predict values
            y_predicted = [round(y,3) for y in y_predicted] #round the values to 3
            predictions.append(y_predicted) #add the predicted
            discretize = []
            for y in y_predicted:
                discretize.append(self.discretizer(y)) #get values from discretizer
            predictions.append(discretize) #add new labels to predicted
        return predictions #retrun predicted

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances_copy = copy.deepcopy(self.X_train) #makes copy of x_train
        new_distances = []
        for i, instance in enumerate(distances_copy):
            instance.append(self.y_train[i]) #gets y_train values
            instance.append(i) #gets index

            if isinstance(X_test[0], str):
                dist = (myutils.compute_euclidean_distance2(instance[:len(X_test[0])], X_test[0]))
            else:
                dist = (myutils.compute_euclidean_distance(instance[:len(X_test[0])], X_test[0]))
            instance.append(dist) #adds dist
        for instance in distances_copy:
            new_distances.append(instance) #collects all insatnces
        train_sorted = sorted(new_distances, key=myutils.itemgetter(-1)) #gets the sorted disatances
        top = train_sorted[:self.n_neighbors] #gets top K amount
        distances = []
        neighbor_indices = []
        list_of_closest = []
        list_of_closest_not_rounded = []
        for i in range(self.n_neighbors):
            list_of_closest.append(round(top[i][-1], 3))
            list_of_closest_not_rounded.append(top[i][3])
        distances.append(list_of_closest)
        neighbor_indices.append(list_of_closest_not_rounded)
        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        _, neighbor_indices = self.kneighbors(X_test) #gets distance and indexes
        vals = []
        for i in range(self.n_neighbors):
            vals.append(neighbor_indices[0][i]) #get values from indexes
        neighbors = {}
        try:
            vals = [int(value) for value in vals] #convert to ints
        except TypeError:
            print("Can not convert to Ints")
        except ValueError:
            pass
        for index in vals:
            try:
                if self.y_train[index] in neighbors:
                    neighbors[self.y_train[index]] += 1 #add 1 to neighbors
                else:
                    neighbors[self.y_train[index]] = 1 #set to one
            except IndexError:
                if index in self.y_train and index in neighbors:
                    neighbors[index] += 1 #add 1 to neighbors
                else:
                    neighbors[index] = 1 #set to one
            except TypeError:
                if index in self.y_train and index in neighbors:
                    neighbors[index] += 1 #add 1 to neighbors
                else:
                    neighbors[index] = 1
        sorted_neighbors = sorted(neighbors.items(), key=myutils.itemgetter(1), reverse=True) #sort list
        y_predicted.append(sorted_neighbors[0][0]) #add the predicted
        return y_predicted #return predicted

class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.

        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        if X_train is not None and y_train is not None:
            most_frequent = (max(set(y_train), key = y_train.count)) #gets most frequent
            self.most_common_label = most_frequent

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        return [self.most_common_label for i in X_test] #returns list of most common label
class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        self.priors = {} #empty dic
        self.posteriors = {} #empty dic
        data_array = []
        for value in y_train:
            if value in self.priors: #if in priors
                self.priors[value] += 1 #add 1
            else: #else set to 1
                self.priors[value] = 1
        myutils.x_train_helper(X_train, data_array) #go to helper func
        for value in self.priors:
            self.posteriors[value] = copy.deepcopy(data_array) #make a copy
        for row, x in zip(X_train, y_train): #traver xtrain and ytrain
            for i, j in enumerate(row): #look at row
                self.posteriors[x][i][j] += 1 #set posteries to +1
        for value in self.posteriors:
            for i, row in enumerate(self.posteriors[value]):
                for data in row:
                    self.posteriors[value][i][data] /= self.priors[value] #devide by prior
        for value in self.priors:
            self.priors[value] /= len(y_train) #divide by len

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = [] #y predicted
        for row in X_test:
            probability = {}
            #print(self.posteriors)
            for value in self.posteriors: #for values in posteriors
                probability[value] = self.priors[value] #set the value of probability
                for i, j in enumerate(row):
                    try:
                        probability[value] *= self.posteriors[value][i][j] #try to multipy
                    except KeyError:
                        pass
            max_str_value = "" #max str
            max_val = -1 #max val
            for key, value in probability.items():
                if value > max_val: #if val is less
                    max_str_value = key #set the str
                    max_val = value #set the value
            y_predicted.append(max_str_value) #att the max str for each row
        return y_predicted #retrun predicted

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train, allowed_attributes = None):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train
        header = []
        if allowed_attributes is not None:
            header = allowed_attributes
        else:
            header = ['att' + str(i) for i in range(len(self.X_train[0]))]
        del self.X_train[0]
        available_attributes = header.copy()
        attribute_domains = myutils.get_attribute_domains(self.X_train, header)
        train = [self.X_train[i] + [self.y_train[i]] for i in range(len(self.X_train))]
        self.tree = myutils.tdidt(train, available_attributes, attribute_domains, header)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        header = ['att' + str(i) for i in range(len(self.X_train[0]))]
        for test in X_test:
            y_predicted.append(myutils.tdidt_predict(header, self.tree, test))
        return y_predicted

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        myutils.print_helper(self.tree, [], attribute_names, class_name)

    # BONUS method
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).

        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        myutils.visual_tree(self.tree, dot_fname, pdf_fname, attribute_names)# (BONUS) fix this

class MyRandomForestClassifier:
    def __init__(self, N, M, F, forest=None):
        """Initializer for MyRandomForestClassifier.

        """
        self.X_train = None
        self.y_train = None
        if forest != None:
            self.forest = forest
        else:
            self.forest = None
        self.N = N
        self.M = M
        self.F = F

    def fit(self, X_train, y_train):
        # Set X_train
        self.X_train = X_train
        # Set y_train
        self.y_train = y_train
        # Set header
        header = ['att' + str(i) for i in range(len(self.X_train[0]))]
        # Delete header
        del self.X_train[0]
        # Set train
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        # Create forest array
        forest = []
        # Traverse
        for _ in range(self.N):
            # Create tree dictionary
            tree = {}
            # Set attributes
            tree['atts'] = myutils.compute_random_subset(header[:-1], self.F)
            # Get train set and test set
            train_set, test_set = myutils.compute_bootstrapped_sample(train)
            # Set tree
            tree['tree'] = self.get_tree(train_set, tree['atts'] + [header[-1]])
            # Set accuracy
            tree['accuracy'] = self.compute_tree_accuracy(tree, test_set)
            # Append tree to forest
            forest.append(tree)
        # Sort
        sort = sorted(forest, key=myutils.itemgetter('accuracy'), reverse=True)
        # Set forest
        self.forest = sort[:self.M]

    def predict(self, X_test):
        # Create y_predicted list
        y_predicted = []
        # Predicted list
        predicted = []
        # Traverse forest
        for tree in self.forest:
            # Get the instance
            the_tree = tree['tree']
            # Call predict
            prediction = the_tree.predict(X_test)
            # Append to predicted
            predicted.append(prediction)
        # Traverse X_test
        for i in range(len(X_test)):
            # Create curr list
            curr = []
            # Traverse forest
            for j in range(len(self.forest)):
                # Append predicted[j][i] to curr
                curr.append(predicted[j][i])
            # Get items and frequency
            items, frequency = myutils.get_frequency(curr)
            # Get the best
            best = max(frequency)
            # Get the index of the best
            the_index = frequency.index(best)
            # Append the prediction to y_prediction
            y_predicted.append(items[the_index])
        # Return y_predicted
        return y_predicted

    def get_tree(self, train, attr):
        # Create an instance of MyDecisionTreeClassifier
        tree = MyDecisionTreeClassifier()
        # Set x
        x = [row[:-1] for row in train]
        # Set y
        y = [col[-1] for col in train]
        # Fit tree
        tree.fit(x, y, attr)
        # Return tree
        return tree

    def compute_tree_accuracy(self, tree, test_set):
        # Set x
        x = [row[:-1] for row in test_set]
        # Set Actual
        actual = [row[-1] for row in test_set]
        # Set predicted
        predicted = tree['tree'].predict(x)
        # Set total
        total = 0
        # Traverse
        for i in range(len(actual)):
            # Match
            if actual[i] == predicted[i]:
                # Increment total
                total += 1
        # Set accuracy
        accuracy = total / len(actual)
        # Return accuracy
        return accuracy