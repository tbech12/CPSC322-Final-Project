import mysklearn.myutils as myutils
import mysklearn.myevaluation as myevaluation
from mysklearn.myclassifiers import MyKNeighborsClassifier, MyDummyClassifier, MyNaiveBayesClassifier, MyDecisionTreeClassifier
from mysklearn.mypytable import MyPyTable

def set_up(movie_table: MyPyTable):
    knn = MyKNeighborsClassifier()
    dummy = MyDummyClassifier()
    naive = MyNaiveBayesClassifier()
    tree = MyDecisionTreeClassifier()
    print(movie_table.get_shape())

    return knn, dummy, naive, tree
