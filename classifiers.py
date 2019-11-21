from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from crossValidation import *


class Algorithm(CrossValidation):
    """
    This class contains all the six class classifiers that we are going to use
    """
    def __init__(self):
        super().__init__()
        self.crossValidationForSVM()
        self.svm = SVC(C=self.C, kernel=self.kernel, degree=3)

        self.dtc = DecisionTreeClassifier(criterion=self.criterion, min_samples_split=2, min_samples_leaf=1,)

        self.crossValidationKNN()
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)

        #self.crossValidationLDA()
        self.lda = LinearDiscriminantAnalysis(solver='svd')

        self.nn = MLPClassifier(activation=self.activation, solver='adam')

        #this is allready the cross validation version of the logistic regression
        self.lrcv = LogisticRegressionCV(penalty=self.penalty)