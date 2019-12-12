from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from crossValida import *
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier


class Algorithm(CrossValidation):
    """
    This class contains all the six class classifiers that we are going to use
    and the ensemble.
    """
    def __init__(self):
        '''
        class object to be used
        '''
        super().__init__()
        self.crossValidationForSVM()
        self.svm = SVC(C=self.C, kernel=self.kernel, degree=3, probability=True)

        self.crossValidationForSVM()
        self.dtc = DecisionTreeClassifier(criterion=self.criterion, max_depth=self.max_depth,
                                          max_leaf_nodes=self.max_leaf_nodes, min_samples_leaf=self.min_samples_leaf)

        self.crossValidationKNN()
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)

        self.crossValidationLDA()
        self.lda = LinearDiscriminantAnalysis(solver='svd')

        self.crossValidationNN()
        self.nn = MLPClassifier(activation=self.activation, solver=self.solver,
                                learning_rate=self.learning_rate, learning_rate_init=self.learning_rate_init)

        self.crossValidationLR()
        self.lr = LogisticRegression(penalty=self.penalty, C=self.C, tol=self.tol)

        self.gbc = GradientBoostingClassifier()
        self.rfc = RandomForestClassifier()
        self.abc = AdaBoostClassifier()

    def classifier(self):
        '''
        function to create a liste of the classifiers
        :return: the classifiers
        '''
        #clf = [self.svm, self.dtc, self.knn, self.lda, self.nn, self.lr, self.gbc, self.rfc, self.abc]
        clf = [self.svm, self.dtc, self.knn, self.lda, self.nn, self.lr]
        return clf

