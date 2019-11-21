from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from dataProcessing import *
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold


class AlgorithmEvaluation(Data):
    """
    class the apply the six classifiers on the data
    1) split-out validation dataset
    2) Test options and evaluation metric
    3) Spot Check Algorithms
    4) Compare Algorithms
    """

    def __init__(self, fileNameTrain, fileNameTest, num_folds=10):
        super().__init__(fileNameTrain, fileNameTest)
        self.num_folds = num_folds

    def algoEvaluation(self):
        seed = 7
        scoring = 'accuracy'
        # Spot-Check Algorithms
        models = []
        models.append(('LR', LogisticRegression()))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier()))
        models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC()))
        models.append(('NN', MLPClassifier()))
        models.append(('GBC', GradientBoostingClassifier()))

        results = []
        names = []
        for name, model in models:
            kfold = KFold(n_splits=self.num_folds, random_state=seed)
            cv_results = cross_val_score(model, self.X_train, self.y_train, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)

        # Compare Algorithms
        fig = pyplot.figure()
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)
        pyplot.boxplot(results)
        ax.set_xticklabels(names)
        pyplot.show()

data = Data()
X_train, y_train, X_test, y_test = data.getData(showData=False)

print(data.projLDA(X_train, y_train))
data.scatterPlot(X=X_train, y=y_train)



#algo_evaluation = AlgorithmEvaluation('data/train.csv', 'data/test.csv')
#algo_evaluation.algoEvaluation()
