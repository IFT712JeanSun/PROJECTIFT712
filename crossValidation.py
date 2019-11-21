from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from dataProcessing import *

class CrossValidation(Data):

    def __init__(self):
        super().__init__()
        self.penalty = 'l2'
        self.criterion = 'gini'
        self.n_neighbors = None
        self.activation = 'relu'
        self.C = None
        self.kernel = None

    def crossValidationForSVM(self):
        '''
        Do cross validation and reset model hyperparameters C and kernel to use
        :return: None
        '''
        X_scale = self.scale(self.X_train)
        num_folds = 10
        seed = 7
        scoring = 'accuracy'
        c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2]
        kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
        param_grid = dict(C=c_values, kernel=kernel_values)
        model = SVC()
        kfold = KFold(n_splits=num_folds, random_state=seed)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
        grid_result = grid.fit(X_scale, self.y_train)
        C = grid_result.best_params_['C']
        kernel = grid_result.best_params_['kernel']
        self.C = C
        self.kernel = kernel


    def crossValidationKNN(self):
        '''
        Do cross validation and reset model hyperparameters number of neighbors
        :return: None
        '''
        X_scale = self.scale(self.X_train)
        num_folds = 10
        seed = 7
        scoring = 'accuracy'
        neighbors = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
        param_grid = dict(n_neighbors=neighbors)
        model = KNeighborsClassifier()
        kfold = KFold(n_splits=num_folds, random_state=seed)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
        grid_result = grid.fit(X_scale, self.y_train)
        n_neighbors = grid_result.best_params_['n_neighbors']
        self.n_neighbors = n_neighbors

    def crossValidationLDA(self):
        X_scale = self.scale(self.X_train)
        num_folds = 10
        seed = 7
        scoring = 'accuracy'
        solver_values = ['svd', 'lsqr', 'eigen']
        param_grid = dict(solver=solver_values)
        model = LinearDiscriminantAnalysis()
        kfold = KFold(n_splits=num_folds, random_state=seed)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
        grid_result = grid.fit(X_scale, self.y_train)
        solver = grid_result.best_params_['solver']
        print(solver)
        self.solver = solver

