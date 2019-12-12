import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, cross_val_predict, StratifiedKFold

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from dataProcess import *

class CrossValidation(Data):
    '''
    class for the cross validation
    '''

    def __init__(self):
        super().__init__()
        self.penalty = 'l2'
        self.criterion = 'gini'
        self.n_neighbors = None
        self.activation = 'relu'
        self.C = None
        self.kernel = None
        self.solver = 'lbfgs'
        self.activation = 'identity'
        self.learning_rate = 'constant'
        self.Cs = 0
        self.multi_class = 'auto'
        self.max_depth = None
        self.max_leaf_nodes = None
        self.min_samples_leaf = 1
        self.learning_rate_init = None

    def crossValidationForSVM(self):
        '''
        Do cross validation and reset model hyperparameters C and kernel to use
        :return: None
        '''
        X_scale = self.scale(self.data_X_train)
        num_folds = 10
        seed = 7
        scoring = 'accuracy'
        c_values = [0.1, 0.5, 0.7, 1.0, 1.3, 1.7, 2]
        kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
        param_grid = dict(C=c_values, kernel=kernel_values)
        model = SVC()
        kfold = KFold(n_splits=num_folds, random_state=seed)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
        grid_result = grid.fit(X_scale, self.data_Y_train)
        C = grid_result.best_params_['C']
        kernel = grid_result.best_params_['kernel']
        self.C = C
        self.kernel = kernel


    def crossValidationKNN(self):
        '''
        Do cross validation and reset model hyperparameters number of neighbors
        :return: None
        '''
        X_scale = self.scale(self.data_X_train)
        num_folds = 10
        seed = 7
        scoring = 'accuracy'
        neighbors = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
        param_grid = dict(n_neighbors=neighbors)
        model = KNeighborsClassifier()
        kfold = KFold(n_splits=num_folds, random_state=seed)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
        grid_result = grid.fit(X_scale, self.data_Y_train)
        n_neighbors = grid_result.best_params_['n_neighbors']
        self.n_neighbors = n_neighbors

    def crossValidationLDA(self):
        '''
        Do cross validation and reset model hyperparameters for LDA
        :return: None
        '''
        X_scale = self.scale(self.data_X_train)
        num_folds = 10
        seed = 7
        scoring = 'accuracy'
        solver_values = ['svd', 'lsqr', 'eigen']
        param_grid = dict(solver=solver_values)
        model = LinearDiscriminantAnalysis()
        kfold = KFold(n_splits=num_folds, random_state=seed)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
        grid_result = grid.fit(X_scale, self.data_Y_train)
        solver = grid_result.best_params_['solver']
        self.solver = solver

    def crossValidationNN(self):
        '''
        cross validation for the parameters of neural network
        :return: None
        '''
        X_scale = self.scale(self.data_X_train)
        scoring = 'accuracy'
        num_folds = 10
        seed = 7
        activation = ['identity', 'logistic', 'tanh', 'relu']
        solver = ['lbfgs', 'sgd', 'adam']
        learning_rate = ['constant', 'invscaling', 'adaptive']
        learning_rate_init = [0.001, 0.002, 0.01]
        param_grid = dict(activation=activation, solver=solver, learning_rate=learning_rate,
                          learning_rate_init=learning_rate_init)
        kfold = KFold(n_splits=num_folds, random_state=seed)
        model = MLPClassifier()
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
        grid_result = grid.fit(X_scale, self.data_Y_train)
        solver = grid_result.best_params_['solver']
        activation = grid_result.best_params_['activation']
        learning_rate = grid_result.best_params_['learning_rate']
        learning_rate_init = grid_result.best_params_['learning_rate_init']
        self.learning_rate_init = learning_rate_init
        self.solver = solver
        self.activation = activation
        self.learning_rate = learning_rate

    def crossValidationDT(self):
        '''
        cross validation for DTC parameters
        :return: None
        '''
        X_scale = self.scale(self.X_train)
        scoring = 'accuracy'
        num_folds = 10
        seed = 7
        max_depth = range(5, 50, 5)
        max_leaf_nodes = range(80, 120, 5)
        min_samples_leaf = range(1, 3)
        param_grid = dict(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf)
        kfold = KFold(n_splits=num_folds, random_state=seed)
        model = DecisionTreeClassifier(random_state=0)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
        grid_result = grid.fit(X_scale, self.y_train)
        md = grid_result.best_params_['max_depth']
        mln = grid_result.best_params_['max_leaf_nodes']
        msl = grid_result.best_params_['min_samples_leaf']
        self.max_depth = md
        self.max_leaf_nodes = mln
        self.min_samples_leaf = msl

    def crossValidationLR(self):
        '''
        cross validation for LR hyperpaprameter
        :return:
        '''
        X_scale = self.scale(self.data_X_train)
        scoring = 'accuracy'
        num_folds = 10
        seed = 7
        c = [1, 10, 20, 50, 100, 1000, 2000]
        tol = [0.005, 0.003, 0.001]
        param_grid = dict(C=c, tol=tol)
        kfold = KFold(n_splits=num_folds, random_state=seed)
        model = LogisticRegression(solver='newton-cg', multi_class='multinomial')

        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
        grid_result = grid.fit(X_scale, self.data_Y_train)
        c = grid_result.best_params_['C']
        tol = grid_result.best_params_['tol']
        self.C = c
        self.tol = tol


    def modelCrossValidation(self, model=None, kfold=10, transform=False):
        '''
        general function for Cross validation to increase the precision of the model
        :param model: the model to usse
        :param kfold: the number of k fold
        :param transform: if the transformation will be donne or not
        :return:
        '''
        if transform:
            X_train = self.scale(self.data_X_train)
            seed = 7
            kf = KFold(n_splits=kfold, random_state=seed)
            res = cross_val_score(model, X_train, self.data_Y_train, cv=kf, scoring='accuracy')
        else:
            seed = 7
            kf = KFold(n_splits=kfold, random_state=seed)
            res = cross_val_score(model, self.data_X_train, self.data_Y_train, cv=kf, scoring='accuracy')
        return res.mean()

    def modelCrossValidationPredictProb(self, model=None, kfold=10, transform=False):
        '''
        Same function as before but return the probability
        :param model: the model
        :param kfold: the number of k fold
        :param transform: the transformation of the data or not
        :return: probability
        '''
        if transform:
            X_train = self.scale(self.X_train)
            seed = 7
            kf = KFold(n_splits=kfold, random_state=seed)
            prob = cross_val_predict(model, X_train, self.data_Y_train, cv=kf, method="predict_proba")
        else:
            seed = 7
            kf = KFold(n_splits=kfold, random_state=seed)
            prob = cross_val_predict(model, self.data_X_train, self.data_Y_train, cv=kf, method="predict_proba")
        return prob


