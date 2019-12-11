from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from dataProcessing import Data

class CrossValidation(Data):

    def __init__(self):
        super().__init__()
        self.penalty = 'l2'
        self.criterion = 'gini'
        self.n_neighbors = None
        self.activation = 'relu'
        self.C = None
        self.kernel = None
        self.max_depth = None
        self.max_leaf_nodes = None
        self.min_samples_leaf = None
        

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
        print("SVM accuracy = ",grid_result.best_score_)     
        self.C = C
        self.kernel = kernel
        print("SVM C = ",self.C)
        print("SVM kernel = ",self.kernel)


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
        print("KNN accuracy = ",grid_result.best_score_)
        self.n_neighbors = n_neighbors
        print("KNN n_neighbors = ",self.n_neighbors)

    def crossValidationLDA(self):
        X_scale = self.scale(self.X_train)
        num_folds = 10
        seed = 7
        scoring = 'accuracy'
        solver_values = ['svd', 'lsqr', 'eigen']
        #solver_values = ['svd']
        param_grid = dict(solver=solver_values)
        model = LinearDiscriminantAnalysis()
        kfold = KFold(n_splits=num_folds, random_state=seed)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
        grid_result = grid.fit(X_scale, self.y_train)
        solver = grid_result.best_params_['solver']
        print("LDA accuracy = ",grid_result.best_score_)
        self.solver = solver
        
    def crossValidationDT(self):
        
        X_scale = self.scale(self.X_train)
        scoring = 'accuracy'
        num_folds = 10
        seed = 7
        max_depth = range(5,50,5)
        max_leaf_nodes = range(80,120,5)
        min_samples_leaf = range(1,3)
        param_grid = dict(max_depth = max_depth, max_leaf_nodes = max_leaf_nodes, min_samples_leaf = min_samples_leaf)
        kfold = KFold(n_splits=num_folds, random_state=seed)
        model = DecisionTreeClassifier(random_state=0)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
        grid_result = grid.fit(X_scale, self.y_train)
        md = grid_result.best_params_['max_depth']
        mln = grid_result.best_params_['max_leaf_nodes']
        msl = grid_result.best_params_['min_samples_leaf']
        print("DT accuracy = ",grid_result.best_score_)
        self.max_depth = md
        self.max_leaf_nodes = mln
        self.min_samples_leaf = msl
        
    def crossValidationNN(self):
        
        X_scale = self.scale(self.X_train)
        scoring = 'accuracy'
        num_folds = 10
        seed = 7
        activation = ['identity', 'logistic', 'tanh', 'relu']
        solver =['lbfgs', 'sgd', 'adam']
        learning_rate =['constant', 'invscaling', 'adaptive']
        param_grid = dict(activation = activation, solver = solver, learning_rate = learning_rate)
        kfold = KFold(n_splits=num_folds, random_state=seed)
        model = MLPClassifier()
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
        grid_result = grid.fit(X_scale, self.y_train)
        solver = grid_result.best_params_['solver']
        activation = grid_result.best_params_['activation']
        learning_rate = grid_result.best_params_['learning_rate']
        print("NN accuracy = ",grid_result.best_score_)
        self.solver = solver
        self.activation = activation
        self.learning_rate = learning_rate
        
        

