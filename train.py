from classifier import *
from crossValida import *


class train_testPrecision_Predict(Algorithm):
    '''
    class for training, testing the precision on validation data et predicting
    '''
    def __init__(self):
        super().__init__()

    def train(self, model=None, transform=False):
        '''
        Function that make training one classifiers by one classifiers
        :param model: model or classifier to use
        :param transform: if the data will be transformed by scaling or not
        :return: None
        '''
        if transform:
            X_train = self.scale(self.X_train)
        else:
            X_train = self.X_train

        # train the Logistic Regression
        if model == 'LR':
            print('Training the Logistic Regression ... \n')
            self.lr.fit(X_train, self.y_train)

        # train the neural network
        elif model == 'NN':
            print('Training the Neural Network ... \n')
            self.nn.fit(X_train, self.y_train)

        # train the Linear Discriminant Analysis
        elif model == 'LDA':
            print('Training the Linear Discriminant Analysis ... \n')
            self.lda.fit(X_train, self.y_train)

        # train the  K Neighbors Classifier
        elif model == 'KNN':
            print(' Training the K Neighbors Classifier ... \n')
            self.knn.fit(X_train, self.y_train)

        # train the Decision Tree Classifier
        elif model == 'DTC':
            print(' Training the Decision Tree Classifier ... \n')
            self.dtc.fit(X_train, self.y_train)

        # train the Support Vector Machine
        elif model == 'SVM':
            print(' Training the Support Vector Machine ... \n')
            self.svm.fit(X_train, self.y_train)

        # train the Gradient Boosting Classifier
        elif model == 'GBC':
            print(' Training Gradient Boosting Classifier ... \n')
            self.gbc.fit(X_train, self.y_train)

        # train the Random Forest Classifier
        elif model == 'RFC':
            print(' Training Random Forest Classifier ... \n')
            self.rfc.fit(X_train, self.y_train)

        # train Ada Boost Classifier
        elif model == 'ABC':
            print('Training Ada Boost Classifier ... \n')
            self.abc.fit(X_train, self.y_train)

        else:
            print('Training all the classifiers ... \n')
            print('Training the Logistic Regression ... \n')
            self.lr.fit(X_train, self.y_train)
            print('Training the Neural Network ... \n')
            self.nn.fit(X_train, self.y_train)
            print('Training the Linear Discriminant Analysis ... \n')
            self.lda.fit(X_train, self.y_train)
            print(' Training the K Neighbors Classifier ... \n')
            self.knn.fit(X_train, self.y_train)
            print(' Training the Decision Tree Classifier ... \n')
            self.dtc.fit(X_train, self.y_train)
            print(' Training the Support Vector Machine ... \n')
            self.svm.fit(X_train, self.y_train)
            print(' Training Gradient Boosting Classifier ... \n')
            self.gbc.fit(X_train, self.y_train)
            print(' Training Random Forest Classifier ... \n')
            self.rfc.fit(X_train, self.y_train)
            print(' Ada Boost Classifier ... \n')
            self.abc.fit(X_train, self.y_train)


    def testPrecision(self, model=None, transform=False):
        '''
        function to teste on the validation set
        :param model: the model to use
        :param transform: if transformation or not
        :return: prediction
        '''
        if transform:
            X_test = self.scale(self.X_test)
        else:
            X_test = self.X_test

        precision = {}
        if model == 'LR':
            precision['PRECISION LR: '] = self.lr.score(X_test, self.y_test)
        elif model == 'NN':
            precision['PRECISION NN: '] = self.nn.score(X_test, self.y_test)
        elif model == 'LDA':
            precision['PRECISION LDA: '] = self.lda.score(X_test, self.y_test)
        elif model == 'KNN':
            precision['PRECISION KNN: '] = self.knn.score(X_test, self.y_test)
        elif model == 'DTC':
            precision['PRECISION DTC: '] = self.dtc.score(X_test, self.y_test)
        elif model == 'SVM':
            precision['PRECISION SVM: '] = self.svm.score(X_test, self.y_test)
        elif model == 'GBC':
            precision['PRECISION GBC: '] = self.gbc.score(X_test, self.y_test)
        elif model == 'RFC':
            precision['PRECISION RFC: '] = self.rfc.score(X_test, self.y_test)
        elif model == 'ABC':
            precision['PRECISION ABC: '] = self.abc.score(X_test, self.y_test)
        else:
            print('Precision of all the classifiers ... \n')
            precision['PRECISION LR: '] = self.lr.score(X_test, self.y_test)
            precision['PRECISION NN: '] = self.nn.score(X_test, self.y_test)
            precision['PRECISION LDA: '] = self.lda.score(X_test, self.y_test)
            precision['PRECISION KNN: '] = self.knn.score(X_test, self.y_test)
            precision['PRECISION DTC: '] = self.dtc.score(X_test, self.y_test)
            precision['PRECISION SVM: '] = self.svm.score(X_test, self.y_test)
            precision['PRECISION GBC: '] = self.gbc.score(X_test, self.y_test)
            precision['PRECISION RFC: '] = self.rfc.score(X_test, self.y_test)
            precision['PRECISION ABC: '] = self.abc.score(X_test, self.y_test)

        return precision

    def predict(self, model=None, transform=False):
        '''
        function for prediction
        :param model: the model to use
        :param transform: if transformation or not
        :return: predictoion
        '''
        if transform:
            X_unknown = self.scale(self.X_unknown)
        else:
            X_unknown = self.X_unknown
        prediction = {}

        # prediction of  the Logistic Regression
        if model == 'LR':
            prediction[str('LR PREDICTION')] = ''
            for n in range(len(self.X_unknown)):
                class_index = self.lr.predict([X_unknown.iloc[n]])
                prediction[str('x[')+str(n)+str(']')+str(self.id[n])] = self.className[class_index[0]]

        # prediction of the neural network
        elif model == 'NN':
            prediction[str('NN PREDICTION')] = ''
            for n in range(len(self.X_unknown)):
                class_index = self.nn.predict([X_unknown.iloc[n]])
                prediction[str('x[')+str(n)+str(']')+str(self.id[n])] = self.className[class_index[0]]

        # prediction of the Linear Discriminant Analysis
        elif model == 'LDA':
            prediction[str('LDA PREDICTION')] = ''
            for n in range(len(self.X_unknown)):
                class_index = self.lda.predict([X_unknown.iloc[n]])
                prediction[str('x[')+str(n)+str(']')+str(self.id[n])] = self.className[class_index[0]]

        # prediction of the  K Neighbors Classifier
        elif model == 'KNN':
            prediction[str('KNN PREDICTION')] = ''
            for n in range(len(self.X_unknown)):
                class_index = self.knn.predict([X_unknown.iloc[n]])
                prediction[str('x[')+str(n)+str(']')+str(self.id[n])] = self.className[class_index[0]]

        # prediction of the Decision Tree Classifier
        elif model == 'DTC':
            prediction[str('DTC PREDICTION')] = ''
            for n in range(len(self.X_unknown)):
                class_index = self.dtc.predict([X_unknown.iloc[n]])
                prediction[str('x[')+str(n)+str(']')+str(self.id[n])] = self.className[class_index[0]]

        # prediction of the Support Vector Machine
        elif model == 'SVM':
            prediction[str('SVM PREDICTION')] = ''
            for n in range(len(self.X_unknown)):
                class_index = self.svm.predict([X_unknown.iloc[n]])
                prediction[str('x[')+str(n)+str(']')+str(self.id[n])] = self.className[class_index[0]]
        return prediction


    def crossValidationResults(self, transform=True):
        '''
        function for cross validation result for model performance
        :param transform:
        :return:
        '''
        score = {}
        print(" \n Cross validation and accuracy for SVM ...\n")
        print(self.modelCrossValidation(model=self.svm, transform=transform))
        score['SVM accuracy: '] = self.modelCrossValidation(model=self.svm, transform=transform)
        print(" \n Cross validation and accuracy for DTC ...\n")
        print(self.modelCrossValidation(model=self.dtc, transform=transform))
        score['DTC accuracy: '] = self.modelCrossValidation(model=self.dtc, transform=transform)
        print(" \n Cross validation and accuracy for KNN ...\n")
        print(self.modelCrossValidation(model=self.knn, transform=transform))
        score['KNN accuracy: '] = self.modelCrossValidation(model=self.knn, transform=transform)
        print(" \n Cross validation and accuracy for LDA ...\n")
        print(self.modelCrossValidation(model=self.lda, transform=transform))
        score['LDA accuracy: '] = self.modelCrossValidation(model=self.lda, transform=transform)
        print(" \n Cross validation and accuracy for NN ...\n")
        print(self.modelCrossValidation(model=self.nn, transform=transform))
        score['NN accuracy: '] = self.modelCrossValidation(model=self.nn, transform=transform)
        print(" \n Cross validation and accuracy for LR ...\n")
        print(self.modelCrossValidation(model=self.lr, transform=transform))
        score['LR accuracy: '] = self.modelCrossValidation(model=self.lr, transform=transform)
        print(" \n Cross validation and accuracy for all ...\n")
        print(score)
        return score


    def predictProb(self):
        '''
        function for probability from cross validation
        :return:
        '''
        submit = self.modelCrossValidationPredictProb(model=self.nn)
        return submit


