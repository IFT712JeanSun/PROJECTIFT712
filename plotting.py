from sklearn.metrics import confusion_matrix, classification_report

from classifier import *
from crossValidation import *
from sklearn import metrics
from sklearn.utils.multiclass import unique_labels

class Plotting(Algorithm):
    '''
    class for making figures
    '''
    def __init__(self):
        super().__init__()

    def evaluateModel(self, model, transform=False):
        '''
        Evaluate the model
        :param model: the model or classifier
        :return: the accuracy score
        '''
        if transform:
            X_train = self.scale(self.X_train)
            X_test = self.scale(self.X_test)
            model.fit(X_train, self.y_train)
            y_predict = model.predict(X_test)
            accuracy_score = metrics.accuracy_score(self.y_test, y_predict)
        else:
            model.fit(self.X_train, self.y_train)
            y_predict = model.predict(self.X_test)
            accuracy_score = metrics.accuracy_score(self.y_test, y_predict)
        return accuracy_score

    def evaluateModelCrossValidation(self, model, transform=False):
        '''
        Evaluate the model
        :param model: the model or classifier
        :return: the accuracy score
        '''
        accuracyScore = self.modelCrossValidation(model=model, transform=transform)
        return accuracyScore

    def barPlot(self):
        '''
        Plot barplot of accuracy of each classifier.
        :return: None accuracy_score = self.evaluateModel(clf, transform=False)
        '''
        print("ploting the barplot ...\n")

        classifiers = self.classifier()
        accuracy_scoresF = list()
        accuracy_scoresT = list()
        for clf in classifiers:
            accuracy_scoreF = self.evaluateModel(clf, transform=False)
            accuracy_scoreT = self.evaluateModel(clf, transform=True)
            accuracy_scoresF.append(accuracy_scoreF)
            accuracy_scoresT.append(accuracy_scoreT)
        log_cols = ['Classificateurs', 'Précision en pourcentage']
        clf = ['SVM', 'DTC', 'KNN', 'LDA', 'NN', 'LR']
        logF = pd.DataFrame(columns=log_cols)
        logT = pd.DataFrame(columns=log_cols)
        for i in range(0, len(classifiers)):
            logF = logF.append(
                pd.DataFrame([[clf[i], 100*accuracy_scoresF[i]]], columns=log_cols), ignore_index=True)
        for i in range(0, len(classifiers)):
            logT = logT.append(
                pd.DataFrame([[clf[i], 100*accuracy_scoresT[i]]], columns=log_cols), ignore_index=True)
        sns.barplot(x='Précision en pourcentage', y='Classificateurs', data=logT, color='black', alpha=0.4)
        sns.barplot(x='Précision en pourcentage', y='Classificateurs', data=logF, color='blue', alpha=0.4)
        plt.xlabel('Précision en pourcentage')
        plt.savefig("barplot.pdf")
        plt.show()

    def barPlotCrossValidation(self, transform=False):
        '''
        Plot barplot of accuracy of each classifier.
        :return: None
        '''
        print("ploting the barplot ...\n")
        classifiers = self.classifier()
        accuracy_scoresF = list()
        accuracy_scoresT = list()
        for clf in classifiers:
            accuracy_scoreF = self.evaluateModelCrossValidation(clf, transform=False)
            accuracy_scoreT = self.evaluateModelCrossValidation(clf, transform=True)
            accuracy_scoresF.append(accuracy_scoreF)
            accuracy_scoresT.append(accuracy_scoreT)
        log_cols = ['Classificateurs', 'Précision en pourcentage']
        clf = ['SVM', 'DTC', 'KNN', 'LDA', 'NN', 'LR']
        logF = pd.DataFrame(columns=log_cols)
        logT = pd.DataFrame(columns=log_cols)
        for i in range(0, len(classifiers)):
            logF = logF.append(
                pd.DataFrame([[clf[i], 100*accuracy_scoresF[i]]], columns=log_cols), ignore_index=True)
        for i in range(0, len(classifiers)):
            logT = logT.append(
                pd.DataFrame([[clf[i], 100*accuracy_scoresT[i]]], columns=log_cols), ignore_index=True)
        sns.barplot(x='Précision en pourcentage', y='Classificateurs', data=logT, color='black', alpha=0.4)
        sns.barplot(x='Précision en pourcentage', y='Classificateurs', data=logF, color='blue', alpha=0.4)
        plt.xlabel('Précision en pourcentage')
        plt.savefig("barplotCV.pdf")
        plt.show()

    def information(self):
        """
        calcul the confusion matrix and the classification report
        :param model: model to use
        :return: confusion matrice and classification report
        """
        model = self.lda

        model.fit(self.X_train, self.y_train)
        prediction = model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, prediction)
        cr = classification_report(self.y_test, prediction)
        return cm, cr, prediction

    def plot_confusion_matrix(self, normalize=False, title=None):

        """
        This function prints and plots the confusion matrix.
         Normalization can be applied by setting `normalize=True`.
        """
        print("plotting the confusion matrix")
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'
        classes = self.y_test
        # Compute confusion matrix
        cm, cr, predictions = self.information()

        # Only use the labels that appear in the data
        classes = classes[unique_labels(self.y_test, predictions)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=classes, yticklabels=classes,
            title=title,
            ylabel='True label',
            xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax

    def scatterPlot(self, colorMap='Paired'):
        """
        function for scatter plot.
        :param X: data to plot, the scaled data from scale() function
        :param y: the target
        :param colorMap: color
        :return: scatter plot to show the class
        """
        X = self.projLDA(self.X_train, self.y_train)
        plt.figure()
        plt.subplot(1, 1, 1)
        plt.scatter(X[:, 0], X[:, 1], c=self.y_train, cmap=colorMap, s=100, alpha=0.9)
        plt.title('lda representation')
        plt.xlabel('coeff1')
        plt.ylabel('coeff2')
        plt.show()
