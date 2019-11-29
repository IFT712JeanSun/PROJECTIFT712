from sklearn.metrics import confusion_matrix, classification_report

from classifiers import *
from sklearn import metrics
from sklearn.utils.multiclass import unique_labels

class Plotting(Algorithm):
    def __init__(self):
        super().__init__()

    def evaluate_model(self, model):
        '''
        Evaluate the model
        :param model: the model or classifier
        :return: the accuracy score
        '''
        model.fit(self.X_train, self.y_train)
        y_predict = model.predict(self.X_test)
        accuracy_score = metrics.accuracy_score(self.y_test, y_predict)
        return accuracy_score

    def barPlot(self):
        '''
        Plot barplot of accuracy of each classifier.
        :return: None
        '''
        print("ploting the barplot ...\n")
        classifiers = self.classifier()
        accuracy_scores = list()
        for clf in classifiers:
            accuracy_score = self.evaluate_model(clf)
            accuracy_scores.append(accuracy_score)
        log_cols = ['Classifier', 'Accuracy score']
        log = pd.DataFrame(columns=log_cols)
        for i in range(0, len(classifiers)):
            log = log.append(
                pd.DataFrame([[classifiers[i].__class__.__name__, accuracy_scores[i]]], columns=log_cols), ignore_index=True)
        sns.barplot(x='Accuracy score', y='Classifier', data=log, color='black')
        plt.xlabel('Accuracy score')
        plt.title('Classifier accuracy score')
        plt.savefig("barplot.pdf")
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
            # ... and label them with the respective list entries
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
