from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import decomposition
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
import warnings
warnings.filterwarnings("ignore")
from pandas import set_option

class Data(object):
    '''
    Class for preparing the problem:
    1) all the needed library
    2) loading all the data to use
    '''
    def __init__(self, fileNameTrain='data/train.csv', fileNameTest='data/test.csv', n_component=10):
        """
        Importation of the leaf data.
        :param fileNameTrain: this is the training data, it will be separating for real training and validation
        :param fileNameTest: the data testing
        :param n_component:
        """
        # The data_train will be separated in train and test data
        data_train = read_csv(fileNameTrain)
        self.data_to_print = data_train
        # the data_unknown are the data without target, the test data in
        data_unknown = read_csv(fileNameTest)
        self.data_X_train = data_train.iloc[:, 2:]
        X = data_train.iloc[:, 2:]
        y = data_train['species'].astype('category')
        y = y.cat.codes.as_matrix()
        self.data_Y_train = y
        sss = StratifiedShuffleSplit(10, 0.2, random_state=15)
        for train_index, test_index in sss.split(X, y):
            self.X_train, self.X_test = X.iloc[train_index], X.iloc[test_index]
            self.y_train, self.y_test = y[train_index], y[test_index]
        le = LabelEncoder().fit(data_train.iloc[:, 1])
        self.className = list(le.classes_)
        submission_data = pd.read_csv('data/sample_submission.csv')
        categories = submission_data.columns.values[1:]
        self.X_unknown = data_unknown.iloc[:, 1:]
        self.id = data_unknown.iloc[:, 0]



    def printSomeData(self):
        '''
        function that print some data
        :return:
        '''
        printing = self.data_to_print[['id', 'species', 'margin20', 'shape20', 'texture20']]
        print(printing)

    def getShape(self):
        '''
        Function that print the shape of the data
        :return:
        '''
        print(self.data_to_print.shape)


    def getData(self, showData=False, n=5):
        """
        Function to get the data if needed
        :param n: the number of rows to be displayed
        :param showData: if showData is True, the data will be printed on screen
        :return: self.X_train, self.y_train, self.X_test, self.y_test
        """
        if showData:
            print('data_train:\n ')
            print(self.X_train.head(n))
            print('\n\n')
            print('data_test:\n')
            print(self.y_train.head(n))
        return self.X_train, self.y_train, self.X_test, self.y_test

    def dataDescription(self):
        """
        function for the statistic description of the data
        This function can suggest us to stardardize our data when the mean are differents
        :return: the described data: mean std etc.
        """
        x = self.X_train[['margin20', 'shape20', 'texture20']]
        description = x.describe()
        return description

    def classDistribution(self):
        """
        function that gives the number of instance in each class
        :return: the species with their size
        """
        return self.data_to_print.groupby('species').size()

    def univariantePlot(self):
        """
        since some of the algo suppose data follow gaussian distribution
        we use histogramme to see if some features follow in average gaussian distribution
        :return: histrogram figure
        """
        data = self.X_train.loc[:, 'texture45']
        sns.distplot(data, hist=True, kde=True, bins=int( len(data)/20), color='darkblue',
                     hist_kws={'edgecolor': 'black'},kde_kws={'linewidth': 4})
        pyplot.show()

    def correlationMatrix(self):
        """
        function for testing the correlation between features
        :return: correlation matrix as figure
        """
        fig = pyplot.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(self.X_train.corr(), vmin=-1, vmax=1, interpolation='none')
        fig.colorbar(cax)
        pyplot.show()

    def correlation(self):
        set_option('precision', 3)
        x_corr = self.X_train[['margin10', 'shape10', 'texture10', 'margin20', 'shape20', 'texture20']]
        corr = x_corr.corr(method='pearson')
        print(corr)

    def multivariantePlot(self):
        """
        the multivariante case to see if there is correlation between feaction
        :return: the scatter matrix plot
        """
        shape = self.X_train.loc[:, 'texture1':'texture5']
        scatter_matrix(shape)
        pyplot.show()
        #pyplot.savefig('shape1-5.pdf')

    def pca(self, X, n_components=10):
        """
        function for reducing the dimension for scatter plot. Does not work well
        :param X: the data to trnasform
        :param n_components: the number of component to save
        :return: the fitted data in n_components dimension
        """
        pcaModel = decomposition.PCA(n_components=n_components, whiten=True)
        return pcaModel.fit_transform(X)

    def projLDA(self, X, y, n_components=2):
        """
        function for reducing the dimension for scatter plot. Does not work well
        :param X: the data to trnasform
        :param y: the target
        :param n_components: the number of component to save
        :return: the fitted data in n_components dimension
        """
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        lda.fit(X, y)
        return lda.transform(X)

    def scale(self, X):
        """
        function that scale the data between [0,1]
        :param X: the data to scale
        :return: the scaled data
        """
        min_max_scaler = preprocessing.MinMaxScaler()
        X_transform = min_max_scaler.fit_transform(X)
        return X_transform





