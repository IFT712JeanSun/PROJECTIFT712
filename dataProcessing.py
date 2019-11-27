from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import decomposition
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")

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
        # the data_unknown are the data without target, the test data in
        data_unknown = read_csv(fileNameTest)
        X = data_train.iloc[:, 2:]
        le = LabelEncoder().fit(data_train.iloc[:, 1])
        y = le.transform(data_train.iloc[:, 1])
        self.className = list(le.classes_)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_unknown = data_unknown.iloc[:, 1:]
        self.id = data_unknown.iloc[:, 0]

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
        description = self.X_train.describe()
        return description

    def classDistribution(self):
        """
        function that gives the number of instance in each class
        :return: the species with their size
        """
        return self.X_train.groupby('species').size()

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

    def scatterPlot(self, X=None, y=None, colorMap='Paired'):
        """
        function for scatter plot.
        :param X: data to plot, the scaled data from scale() function
        :param y: the target
        :param colorMap: color
        :return: scatter plot to show the class
        """
        X = self.projLDA(X, y)
        plt.figure()
        plt.subplot(1, 1, 1)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=colorMap, s=100, alpha=0.9)
        plt.title('lda representation')
        plt.xlabel('coeff1')
        plt.ylabel('coeff2')
        plt.show()
