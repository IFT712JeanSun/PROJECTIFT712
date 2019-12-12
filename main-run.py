from plotting import *
from submission import *
from train import train_testPrecision_Predict

# To run the code python main-run.py

def main():
    '''
    This is the main function for running the code
    :return:
    '''
    data = Data()
    print("\nPrinting some data for quick observation ...\n")
    data.printSomeData()

    print("\nPrinting the shape of the data ...\n")
    data.getShape()

    print("\n Printing a quick description of the data ...\n")
    print(data.dataDescription())

    print("\n Printing the Class distribution ...\n")
    print(data.classDistribution())

    print("\n Printing the Correlation between variables ...\n")
    print(data.correlation())

    print(' \n\n Training without data transformation ...\n\n')
    algo = train_testPrecision_Predict()
    algo.train(transform=False)
    print(algo.testPrecision(transform=False))

    print('\n\n Training with data transformation ... \n\n')
    algo.train(transform=True)
    print(algo.testPrecision(transform=True))


    print("\n\n Accuracy with data transformation ... \n\n")
    print(algo.testPrecision(transform=True))

    print("\n\n Cross validation accuracy without data transformation ... \n\n")
    algo.crossValidationResults(transform=False)

    print("\n\n Cross validation accuracy with data transformation ... \n\n")
    algo.crossValidationResults(transform=True)

    print("\n\n Prediction of some classifiers on validation data ... \n\n")
    print(algo.predict(model='LR'))
    print(algo.predict(model='SVM'))
    print(algo.predict(model='LDA'))
    print(algo.predict(model='DTC'))
    print(algo.predict(model='KNN'))
    print(algo.predict(model='NN'))

    print("\n\n  Making plots for the report ... \n\n")
    toplot = Plotting()
    toplot.barPlot()
    toplot.scatterPlot()
    toplot.barPlotCrossValidation()

    print('\n\n Making file for submission ...\n')
    s = Submission()
    sub = s.submission()
    sub.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    main()
