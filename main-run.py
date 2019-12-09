from plotting import *
from submission import *
from training import train_testPrecision_Predict


def main():
    data = Data()
    print("\n**********  Printing some data for quick observation ************\n")
    data.printSomeData()
    print("\n**********  Printing the shape of the data ************\n")
    data.getShape()

    print("\n********** Printing a quick description of the data *******\n")
    print(data.dataDescription())

    algo = train_testPrecision_Predict()
    print(' ***********      SANS TRANSFORMATION      **************')
    algo.train(transform=False)
    print(algo.testPrecision(transform=False))
    print('\n\n ***********   AVEC TRANSFORMATION      **************')
    algo.train(transform=True)
    print(algo.testPrecision(transform=True))

    #print(algo.predict(model='LR'))
    #print(algo.predict(model='SVM'))
    #print(algo.predict(model='LDA'))
    #print(algo.predict(model='DTC'))
    #print(algo.predict(model='KNN'))
    #print(algo.predict(model='NN'))

    # Also we can add ensemble here

    toplot = Plotting()
    toplot.barPlot()
    toplot.scatterPlot()

    print('\n\n ***********  Printing the submission  **************')
    sub = Submission()
    print(sub.submission())

if __name__ == "__main__":
    main()
