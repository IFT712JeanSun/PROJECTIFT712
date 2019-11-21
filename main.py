from training import *


def main():
    algo = train_testPrecision_Predict()
    print(' ***********      SANS TRANSFORMATION      **************')
    algo.train(transform=False)
    print(algo.testPrecision(transform=False))
    print('\n\n ***********   AVEC TRANSFORMATION      **************')
    algo.train(transform=True)
    print(algo.testPrecision(transform=True))
    print(algo.predict(model='LR'))
    print(algo.predict(model='SVM'))
    print(algo.predict(model='LDA'))
    print(algo.predict(model='DTC'))
    print(algo.predict(model='KNN'))
    print(algo.predict(model='NN'))

    # compare, results, names = algo.checkAlgorithm()
    # algo.boxplot()
    # subm = algo.submission()
    # print(subm.tail())


if __name__ == "__main__":
    main()
