import pandas as pd

from classifier import *


class Submission(Algorithm):
    '''
    class for creation and submission file
    '''
    def __init__(self):
        super().__init__()

    def submission(self):
        """
        compute the probability for a sample to be in a class and
        store the result in the file named submission.csv
        :return:
        """
        X_train = self.scale(self.data_X_train)
        X_unknown = self.scale(self.X_unknown)
        model = self.nn
        model.fit(X_train, self.data_Y_train)
        prediction = model.predict_proba(X_unknown)
        submission = pd.DataFrame(prediction, columns=self.className)
        submission.insert(0, 'id', self.id)
        submission.reset_index()
        return submission
