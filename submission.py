from classifiers import *


class Submission(Algorithm):
    def __init__(self):
        print("initialize start.....")
        super().__init__()
        print("initialize finish.....")
    def submission(self):
        """
        compute the probability for a sample to be in a class
        :return:
        """
        self.data_X_train = self.scale(self.data_X_train)
        self.X_unknown = self.scale(self.X_unknown)

        model = self.nn
        model.fit(self.data_X_train, self.data_Y_train)
        prediction = model.predict_proba(self.X_unknown)
        #prediction = model.predict_proba(self.X_test)
        submission = pd.DataFrame(prediction, columns=self.className)
        submission.insert(0, 'id', self.id)
        #submission.insert(0, 'id', list(self.X_test.index))
        submission.reset_index()
        
        return submission
