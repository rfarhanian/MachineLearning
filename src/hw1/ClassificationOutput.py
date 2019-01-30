from functools import reduce
from typing import List

from numpy import average


class ClassificationOutput:

    def __init__(self, classifier_name: str, title: str, response):
        self.accuracy = average([x['accuracy'] for x in response.values()])
        self.title = title
        self.classifier_name = classifier_name

    def str(self):
        print(self.classifier_name)
        # print(self.classification_report)
        print(self.title)

    def get_classifier_name(self):
        return self.classifier_name

    def get_title(self):
        return self.title

    # def get_classification_report(self):
    #     return self.classification_report

    def get_accuracy(self):
        return self.accuracy

    def get_hyperparam_set(self):
        return self.title
