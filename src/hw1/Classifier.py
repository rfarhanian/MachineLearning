from hw1 import HyperParameterAttributes


class Classifier():

    def __init__(self, name, classifier_type, hyper_parameter_attributes: HyperParameterAttributes, title=""):
        self.name = name
        self.classifierType = classifier_type
        self.hyper_parameter_attributes = hyper_parameter_attributes
        self.title = title

    def desc(self):
        print(self.name)
        print(self.hyper_parameter_attributes)

    def get_name(self):
        return self.name

    def get_classifier_type(self):
        return self.classifierType

    def get_hyper_parameter_attributes(self):
        return self.hyper_parameter_attributes

    def get_title(self):
        return self.title

