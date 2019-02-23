from numpy import average


class ClassificationOutput:

    def __init__(self, classifier_name: str, title: str, response, hyper_params: {}):
        self.accuracy = average([x['accuracy'] for x in response.values()])
        self.title = title
        self.classifier_name = classifier_name
        self.hyper_params = hyper_params

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

    def description(self):
        return '{' + 'classifier:' + self.classifier_name + ', title:' + self.title + ', accuracy:' + str(
            self.accuracy) + \
               ', hyper_params:' + str(self.hyper_params) + '}'
