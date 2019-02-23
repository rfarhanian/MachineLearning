from typing import Type

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold

from hw1.ClassificationOutput import ClassificationOutput
from hw1.Classifiers import Classifiers


class ClassifiersExecutor:

    def execute(self, data):
        classifiers = Classifiers()
        result = self.__run_all__(classifiers.all_classifiers(), data)
        Classifiers.visualize(result, 'Accuracy Of Classifiers', 'classificationResult.png')

    def __run__(a_clf: Type, data, clf_hyper={}):
        M, L, n_folds = data  # unpack data container
        kf = KFold(n_splits=n_folds)  # Establish the cross validation
        ret = {}  # classic explication of results

        for ids, (train_index, test_index) in enumerate(kf.split(M, L)):
            clf = a_clf(**clf_hyper)  # unpack parameters into clf is they exist

            clf.fit(M[train_index], L[train_index])

            pred = clf.predict(M[test_index])

            ret[ids] = {'clf': clf,
                        'train_index': train_index,
                        'test_index': test_index,
                        'accuracy': accuracy_score(L[test_index], pred),
                        'classificationReport': classification_report(L[test_index], pred)
                        }
        return ret

    def __run_all__(self, classifiers, data):
        result = []
        for classifier in classifiers:
            classifier_name = classifier.get_name()
            classifier_type = classifier.get_classifier_type()
            response = ClassifiersExecutor.__run__(classifier_type, data,
                                                   classifier.get_hyper_parameter_attributes().get_attributes())
            result.append(ClassificationOutput(classifier_name, classifier.get_title(), response))
        return result
