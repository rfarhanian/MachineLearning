from typing import Type

from sklearn import datasets
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold

from hw1.ClassificationOutput import ClassificationOutput
from hw1.HyperParameterAttributes import HyperParameterAttributes
from hw1.Classifiers import Classifiers

iris = datasets.load_iris()
data = (iris['data'], iris['target'], 5)


def run(a_clf: Type, data, clf_hyper={}):
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


def run_all(cls, data):
    result = []
    for classifier in cls:
        classifier_name = classifier.get_name()
        classifier_type = classifier.get_classifier_type()
        response = run(classifier_type, data, classifier.get_hyper_parameter_attributes().get_attributes())
        result.append(ClassificationOutput(classifier_name, classifier.get_title(), response))
    return result


classifiers = Classifiers()
result = run_all(classifiers.all_classifiers(), data)
Classifiers.visualize(result, 'Accuracy Of Classifiers', 'classificationResult.png')
Classifiers.visualize_with_histogram(result)
