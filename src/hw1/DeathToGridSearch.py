from typing import Type

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from hw1.ClassificationOutput import ClassificationOutput
from hw1.Classifiers import Classifiers
from hw1.MyGridSearch import MyGridSearch

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

classifier_dictionary = {'RandomForestClassifier': RandomForestClassifier(),
                         'LogisticRegression': LogisticRegression(),
                         'MLPClassifier': MLPClassifier(),
                         'GradientBoostingClassifier': GradientBoostingClassifier(),
                         'KNeighborsClassifier': KNeighborsClassifier(),
                         }

hyperparameter_dictionary = {
    'RandomForestClassifier': {"criterion": ['gini', 'entropy'], "splitter": ['random'],
                               "max_depth": [1, 2], "min_samples_split": [4, 2]},

    'LogisticRegression': {"tol": [0.0001], "C": [1, 3, 5], "penalty": ['l1', 'l2'], "dual": [False],
                           "fit_intercept": [True]},
    'KNeighborsClassifier': {
        "algorithm": ['auto', 'ball_tree', 'kd_tree'], "n_neighbors": [5, 6, 7],
        "n_jobs": [-1],
        "leaf_size": [25, 30, 40]}
}
gridSearch = MyGridSearch(classifier_dictionary, hyperparameter_dictionary)
gridSearch.fit(M, L, scoring='accuracy', n_jobs=2)
gridSearch.score_summary(sort_by='max_score')
