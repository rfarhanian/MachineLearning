from itertools import product
from typing import List

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from hw1.ClassificationOutput import ClassificationOutput
from hw1.Classifier import Classifier
from hw1.HyperParameterAttributes import HyperParameterAttributes


# This class is responsible to contain each classifier with possible combinations of hyperparameter set.
# It is also responsible to generate all possible combinations
class Classifiers:
    def __init__(self):
        self.logisticRegressionClassifiers = Classifier('logistic Regression', LogisticRegression,
                                                        HyperParameterAttributes(
                                                            dict(penalty=['l1', 'l2'], dual=[False], tol=[0.0001],
                                                                 C=[1.0, 3.0, 5.0], fit_intercept=[True],
                                                                 intercept_scaling=[1], class_weight=[None],
                                                                 random_state=[None])))

        # n_estimator value does not have any impact in range (100, 200)
        # entropy criterion change did not have any impact on the accuracy of the model
        # increasing min_samples_split had a negative impact on the accuracy of the model
        # warm_start had no impact on the accuracy of the model
        self.randomForestClassifiers = Classifier('random Forest', RandomForestClassifier, HyperParameterAttributes(
            dict(n_estimators=['warn'], criterion=['gini', 'entropy'], max_depth=[4, None], min_samples_split=[2, 6],
                 max_features=['auto', 'sqrt'])))

        self.decisionTreeClassifier = Classifier("decision Tree", DecisionTreeClassifier, HyperParameterAttributes(
            dict(criterion=['gini', 'entropy'], splitter=['random'], max_depth=[1, 2], min_samples_split=[4, 2])))

        self.mlpClassifier = Classifier("MLP Classifier", MLPClassifier, HyperParameterAttributes(
            dict(activation=['identity', 'logistic', 'tanh', 'relu'], solver=['lbfgs', 'sgd', 'adam'])))

        self.perceptronClassifiers = Classifier('perceptron', Perceptron, HyperParameterAttributes(
            dict(penalty=[None], alpha=[0.001, 0.003, 0.0005], fit_intercept=[True, False], max_iter=[3, 12, 20])))

        self.gradientBoostingClassifiers = Classifier('gradient boosting', GradientBoostingClassifier, HyperParameterAttributes(
            dict(min_samples_split=[2,4], min_weight_fraction_leaf=[0.2, 0.3])))

        self.k_neighborsClassifiers = Classifier('K Neighbors', KNeighborsClassifier, HyperParameterAttributes(
            dict(algorithm=['auto', 'ball_tree', 'kd_tree'], n_neighbors=[5, 6, 7], n_jobs=[-1],
                 leaf_size=[25, 30, 40])))

    # This method returns all possible classifiers with all combinations of hyperparameter sets.
    def all_classifiers(self):
        parameters = []
        parameters += Classifiers.to_list(self.decisionTreeClassifier)
        parameters += Classifiers.to_list(self.mlpClassifier)
        parameters += Classifiers.to_list(self.perceptronClassifiers)
        parameters += Classifiers.to_list(self.randomForestClassifiers)
        parameters += Classifiers.to_list(self.logisticRegressionClassifiers)
        parameters += Classifiers.to_list(self.gradientBoostingClassifiers)
        parameters += Classifiers.to_list(self.k_neighborsClassifiers)
        return parameters

    # This method transforms a generic classifier object with possible combinations of hyper-parameter configuration
    # into a set of a classifiers.
    @classmethod
    def to_list(cls, classifier: Classifier):
        result = []
        name = classifier.get_name()
        attributes = classifier.hyper_parameter_attributes.get_attributes()
        key, value = zip(*attributes.items())
        for values in product(*value):
            hyperset = dict(zip(key, values))
            result.append(Classifier(name, classifier.classifierType, HyperParameterAttributes(hyperset)))
        return result

    # This method visualizes the output of classification for all possible accuracy values. It also serializes
    # the file into file system. I tried to visualize the data using a "BoxPlot" but I could not make it work.
    @classmethod
    def visualize(cls, report: List[ClassificationOutput], title: str, plot_name: str):
        fig = plt.figure()
        for item in report:
            plt.scatter(item.get_classifier_name(), item.get_accuracy(), label=item.get_classifier_name())

        plt.xlabel('Classification Algorithms', multialignment='center')
        plt.xticks(rotation=25, horizontalalignment='right', fontsize=6)
        plt.ylabel('accuracy')
        fig.suptitle(title)
        plt.show()
        plt.savefig(('./result/' + plot_name))

    # This method visualizes the accuracy of the average of fold accuracy per classifier
    @classmethod
    def visualize_with_histogram(cls, report: List[ClassificationOutput]):
        plt.clf()
        report_dictionary = dict()

        for reportItem in report:
            if reportItem.get_classifier_name() not in report_dictionary:
                report_dictionary[reportItem.get_classifier_name()] = []
            report_dictionary[reportItem.get_classifier_name()].append(reportItem)

        for key, value in report_dictionary.items():
            classifier_name = key
            x = list(map(lambda x: x.get_accuracy(), value))
            num_bins = len(report)
            n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=1, ec='black')
            plt.xlabel('Accuracy')
            plt.ylabel('Frequency')
            plt.title(classifier_name)
            plt.subplots_adjust(left=0.15)
            plt.show()
            plt.savefig(('./result/' + classifier_name + ".png"))
