from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from hw3.domain.ClusterMetricScore import ClusterMetricScore
from hw3.domain.PcaContext import PcaContext


class ClusterProcessor:

    @classmethod
    def init_pca(cls, matrix, n_components):
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(matrix)
        print(pca.components_)
        print(pca.explained_variance_)
        print(pca.explained_variance_ratio_)
        return PcaContext(reduced, pca.explained_variance_ratio_)

    @classmethod
    def process(cls, pca_reduced, cluster_collection, names: list):
        result = list()
        print(
            'Evaluating the performance of a clustering algorithm is not as trivial as counting the number of errors ')
        print(
            'or the precision and recall of a supervised classification algorithm. In particular any evaluation metric ')
        print('should not take the absolute values of the cluster labels into account but rather if this clustering ')
        print(
            'define separations of the data similar to some ground truth set of classes or satisfying some assumption ')
        print(
            'such that members belong to the same class are more similar that members of different classes according to some similarity metric.')
        for cluster_size in cluster_collection:
            result.extend(cls.__kmeans__(cluster_size, pca_reduced, names))
        result = sorted(result, key=lambda x: x.method, reverse=True)
        print('-----------------')
        print(
            'Calinski harabaz score is defined as ratio between the within-cluster dispersion and the between-cluster dispersion. Higher scores are better')
        print('-----------------')
        print(
            'Davies Bouldin score is defined as the ratio of within-cluster distances to between-cluster distances. Zero is the lowest possible score. Values closer to zero indicate a better partition.')
        print('-----------------')
        print(
            'Silhouette score computes the mean Silhouette Coefficient of all samples. The Silhouette Coefficient is calculated using the mean intra-cluster distance (a) and the mean nearest-cluster distance (b) for each sample.')
        print(
            'The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters. Negative values generally indicate that a sample has been assigned to the wrong cluster, as a different cluster is more similar.')
        print('-----------------')
        for i in range(len(result)):
            print(result[i].description())
        print('-----------------')
        return result

    @classmethod
    def __kmeans__(cls, cluster_size, pca_reduced, names: list):
        """
            The KMeans algorithm clusters data by trying to separate samples in n groups of equal variance,
            minimizing a criterion known as the inertia or within-cluster sum-of-squares. This algorithm requires
            the number of clusters to be specified. It scales well to large number of samples and has been used
            across a large range of application areas in many different fields.
        """
        import warnings
        warnings.filterwarnings("ignore")
        clusterer = KMeans(n_clusters=cluster_size, random_state=10)
        cluster_labels = clusterer.fit_predict(pca_reduced)
        result = list()
        result.append(ClusterProcessor.davies_bouldin(cluster_labels, pca_reduced, cluster_size, names))
        result.append(ClusterProcessor.variance_ratio_criterion(cluster_labels, pca_reduced, cluster_size, names))
        result.append(ClusterProcessor.silhouette_coefficient(cluster_labels, pca_reduced, cluster_size, names))
        return result

    @classmethod
    def silhouette_coefficient(cls, cluster_labels, pca_reduced, cluster_size, names: list):
        """
             If the ground truth labels are not known, evaluation must be performed using the model itself.
             The Silhouette Coefficient is an example of such an evaluation, where a higher Silhouette Coefficient
             score relates to a model with better defined clusters.
         """
        score = metrics.silhouette_score(pca_reduced, cluster_labels)
        return ClusterMetricScore('Silhouette score', score, cluster_size, cluster_labels, names)

    @classmethod
    def variance_ratio_criterion(cls, cluster_labels, pca_reduced, cluster_size, names: list):
        """
        If the ground truth labels are not known, the Calinski-Harabaz index (sklearn.metrics.calinski_harabaz_score) -
        also known as the Variance Ratio Criterion - can be used to evaluate the model, where a higher Calinski-Harabaz
        score relates to a model with better defined clusters.
        """
        score = metrics.calinski_harabaz_score(pca_reduced, cluster_labels)
        return ClusterMetricScore('calinski harabaz', score, cluster_size, cluster_labels, names)

    @classmethod
    def davies_bouldin(cls, cluster_labels, pca_reduced, cluster_size, names: list):
        """
        If the ground truth labels are not known, the Davies-Bouldin index (sklearn.metrics.davies_bouldin_score) can be
        used to evaluate the model, where a lower Davies-Bouldin index relates to a model with better separation
        between the clusters.
        """
        score = metrics.davies_bouldin_score(pca_reduced, cluster_labels)
        return ClusterMetricScore('davies_bouldin', score, cluster_size, cluster_labels, names)
