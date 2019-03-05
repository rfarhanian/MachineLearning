import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans

from hw3.domain.People import People
from hw3.domain.Restaurant import Restaurant


class DataVisualizer:

    @classmethod
    def draw_vector(cls, v0, v1, ax=None):
        ax = ax or plt.gca()
        arrowprops = dict(arrowstyle='->',
                          linewidth=2,
                          shrinkA=0, shrinkB=0)
        ax.annotate('', v1, v0, arrowprops=arrowprops)

    @classmethod
    def visualize_cluster(cls, matrix, labelList):
        # we don't need to apply standard scaler since the data is already scaled
        # sc = StandardScaler()
        # peopleMatrixScaled = sc.fit_transform(peopleMatrix)

        # The example PCA was taken from.
        # https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
        # peopleMatrixPcaTransform = cls.init_pca(people.get_matrix())

        # This function was shamefully taken from the below and modified for our purposes
        # https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
        # plot principal components

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

        ax.scatter(matrix[:, 0], matrix[:, 1], alpha=0.2)
        cls.draw_vector([0, 0], [0, 1], ax=ax)
        cls.draw_vector([0, 0], [1, 0], ax=ax)
        ax.axis('equal')
        ax.set(xlabel='component 1', ylabel='component 2', title='principal components', xlim=(-1.5, 1.5),
               ylim=(-1.5, 1.5))
        fig.show

        # Now use peoplePCA for clustering and plotting
        # https://scikit-learn.org/stable/modules/clustering.html
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(matrix)

        centroid = kmeans.cluster_centers_
        labels = kmeans.labels_

        print(centroid)
        print(labels)

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

        # https://matplotlib.org/users/colors.html
        colors = ["g.", "r.", "c."]

        for i in range(len(matrix)):
            print("coordinate:", matrix[i], "label:", labels[i])
            ax.plot(matrix[i][0], matrix[i][1], colors[labels[i]], markersize=10)
            # https://matplotlib.org/users/annotations_intro.html
            # https://matplotlib.org/users/text_intro.html
            ax.annotate(labelList[i], (matrix[i][0], matrix[i][1]), size=25)
        ax.scatter(centroid[:, 0], centroid[:, 1], marker="x", s=150, linewidths=5, zorder=10)

        plt.show()

        # cluster 0 is green, cluster 1 is red, cluster 2 is cyan (blue)

        # remember, that the order here is:

        # x=0 (Jane), x=1 (Bob), x=2 (Mary), x=3 (Mike), x=4 (Alice),
        # x=5 (Skip), x=6 (Kira), x=7 (Moe), x=8 (Sara), x=9 (Tom)

    def heat_map(self, results, people: People, restaurants: Restaurant):
        # first plot heatmap
        # https://seaborn.pydata.org/generated/seaborn.heatmap.html
        plot_dims = (12, 10)
        fig, ax = plt.subplots(figsize=plot_dims)
        sns.heatmap(ax=ax, data=results, annot=True)
        plt.show()

        # remember a_ij is the score for a restaurant for a person
        # x is the person, y is the restaurant

        print(people.get_keys())
        # x=0 (Jane), x=1 (Bob), x=2 (Mary), x=3 (Mike), x=4 (Alice),
        # x=5 (Skip), x=6 (Kira), x=7 (Moe), x=8 (Sara), x=9 (Tom)

        print(restaurants.get_keys())
        # y=0 (flacos), y=1 (Joes), y=2 (Poke), y=3 (Sush-shi), y=4 (Chick Fillet),
        # y=5 (Mackie Des), y=6 (Michaels), y=7 (Amaze), y=8 (Kappa), y=9 (Mu)

        # What is the problem if we want to do clustering with this matrix?

        results.shape

        # from sklearn.preprocessing import StandardScaler
        # from sklearn.decomposition import PCA

        people.get_matrix().shape

    @classmethod
    def visualize_hierarchical_cluster(cls, matrix, labelList):
        # Now lets try heirarchical clustering
        linked = linkage(matrix, 'single')

        # y=0 (flacos), y=1 (Joes), y=2 (Poke), y=3 (Sush-shi), y=4 (Chick Fillet),
        # y=5 (Mackie Des), y=6 (Michaels), y=7 (Amaze), y=8 (Kappa), y=9 (Mu)

        # labelList = ['Flacos', 'Joes', 'Poke', 'Sush-shi', 'Chick Fillet', 'Mackie Des', 'Michaels', 'Amaze', 'Kappa', 'Mu']

        fig = plt.figure(figsize=(30, 15))
        ax = fig.add_subplot(1, 1, 1)
        dendrogram(linked,
                   orientation='top',
                   labels=labelList,
                   distance_sort='descending',
                   show_leaf_counts=True, ax=ax)
        ax.tick_params(axis='x', which='major', labelsize=25)
        ax.tick_params(axis='y', which='major', labelsize=25)
        plt.show()
