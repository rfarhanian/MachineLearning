import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import metrics
from sklearn.cluster import KMeans

from hw3.domain.PcaContext import PcaContext
from hw3.domain.People import People
from hw3.domain.Restaurant import Restaurant


class DataVisualizer:

    @classmethod
    def plot_elbow(cls, pca_context: PcaContext, title: str):
        plot_dims = (10, 8)
        fig, ax = plt.subplots(figsize=plot_dims)

        plt.plot(pca_context.get_explained_variance_ratio())
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.title(title)
        plt.show()

    @classmethod
    def draw_vector(cls, v0, v1, ax=None):
        ax = ax or plt.gca()
        arrowprops = dict(arrowstyle='->',
                          linewidth=2,
                          shrinkA=0, shrinkB=0)
        ax.annotate('', v1, v0, arrowprops=arrowprops)

    @classmethod
    def visualize_silhouette(cls, reduced, labels):
        # Silhouette Analysis with Kmeans Clustering on the PCA transformed data Matrix
        range_n_clusters = [2, 3, 4, 5, 6]

        for n_clusters in range_n_clusters:
            # Create a subplot with 1 row and 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(reduced) + (n_clusters + 1) * 10])

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(reduced)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = metrics.silhouette_score(reduced, cluster_labels)

            # Compute the silhouette scores for each sample
            sample_silhouette_values = metrics.silhouette_samples(reduced, cluster_labels)

            # The score is bounded between -1 for incorrect clustering and +1 for highly dense clustering.
            # Scores around zero indicate overlapping clusters.
            # The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.

            print("\n\n\nFor n_clusters =", n_clusters,
                  "\n\nThe average silhouette_score is :", silhouette_avg,
                  "\n\n* The silhouette score is bounded between -1 for incorrect clustering and +1 for highly dense clustering.",
                  "\n* Scores around zero indicate overlapping clusters.",
                  "\n* The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster",
                  "\n\nThe individual silhouette scores were :", sample_silhouette_values,
                  "\n\nAnd their assigned clusters were :", cluster_labels,
                  "\n\nWhich correspond to : ", labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.jet(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.9)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            # ax1.set_title("The silhouette plot for the various clusters.", fontsize=20)
            ax1.set_xlabel("The silhouette coefficient values", fontsize=20)
            ax1.set_ylabel("Cluster label", fontsize=20)

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            ax1.xaxis.set_tick_params(labelsize=20)
            ax1.yaxis.set_tick_params(labelsize=20)

            # 2nd Plot showing the actual clusters formed
            colors = cm.jet(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(reduced[:, 0], reduced[:, 1], marker='.', s=300,
                        lw=0, alpha=0.7,
                        c=colors, edgecolor='k')

            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                        c="white", alpha=1, s=400, edgecolor='k')

            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                            s=400, edgecolor='k')

            # ax2.set_title("The visualization of the clustered data.", fontsize=20)
            ax2.set_xlabel("Feature space for the 1st feature", fontsize=20)
            ax2.set_ylabel("Feature space for the 2nd feature", fontsize=20)

            plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                          "with n_clusters = %d" % n_clusters),
                         fontsize=25, fontweight='bold')

            ax2.xaxis.set_tick_params(labelsize=20)
            ax2.yaxis.set_tick_params(labelsize=20)

        plt.show()

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

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
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
        # Now lets try hierarchical clustering
        linked = linkage(matrix, 'single')

        # y=0 (flacos), y=1 (Joes), y=2 (Poke), y=3 (Sush-shi), y=4 (Chick Fillet),
        # y=5 (Mackie Des), y=6 (Michaels), y=7 (Amaze), y=8 (Kappa), y=9 (Mu)

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
