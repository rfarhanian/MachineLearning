import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from hw3.domain.People import People


class PrincipleComponent:
    def __init__(self, people: People):
        # we don't need to apply standard scaler since the data is already scaled
        # sc = StandardScaler()
        # peopleMatrixScaled = sc.fit_transform(peopleMatrix)

        # The example PCA was taken from.
        # https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
        pca = PCA(n_components=2)
        self.peopleMatrixPcaTransform = pca.fit_transform(people.get_matrix())

        print(pca.components_)
        print(pca.explained_variance_)

    # This function was shamefully taken from the below and modified for our purposes
    # https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
    # plot principal components
    def draw_vector(v0, v1, ax=None):
        ax = ax or plt.gca()
        arrowprops = dict(arrowstyle='->',
                          linewidth=2,
                          shrinkA=0, shrinkB=0)
        ax.annotate('', v1, v0, arrowprops=arrowprops)

    def draw(self):
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

        ax.scatter(self.peopleMatrixPcaTransform[:, 0], self.peopleMatrixPcaTransform[:, 1], alpha=0.2)
        PrincipleComponent.draw_vector([0, 0], [0, 1], ax=ax)
        PrincipleComponent.draw_vector([0, 0], [1, 0], ax=ax)
        ax.axis('equal')
        ax.set(xlabel='component 1', ylabel='component 2',
               title='principal components',
               xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
        fig.show
