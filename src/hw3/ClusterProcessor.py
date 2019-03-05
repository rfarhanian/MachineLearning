from sklearn.decomposition import PCA


class ClusterProcessor:

    @classmethod
    def init_pca(cls, matrix):
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(matrix)
        print(pca.components_)
        print(pca.explained_variance_)
        return reduced
