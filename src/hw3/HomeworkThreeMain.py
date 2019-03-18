from hw3.ClusterProcessor import ClusterProcessor
from hw3.DataParser import DataParser
from hw3.Datavisualizer import DataVisualizer
from hw3.VoteProcessor import VoteProcessor


class HomeworkThreeMain:
    @classmethod
    def start(cls):
        parser = DataParser()
        restaurants = parser.get_restaurant()
        people = parser.get_people()
        result = VoteProcessor.process(restaurants, people)
        visualizer = DataVisualizer()
        visualizer.heat_map(result, parser.get_people(), parser.get_restaurant())
        people_pca_context = ClusterProcessor.init_pca(people.get_matrix(), n_components=4)
        DataVisualizer.plot_elbow(people_pca_context, 'People PCA Elbow Plot')
        DataVisualizer.visualize_cluster(people_pca_context.get_reduced(), people.get_names())
        restaurant_pca_context = ClusterProcessor.init_pca(restaurants.get_matrix(), n_components=4)
        DataVisualizer.visualize_cluster(restaurant_pca_context.get_reduced(), restaurants.get_names())
        DataVisualizer.plot_elbow(restaurant_pca_context, 'Restaurant PCA Elbow Plot')
        print(
            'The elbow in the scree plots of both restaurant and people illustrate that the first 2 PCA components explain most of the variance.')
        restaurant_pca_context = ClusterProcessor.init_pca(restaurants.get_matrix(), n_components=2)
        people_pca_context = ClusterProcessor.init_pca(people.get_matrix(), n_components=2)
        ClusterProcessor.kmeans_in_range(restaurant_pca_context.get_reduced(), [2, 3, 4, 5, 6])
        print(
            'Calinski Harabaz and Davies Bouldin scores suggest that restaurants can be categorized into 6 clusters. However, silouhette coeffcient suggests clusters of size 2.')
        ClusterProcessor.kmeans_in_range(people_pca_context.get_reduced(), [2, 3, 4, 5, 6])
        print(
            'Calinski Harabaz and Davies Bouldin scores again suggest that people should go to restaurants in groups of 6. However, silouhette coeffcient suggests groups of 3.')
        # DataVisualizer.visualize_hierarchical_cluster(people_pca_context.get_reduced(), people.get_names())
        # DataVisualizer.visualize_hierarchical_cluster(restaurant_pca_context.get_reduced(), restaurants.get_names())


HomeworkThreeMain.start()
