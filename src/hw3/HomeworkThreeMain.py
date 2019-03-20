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
        visualizer.heat_map(result, restaurants, people)
        people_pca_context = ClusterProcessor.init_pca(people.get_matrix(), n_components=4)
        DataVisualizer.plot_elbow(people_pca_context, 'People PCA Elbow Plot')
        DataVisualizer.visualize_cluster(people_pca_context.get_reduced(), people.get_names())
        restaurant_pca_context = ClusterProcessor.init_pca(restaurants.get_matrix(), n_components=4)
        DataVisualizer.visualize_cluster(restaurant_pca_context.get_reduced(), restaurants.get_names())
        DataVisualizer.plot_elbow(restaurant_pca_context, 'Restaurant PCA Elbow Plot')
        print(
            'The elbow in the scree plots of both restaurant and people illustrate that the first 2 PCA components explain about 60 percent of the variance.')
        restaurant_pca_context = ClusterProcessor.init_pca(restaurants.get_matrix(), n_components=2)
        people_pca_context = ClusterProcessor.init_pca(people.get_matrix(), n_components=2)
        ClusterProcessor.process(restaurant_pca_context.get_reduced(), [2, 3, 4, 5, 6], restaurants.get_names())
        ClusterProcessor.process(people_pca_context.get_reduced(), [2, 3, 4, 5, 6], people.get_names())
        print(
            'A. Calinski Harabaz and Davies Bouldin scores suggest that people should go to restaurants in 6 groups, but this will send Jane to a restaurant alone which is not ideal.')
        print(
            'A. The next two best choices of Calinski Harabaz and Davies Bouldin(5 and 4) will lead to the same problem for Jane.')
        print(
            'A. The best choice that leads to groups larger than one is cluster size equal to 3 (which also gains the best silhouette scores) is  Group 1(Bob, Moe), Group 2(Mary, Mike, John, Kira, Tom), Group 3 (Jane, Alice, Sara)')
        print(
            'Jane has a problematic profile as her unique taste partitioned her into a cluster of size 1 several times')

        DataVisualizer.visualize_silhouette(restaurant_pca_context.get_reduced(), restaurants.get_names(), 'restaurant')
        DataVisualizer.visualize_silhouette(people_pca_context.get_reduced(), people.get_names(), 'people')

        DataVisualizer.visualize_cluster(people_pca_context.get_reduced(), people.get_names())
        DataVisualizer.visualize_hierarchical_cluster(people_pca_context.get_reduced(), people.get_names())
        DataVisualizer.visualize_hierarchical_cluster(restaurant_pca_context.get_reduced(), restaurants.get_names())

        VoteProcessor.process_without_cost_and_distance(restaurants, people)



HomeworkThreeMain.start()
