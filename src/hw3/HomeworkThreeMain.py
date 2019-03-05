from hw3.ClusterProcessor import ClusterProcessor
from hw3.DataParser import DataParser
from hw3.VoteProcessor import VoteProcessor
from hw3.datavisualizer import DataVisualizer


class HomeworkThreeMain:
    @classmethod
    def start(cls):
        parser = DataParser()
        restaurants = parser.get_restaurant()
        people = parser.get_people()
        result = VoteProcessor.process(restaurants, people)
        people_reduced = ClusterProcessor.init_pca(people.get_matrix())
        restaurant_reduced = ClusterProcessor.init_pca(restaurants.get_matrix())
        visualizer = DataVisualizer()
        visualizer.heat_map(result, parser.get_people(), parser.get_restaurant())
        DataVisualizer.visualize_cluster(people_reduced, people.get_names())
        DataVisualizer.visualize_cluster(restaurant_reduced, restaurants.get_names())
        DataVisualizer.visualize_hierarchical_cluster(people_reduced, people.get_names())
        DataVisualizer.visualize_hierarchical_cluster(restaurant_reduced, restaurants.get_names())


HomeworkThreeMain.start()
