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
        people_reduced = ClusterProcessor.init_pca(people.get_matrix())
        restaurant_reduced = ClusterProcessor.init_pca(restaurants.get_matrix())
        DataVisualizer.visualize_cluster(people_reduced, people.get_names())
        DataVisualizer.visualize_cluster(restaurant_reduced, restaurants.get_names())
        DataVisualizer.visualize_hierarchical_cluster(people_reduced, people.get_names())
        DataVisualizer.visualize_hierarchical_cluster(restaurant_reduced, restaurants.get_names())


HomeworkThreeMain.start()
