from hw3.DataParser import DataParser
from hw3.datavisualizer import DataVisualizer


class Main:
    @classmethod
    def start(cls):
        parser = DataParser()
        visualizer = DataVisualizer
        visualizer.visualize(results=parser.get_result(), people=parser.get_people(),
                             restaurants=parser.get_restaurant())


Main.start()
