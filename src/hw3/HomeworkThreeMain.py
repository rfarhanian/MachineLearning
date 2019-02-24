from hw3.DataParser import DataParser
from hw3.VoteProcessor import VoteProcessor
from hw3.datavisualizer import DataVisualizer


class HomeworkThreeMain:
    @classmethod
    def start(cls):
        parser = DataParser()
        result = VoteProcessor.process(parser.get_restaurant(), parser.get_restaurant())
        visualizer = DataVisualizer()
        visualizer.visualize(result, parser.get_people(), parser.get_restaurant())


HomeworkThreeMain.start()
