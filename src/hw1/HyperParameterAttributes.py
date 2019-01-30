from typing import Dict


class HyperParameterAttributes:
    def __init__(self, attributes: Dict, title=""):
        self.attributes = attributes
        self.title = title

    def get_attributes(self):
        return self.attributes

    def get_title(self):
        return self.title
