class Restaurant:
    def __init__(self, matrix, keys, values, names):
        self.matrix = matrix
        self.keys = keys
        self.values = values
        self.names = names

    def get_matrix(self):
        return self.matrix

    def get_keys(self):
        return self.keys

    def get_values(self):
        return self.values

    def get_names(self):
        return self.names
