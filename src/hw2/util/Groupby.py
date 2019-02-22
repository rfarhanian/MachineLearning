import numpy as np


class Groupby:
    def __init__(self, keys):
        _, self.keys_as_int = np.unique(keys, return_inverse=True)
        self.n_keys = max(self.keys_as_int)
        self.set_indices()

    def set_indices(self):
        self.indices = [[] for i in range(self.n_keys + 1)]
        for i, k in enumerate(self.keys_as_int):
            self.indices[k].append(i)
        self.indices = [np.array(elt) for elt in self.indices]

    def apply(self, function, vector, broadcast):
        if broadcast:
            result = np.zeros(len(vector))
            for idx in self.indices:
                result[idx] = function(vector[idx])
        else:
            result = np.zeros(self.n_keys)
            for k, idx in enumerate(self.indices):
                result[self.keys_as_int[k]] = function(vector[idx])

        return result
