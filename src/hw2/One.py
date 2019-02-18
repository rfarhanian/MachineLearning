import numpy as np


class One:
    @classmethod
    def count_column_values(cls, data, column_name, prefix):
        Jcodeclaims = np.flatnonzero(np.core.defchararray.find(data[column_name], prefix.encode()) == 1)
        Jcodes = data[Jcodeclaims]
        return len(Jcodes)
