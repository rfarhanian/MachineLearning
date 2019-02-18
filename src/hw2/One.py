import numpy as np


class One:
    @classmethod
    def count_column_values(cls, data, column_name, prefix):
        jay_starting_claims = np.flatnonzero(np.core.defchararray.find(data[column_name], prefix.encode()) == 1)
        Jcodes = data[jay_starting_claims]
        return len(Jcodes)

    @classmethod
    def in_network_price_paid(cls):
        return None
