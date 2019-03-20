class ClusterMetricScore:
    def __init__(self, method, score, cluster, labels, names: list):
        self.method = method
        self.score = score
        self.cluster = cluster
        self.labels = labels
        self.names = names

    def description(self):
        return str(self.method) + ' score for cluster of size ' + str(self.cluster) + ' is : ' + str(self.score) + \
               str(self.labels) + ' which corresponds to ' + str(self.names)
