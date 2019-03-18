class ClusterMetricScore:
    def __init__(self, method, score, cluster):
        self.method = method
        self.score = score
        self.cluster = cluster

    def description(self):
        return str(self.method) + ' score for cluster of size ' + str(self.cluster) + ' is : ' + str(self.score)
