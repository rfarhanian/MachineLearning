class PcaContext:
    def __init__(self, reduced, explained_variance_ratio_):
        self.reduced = reduced
        self.explained_variance_ratio_ = explained_variance_ratio_

    def get_explained_variance_ratio(self):
        return self.explained_variance_ratio_

    def get_reduced(self):
        return self.reduced
