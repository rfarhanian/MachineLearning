class JayProviderContext:
    def __init__(self, paidProviders):
        self.paidProviders = paidProviders
        self.size = len(paidProviders)

    def get_paid_providers(self):
        return self.paidProviders

    def get_size(self):
        return self.size
