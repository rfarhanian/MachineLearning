class ProviderSortedJayCodes:

    def __init__(self, sorted_jay_codes):
        self.sorted_jay_codes = sorted_jay_codes

    def get_jay_codes(self):
        return self.sorted_jay_codes

    def get_top_procedure_codes(self, n):
        return self.sorted_jay_codes[:n]['ProcedureCode']
