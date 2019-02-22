class JayStartingClaimsContext:
    def __init__(self, jay_starting_claims, data):
        self.jay_starting_claims = jay_starting_claims
        self.jay_starting_codes = data[jay_starting_claims]
        self.size = len(jay_starting_claims)

    def get_jay_starting_claims(self):
        return self.jay_starting_claims

    def get_jay_starting_codes(self):
        return self.jay_starting_codes

    def get_size(self):
        return self.size
