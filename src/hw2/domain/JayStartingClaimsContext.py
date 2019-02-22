
class JayStartingClaimsContext:
    def __init__(self, claims, data):
        self.claims = claims
        self.codes = data[claims]
        self.size = len(claims)

    def get_claims(self):
        return self.claims

    def get_codes(self):
        return self.codes

    def get_size(self):
        return self.size
