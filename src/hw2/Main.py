from hw2.ClaimParser import ClaimParser
from hw2.One import One

parser = ClaimParser("./input/claim.sample.csv")
result = One.get_procedure_code(parser.get_rows())
print(result)
# print(parser.__str__())
# print(parser.get_rows())
