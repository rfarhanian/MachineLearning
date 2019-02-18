from hw2.ClaimParser import ClaimParser
from hw2.One import One

parser = ClaimParser("./input/claim.sample.csv")
print("One A result:", One.count_column_values(parser.get_rows(), column_name='ProcedureCode', prefix='J'))
print("One B result:", One.in_network_price_paid())
