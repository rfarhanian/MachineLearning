from hw2.ClaimParser import ClaimParser
from hw2.One import One
from hw2.Two import Two

parser = ClaimParser("./input/claim.sample.csv")
jay_starting_claims_context = One.count_column_values(parser.get_rows(), column_name='ProcedureCode', prefix='J')
print("One A result:", jay_starting_claims_context.get_size())
print("One B result:", One.in_network_price_paid(parser.get_rows()))
print("One C result:", One.top_five_jay_code_based_on_providers(parser.get_rows()))

two = Two('./output/', 'HW2-Result')
two.a(parser.get_rows(), jay_starting_claims_context.get_jay_starting_codes())
two.b()
two.c()
