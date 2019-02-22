from hw2.ClaimParser import ClaimParser
from hw2.One import One
from hw2.Three import Three
from hw2.Two import Two

parser = ClaimParser("./input/claim.sample.csv")
jay_starting_claims_context = One.count_column_values(parser.get_rows(), column_name='ProcedureCode', prefix='J')
print("One A result:", jay_starting_claims_context.get_size())
print("One B result:", One.in_network_price_paid(parser.get_rows()))
provider_sorted_jay_codes = One.top_jay_code_based_on_providers(jay_starting_claims_context)
print("One C result:", provider_sorted_jay_codes.get_top_procedure_codes(5))

two = Two('./result/', 'HW2-Result')
two.a(parser.get_rows(), jay_starting_claims_context.get_codes())
two.b()
two.c()
two.finalize()

three = Three(provider_sorted_jay_codes)
print('Three A: What percentage of J-code claim lines were unpaid?', three.a())
print('Three B:', three.b())
