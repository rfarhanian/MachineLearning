import numpy as np

# A medical claim is denoted by a claim number ('Claim.Number'). Each claim consists of one or more medical lines
# denoted by a claim line number ('Claim.Line.Number').
from hw2.domain.JayStartingClaimsContext import JayStartingClaimsContext


class One:

    #  Find the number of claim lines that have J-codes.
    @classmethod
    def count_column_values(cls, data, column_name, prefix):
        jay_starting_claims = np.flatnonzero(np.core.defchararray.find(data[column_name], prefix.encode()) == 1)
        return JayStartingClaimsContext(jay_starting_claims, data)

    # How much was paid for J-codes to providers for 'in network' claims?
    @classmethod
    def in_network_price_paid(cls, data):
        with_jay_procedure_code = data[np.where((np.core.defchararray.find(data['ProcedureCode'], 'J'.encode()) == 1))]
        in_network_indices = with_jay_procedure_code[
            np.where(with_jay_procedure_code['InOutOfNetwork'] == '\"I\"'.encode())]
        return in_network_indices['ClaimChargeAmount'].sum()

    # What are the top five J-codes based on the payment to providers?
    @classmethod
    def top_five_jay_code_based_on_providers(cls, data):
        with_jay_procedure_code = data[np.where((np.core.defchararray.find(data['ProcedureCode'], 'J'.encode()) == 1))]
        sorted = np.sort(with_jay_procedure_code, order='ProviderPaymentAmount')
        sorted_Jcodes = sorted[::-1]
        print(sorted_Jcodes)
        topFive = sorted_Jcodes[:5]
        return topFive['ProcedureCode']
