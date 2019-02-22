import matplotlib.pyplot as plt
import numpy as np
import numpy_groupies as npg
from matplotlib.backends.backend_pdf import PdfPages
from numpy.lib.recfunctions import append_fields

from hw2.domain.PaidProviderContext import PaidProviderContext


# For the following exercises, determine the number of providers that were paid for at least one J-code.
# Use the J-code claims for these providers to complete the following exercises.
class Two:

    def __init__(self, path, plot_file_name):
        pdf_file_name = path + plot_file_name + '.pdf'
        self.pdf = PdfPages(pdf_file_name)

    def get_jay_code_paid_providers(cls, data):
        with_jay_procedure_code = data[np.where((np.core.defchararray.find(data['ProcedureCode'], 'J'.encode()) == 1))]
        return PaidProviderContext(np.unique(
            with_jay_procedure_code[np.where(with_jay_procedure_code['ProviderPaymentAmount'] > 0)]['ProviderID']))

    # Create a scatter plot that displays the number of unpaid claims
    # (lines where the ‘Provider.Payment.Amount’ field is equal to zero) for each provider versus the number of
    # paid claims.
    def a(self, data, all_jay_codes):
        paid_provider_context = self.get_jay_code_paid_providers(data)
        providers_sum_context = self.__resolve_provider_sum_context__(all_jay_codes, paid_provider_context)
        self.__first_scatter_plot__(providers_sum_context)
        self.__second_scatter_plot__(providers_sum_context)

    def __resolve_provider_sum_context__(self, all_jay_codes, paid_provider_context: PaidProviderContext):
        paid_providers = paid_provider_context.get_paid_providers()
        paid_jay_codes = all_jay_codes[np.where(np.isin(all_jay_codes['ProviderID'], paid_providers))]
        enriched_paid_jay_codes = self.__addColumns__(paid_jay_codes)
        provider_index_map = {id: indi for indi, id in enumerate(set(enriched_paid_jay_codes['ProviderID']))}
        provider_ids = [provider_index_map[id] for id in enriched_paid_jay_codes['ProviderID']]
        unpaid_providers_sum_mapping = npg.aggregate(provider_ids, enriched_paid_jay_codes['isNotPaidCount'],
                                                     func='sum')
        paid_providers_sum_mapping = npg.aggregate(provider_ids, enriched_paid_jay_codes['isPaidCount'], func='sum')
        total = paid_providers_sum_mapping + unpaid_providers_sum_mapping
        provider_sum_context = zip(provider_index_map.keys(), paid_providers_sum_mapping,
                                   unpaid_providers_sum_mapping, total, paid_providers_sum_mapping / total)
        provider_sum_context = sorted(provider_sum_context, key=lambda x: x[3], reverse=True)
        return provider_sum_context

    def __addColumns__(self, paid_jay_codes):
        paid_jay_codes = append_fields(paid_jay_codes, 'isPaid', paid_jay_codes['ProviderPaymentAmount'] > 0)
        paid_jay_codes = append_fields(paid_jay_codes, 'isNotPaidCount',
                                       (paid_jay_codes['ProviderPaymentAmount'] == 0) + 0)
        paid_jay_codes = append_fields(paid_jay_codes, 'isPaidCount', (paid_jay_codes['ProviderPaymentAmount'] > 0) + 0)
        return paid_jay_codes

    def __first_scatter_plot__(self, provider_sums):
        fig, ax = plt.subplots()
        fig.set_size_inches(13, 6)
        for provider in provider_sums:
            ax.scatter(provider[1], provider[2], label='Provider:' + provider[0].decode(), edgecolors='none')
            ax.text(provider[1], provider[2], provider[0].decode(), fontsize=6)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(True)
        plt.title('First Scatterplot - Claims Paid by Provider')
        plt.xlabel('# of Paid Claims')
        plt.ylabel('# of UnPaid Claims')
        self.pdf.savefig()
        plt.show()

    def __second_scatter_plot__(self, paid_provider_sums):
        fig, ax = plt.subplots()
        fig.set_size_inches(13, 6)
        for provider in paid_provider_sums:
            ax.scatter(provider[3], provider[4], label='Provider:' + provider[0].decode(), edgecolors='none')
            ax.text(provider[3], provider[4], provider[0].decode(), fontsize=6)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(True)
        plt.title('Second Scatterplot - The Percentage of Claims Paid by Provider')
        plt.xlabel('Total Claims')
        plt.ylabel('Percentage of Total Claims Paid')
        self.pdf.savefig()
        plt.show()

    # What insights can you suggest from the graph?
    def b(self):
        print('----------------------------------------------------------------------')
        print('2B. What insights can you suggest from the graph?')
        print('FA0004551001 , FA1000014001 , FA1000015001 seem to be small clinics with small number of clients who ')
        print('have very straight forward procedures.')
        print('FA1000016001 , FA1000014002 , FA1000015002 , FA0001411003, FA1000014001 seem to be small clinics ')
        print('with very few patients. ')
        print('They seem to be working on procedures that are not fully supported by the insurance companies. ')
        print('Such procedures might be plastic surgery or very rare unorthodox treatments for rare diseases.')
        print('FA0001411001 , FA0001387001 , FA0001387002 , FA0001389001 seem to be large hospitals with a ')
        print(' large set of clients with certain diagnosis ')
        print('that is not well supported by insurance companies. Cosmetic surgeries or Lasik operation ')
        print('might are such well known operations. ')
        print('These hospitals seem to have either high quality')
        print('or located in affluent neighborhoods')

    # Based on the graph, is the behavior of any of the providers concerning? Explain.
    def c(self):
        print('----------------------------------------------------------------------')
        print('2C. Based on the graph, is the behavior of any of the providers concerning? Explain.')
        print('Observing all the graphs reveals that rejected claims might be ')
        print('correlated with extra or unsupported services. Fraud or quality of ')
        print('service cannot be directly correlated with rejected claims unless more information')
        print('about the location, patients, and rules are revealed.')

    def finalize(self):
        self.pdf.close()
