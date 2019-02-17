from typing import List

import pandas


class ClaimParser():
    columns: List[str]

    def __init__(self, location):
        self.location = location
        self.__parse__()

    def __parse__(self):
        # line_count: int = 0
        # self.my_data = pandas.read_csv(self.location, header=0)
        # data_types = [
        #     ('V1', 'int16'),
        #     ('Claim.Number', 'float64'),
        #     ('Claim.Line.Number', 'int64'),
        #     ('Member.ID', 'int64'),
        #     ('Provider.ID', '<u10'),
        #     ('Line.Of.Business.ID', 'object'),
        #     ('Revenue.Code', 'object'),
        #     ('Service.Code', 'object'),
        #     ('Place.Of.Service.Code', 'object'),
        #     ('Procedure.Code', 'object'),
        #     ('Diagnosis.Code', 'object'),
        #     ('Claim.Charge.Amount', 'float64'),
        #     ('Denial.Reason.Code', 'object'),
        #     ('Price.Index', 'object'),
        #     ('In.Out.Of.Network', 'object'),
        #     ('Reference.Index', 'object'),
        #     ('Pricing.Index', 'object'),
        #     ('Capitation.Index', 'object'),
        #     ('Subscriber.Payment.Amount', 'float64'),
        #     ('Provider.Payment.Amount', 'float64'),
        #     ('Group.Index', 'int64'),
        #     ('Subscriber.Index', 'int64'),
        #     ('Subgroup.Index', 'int64'),
        #     ('Claim.Type', 'object'),
        #     ('Claim.Subscriber.Type', 'object'),
        #     ('Claim.Pre.Prince.Index', 'object'),
        #     ('Claim.Current.Status', 'int64'),
        #     ('Network.ID', 'object'),
        #     ('Agreement.ID', 'object')
        # ]
        data_types = [
            ('V1', 'int16'),
            ('Claim.Number', 'float64'),
            ('Claim.Line.Number', 'int64'),
            ('Member.ID', 'int64'),
            ('Provider.ID', 'str'),
            ('Line.Of.Business.ID', 'str'),
            ('Revenue.Code', 'object'),
            ('Service.Code', 'object'),
            ('Place.Of.Service.Code', 'object'),
            ('Procedure.Code', 'str'),
            ('Diagnosis.Code', 'object'),
            ('Claim.Charge.Amount', 'float64'),
            ('Denial.Reason.Code', 'object'),
            ('Price.Index', 'object'),
            ('In.Out.Of.Network', 'object'),
            ('Reference.Index', 'object'),
            ('Pricing.Index', 'object'),
            ('Capitation.Index', 'object'),
            ('Subscriber.Payment.Amount', 'float64'),
            ('Provider.Payment.Amount', 'float64'),
            ('Group.Index', 'int64'),
            ('Subscriber.Index', 'int64'),
            ('Subgroup.Index', 'int64'),
            ('Claim.Type', 'object'),
            ('Claim.Subscriber.Type', 'object'),
            ('Claim.Pre.Prince.Index', 'object'),
            ('Claim.Current.Status', 'int64'),
            ('Network.ID', 'object'),
            ('Agreement.ID', 'object')
        ]
        # print(pandas.read_csv(self.location, header=0).dtypes)
        self.rows = pandas.np.genfromtxt(self.location, delimiter=',', skip_header=1, dtype=data_types)
        self.columns2 = dict()
        self.columns2 = []
        self.line_count = len(self.rows)

        # with open(self.location) as csv_file:
        #     self.csv_reader = csv.reader(csv_file, delimiter=',')
        #     for row in self.csv_reader:
        #         if line_count == 0:
        #             count=0
        #             for item in row:
        #                 self.columns2[count]=item
        #                 count=count+1
        #         else:
        #             self.rows.append(row)
        #         line_count += 1
        # self.line_count = line_count

    def get_rows(self):
        return self.rows

    def get_columns(self):
        return self.columns2

    def get_row_line_count(self):
        return self.line_count

    def __str__(self) -> str:
        return 'line count:', self.get_row_line_count(), ', columns:', self.get_columns()
