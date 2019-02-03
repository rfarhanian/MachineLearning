import csv

import pandas


class ClaimParser():
    def __init__(self, location):
        self.location = location
        input = pandas.read_csv(location)
        with open(self.location) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                    line_count += 1
                else:
                    # print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
                    line_count += 1
            print(f'Processed {line_count} lines.')


parser = ClaimParser("./input/claim.sample.csv")
