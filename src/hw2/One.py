from typing import List


def startWith(rowItem: List, prefix='j'):
    attributeValue = rowItem[10]
    print("my", attributeValue)
    return attributeValue.startswith(prefix.encode())


class One:
    @classmethod
    def get_procedure_code(cls, data, column_name='Procedure.Code', prefix='j'.encode()):
        # my_list = [x for x in data if x.attribute == value]
        # my_list = list(filter(startWith, data))
        # np.where(a < 5, a, 10 * a)
        # my_list =np.where(data)
        count = 0
        total = 0
        unfiltered_result = data.tolist()[:][9]
        for x in unfiltered_result:
            total += 1
            if type(x) == bytes:
                x = x.decode()
                if x.startswith('J'):
                    count += 1
                else:
                    print("bytes", x)
            if type(x) == str:
                if x.startswith('J'):
                    count += 1
                else:
                    print("str", x)
            else:
                print(type(x))
                print(x)

            # str_value = str(x).decode('utf-8')
            # if str_value.startswith('j'.encode()):
            #     count += 1
            # print(str_value)
        print(total)
        return count
        # return data.loc[:, column_name].str.startswith(prefix)
