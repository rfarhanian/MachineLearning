import numpy as np

from hw3.domain.People import People
from hw3.domain.Restaurant import Restaurant


class DataParser:

    def __init__(self):
        test_result = self.test()
        print(test_result)
        self.people = self.init_people_matrix()
        self.restaurant = self.init_restaurant_matrix()

    def test(self):
        dataDict = {'device1': (1, 1, 0, 1), 'device2': (0, 1, 0, 1), 'device3': (1, 0, 0, 1)}
        orderedNames = ['device1', 'device2', 'device3']

        dataMatrix = np.array([dataDict[i] for i in orderedNames])

        return dataMatrix

    def init_people_matrix(self):
        print(np.array([np.random.dirichlet(np.ones(4), size=1)]))
        people = {
            'Jane': {'willingness to travel': 0.3796993,
                     'desire for new experience': 0.67131344,
                     'cost': 0.34006726,
                     # 'indian food':1,
                     # 'Mexican food':1,
                     # 'hipster points':3,
                     'vegetarian': 0.31892,
                     },
            'Bob': {'willingness to travel': 0.0731,
                    'desire for new experience': 0.302,
                    'cost': 0.2,
                    # 'indian food':1,
                    # 'Mexican food':1,
                    # 'hipster points':3,
                    'vegetarian': 0.45251223,
                    },
            'Mary': {'willingness to travel': 0.38938,
                     'desire for new experience': 0.027,
                     'cost': 0.05525843,
                     # 'indian food':1,
                     # 'Mexican food':1,
                     # 'hipster points':3,
                     'vegetarian': 0.03257365,
                     },
            'Mike': {'willingness to travel': 0.2936756,
                     'desire for new experience': 0.14813813,
                     'cost': 0.43602425,
                     # 'indian food':1,
                     # 'Mexican food':1,
                     # 'hipster points':3,
                     'vegetarian': 0.32647006,
                     },
            'Alice': {'willingness to travel': 0.05846052,
                      'desire for new experience': 0.6550466,
                      'cost': 0.1020457,
                      # 'indian food':1,
                      # 'Mexican food':1,
                      # 'hipster points':3,
                      'vegetarian': 0.18444717,
                      },
            'John': {'willingness to travel': 0.08534087,
                     'desire for new experience': 0.20286902,
                     'cost': 0.49978215,
                     # 'indian food':1,
                     # 'Mexican food':1,
                     # 'hipster points':3,
                     'vegetarian': 0.21200796,
                     },
            'Kira': {'willingness to travel': 0.14621567,
                     'desire for new experience': 0.08325185,
                     'cost': 0.59864525,
                     # 'indian food':1,
                     # 'Mexican food':1,
                     # 'hipster points':3,
                     'vegetarian': 0.17188723,
                     },
            'Moe': {'willingness to travel': 0.07,
                    'desire for new experience': 0.09,
                    'cost': 0.06372092,
                    # 'indian food':1,
                    # 'Mexican food':1,
                    # 'hipster points':3,
                    'vegetarian': 0.84549581,
                    },
            'Sara': {'willingness to travel': 0.38780828,
                     'desire for new experience': 0.59094026,
                     'cost': 0.08490399,
                     # 'indian food':1,
                     # 'Mexican food':1,
                     # 'hipster points':3,
                     'vegetarian': 0.13634747,
                     },
            'Tom': {'willingness to travel': 0.08,
                    'desire for new experience': 0.06586204,
                    'cost': 0.84484121,
                    # 'indian food':1,
                    # 'Mexican food':1,
                    # 'hipster points':3,
                    'vegetarian': 0.01323548,
                    }
        }
        # Transform the user data into a matrix(M_people). Keep track of column and row ids.
        # convert each person's values to a list
        # peopleKeys = list(people.keys())
        # peopleValues = list(people.values())
        peopleKeys, peopleValues = [], []

        lastKey = 'Tom'
        for name, interests in people.items():
            row = []

            for subject, value in interests.items():
                peopleKeys.append(name + '_' + subject)
                if name == lastKey:
                    row.append(value)
                    lastKey = name

                else:
                    peopleValues.append(row)
                    row.append(value)
                    lastKey = name
        # here are some lists that show column keys and values
        print(peopleKeys)
        print(peopleValues)
        peopleMatrix = np.array(peopleValues)
        peopleMatrix.shape
        return People(peopleMatrix, peopleKeys, peopleValues, list(people.keys()))

    def init_restaurant_matrix(self):
        # Next you collected data from an internet website. You got the following information.

        # 1 is bad, 5 is great

        np.random.randint(5, size=4) + 1

        restaurants = {
            'flacos':
                {
                    'distance': 2,
                    'novelty': 3,
                    'cost': 4,
                    'vegetarian': 4
                },
            'Joes':
                {
                    'distance': 5,
                    'novelty': 1,
                    'cost': 5,
                    'vegetarian': 5
                },
            'Poke':
                {
                    'distance': 4,
                    'novelty': 2,
                    'cost': 2,
                    'vegetarian': 4
                },
            'Sush-shi':
                {
                    'distance': 4,
                    'novelty': 3,
                    'cost': 4,
                    'vegetarian': 4
                },
            'Chick Fillet':
                {
                    'distance': 2,
                    'novelty': 2,
                    'cost': 5,
                    'vegetarian': 5
                },
            'Mackie Des':
                {
                    'distance': 2,
                    'novelty': 3,
                    'cost': 4,
                    'vegetarian': 1
                },
            'Michaels':
                {
                    'distance': 3,
                    'novelty': 1,
                    'cost': 3,
                    'vegetarian': 5
                },
            'Amaze':
                {
                    'distance': 3,
                    'novelty': 4,
                    'cost': 2,
                    'vegetarian': 1
                },
            'Kappa':
                {
                    'distance': 5,
                    'novelty': 2,
                    'cost': 2,
                    'vegetarian': 3
                },
            'Mu':
                {
                    'distance': 3,
                    'novelty': 1,
                    'cost': 4,
                    'vegetarian': 3
                }
        }

        # Transform the restaurant data into a matrix(M_resturants) use the same column index.

        restaurantsKeys, restaurantsValues = [], []

        for name, restaurant_attributes in restaurants.items():
            for subject, value in restaurant_attributes.items():
                restaurantsKeys.append(name + '_' + subject)
                restaurantsValues.append(value)

        # here are some lists that show column keys and values
        print(restaurantsKeys)
        print(restaurantsValues)

        len(restaurantsValues)
        # reshape to 2 rows and 4 columns

        # converting lists to np.arrays is easy
        restaurantsMatrix = np.reshape(restaurantsValues, (10, 4))

        restaurantsMatrix

        restaurantsMatrix.shape
        return Restaurant(restaurantsMatrix, restaurantsKeys, restaurantsValues, list(restaurants.keys()))

    def get_people(self):
        return self.people

    def get_restaurant(self):
        return self.restaurant
