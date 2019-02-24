import numpy as np

from hw3.domain.People import People
from hw3.domain.Restaurant import Restaurant


class DataParser:

    def __init__(self):
        self.people = self.init_people_matrix()
        self.restaurant = self.init_restaurant_matrix()
        self.result = self.weave()

    def init_people_matrix(self):
        print(np.array([np.random.dirichlet(np.ones(4), size=1)]))
        people = {'Jane': {'willingness to travel': 0.1596993,
                           'desire for new experience': 0.67131344,
                           'cost': 0.15006726,
                           # 'indian food':1,
                           # 'Mexican food':1,
                           # 'hipster points':3,
                           'vegetarian': 0.01892,
                           },
                  'Bob': {'willingness to travel': 0.63124581,
                          'desire for new experience': 0.20269888,
                          'cost': 0.01354308,
                          # 'indian food':1,
                          # 'Mexican food':1,
                          # 'hipster points':3,
                          'vegetarian': 0.15251223,
                          },
                  'Mary': {'willingness to travel': 0.49337138,
                           'desire for new experience': 0.41879654,
                           'cost': 0.05525843,
                           # 'indian food':1,
                           # 'Mexican food':1,
                           # 'hipster points':3,
                           'vegetarian': 0.03257365,
                           },
                  'Mike': {'willingness to travel': 0.08936756,
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
                  'Skip': {'willingness to travel': 0.08534087,
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
                  'Moe': {'willingness to travel': 0.05101531,
                          'desire for new experience': 0.03976796,
                          'cost': 0.06372092,
                          # 'indian food':1,
                          # 'Mexican food':1,
                          # 'hipster points':3,
                          'vegetarian': 0.84549581,
                          },
                  'Sara': {'willingness to travel': 0.18780828,
                           'desire for new experience': 0.59094026,
                           'cost': 0.08490399,
                           # 'indian food':1,
                           # 'Mexican food':1,
                           # 'hipster points':3,
                           'vegetarian': 0.13634747,
                           },
                  'Tom': {'willingness to travel': 0.77606127,
                          'desire for new experience': 0.06586204,
                          'cost': 0.14484121,
                          # 'indian food':1,
                          # 'Mexican food':1,
                          # 'hipster points':3,
                          'vegetarian': 0.01323548,
                          }
                  }
        # Transform the user data into a matrix(M_people). Keep track of column and row ids.
        # convert each person's values to a list
        peopleKeys, peopleValues = [], []
        lastKey = 0
        for k1, v1 in people.items():
            row = []

            for k2, v2 in v1.items():
                peopleKeys.append(k1 + '_' + k2)
                if k1 == lastKey:
                    row.append(v2)
                    lastKey = k1

                else:
                    peopleValues.append(row)
                    row.append(v2)
                    lastKey = k1
        # here are some lists that show column keys and values
        print(peopleKeys)
        print(peopleValues)
        peopleMatrix = np.array(peopleValues)
        peopleMatrix.shape
        return People(peopleMatrix, peopleKeys, peopleValues)

    def init_restaurant_matrix(self):
        # Next you collected data from an internet website. You got the following information.

        # 1 is bad, 5 is great

        np.random.randint(5, size=4) + 1

        restaurants = {'flacos': {'distance': 2,
                                  'novelty': 3,
                                  'cost': 4,
                                  # 'average rating': 5,
                                  # 'cuisine': 5,
                                  'vegetarian': 5
                                  },
                       'Joes': {'distance': 5,
                                'novelty': 1,
                                'cost': 5,
                                # 'average rating': 5,
                                # 'cuisine': 5,
                                'vegetarian': 3
                                },
                       'Poke': {'distance': 4,
                                'novelty': 2,
                                'cost': 4,
                                # 'average rating': 5,
                                # 'cuisine': 5,
                                'vegetarian': 4
                                },
                       'Sush-shi': {'distance': 4,
                                    'novelty': 3,
                                    'cost': 4,
                                    # 'average rating': 5,
                                    # 'cuisine': 5,
                                    'vegetarian': 4
                                    },
                       'Chick Fillet': {'distance': 3,
                                        'novelty': 2,
                                        'cost': 5,
                                        # 'average rating': 5,
                                        # 'cuisine': 5,
                                        'vegetarian': 5
                                        },
                       'Mackie Des': {'distance': 2,
                                      'novelty': 3,
                                      'cost': 4,
                                      # 'average rating': 5,
                                      # 'cuisine': 5,
                                      'vegetarian': 3
                                      },
                       'Michaels': {'distance': 2,
                                    'novelty': 1,
                                    'cost': 1,
                                    # 'average rating': 5,
                                    # 'cuisine': 5,
                                    'vegetarian': 5
                                    },
                       'Amaze': {'distance': 3,
                                 'novelty': 5,
                                 'cost': 2,
                                 # 'average rating': 5,
                                 # 'cuisine': 5,
                                 'vegetarian': 4
                                 },
                       'Kappa': {'distance': 5,
                                 'novelty': 1,
                                 'cost': 2,
                                 # 'average rating': 5,
                                 # 'cuisine': 5,
                                 'vegetarian': 3
                                 },
                       'Mu': {'distance': 3,
                              'novelty': 1,
                              'cost': 5,
                              # 'average rating': 5,
                              # 'cuisine': 5,
                              'vegetarian': 3
                              }
                       }

        # Transform the restaurant data into a matrix(M_resturants) use the same column index.

        restaurantsKeys, restaurantsValues = [], []

        for k1, v1 in restaurants.items():
            for k2, v2 in v1.items():
                restaurantsKeys.append(k1 + '_' + k2)
                restaurantsValues.append(v2)

        # here are some lists that show column keys and values
        print(restaurantsKeys)
        print(restaurantsValues)

        len(restaurantsValues)
        # reshape to 2 rows and 4 columns

        # converting lists to np.arrays is easy
        restaurantsMatrix = np.reshape(restaurantsValues, (10, 4))

        restaurantsMatrix

        restaurantsMatrix.shape
        return Restaurant(restaurantsMatrix, restaurantsKeys, restaurantsValues)

    def weave(self):
        restaurantsMatrix = self.restaurant.get_matrix()
        peopleMatrix = self.people.get_matrix()
        # Matrix multiplication
        # Dot products are the matrix multiplication of a row vectors and column vectors (n,p) * (p,n)
        #  shape example: ( 2 X 4 ) * (4 X 2) = 2 * 2
        a = np.array([[1, 0], [0, 1]])
        b = np.array([[1], [2]])

        a.shape, b.shape

        # when 2D arrays are involved, np.dot give the matrix product.
        np.dot(a, b)

        # documentation: https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html
        # intuition: https://www.mathsisfun.com/algebra/matrix-multiplying.html
        c = np.array([[1, 2, 3], [4, 5, 6]])
        d = np.array([[7, 8], [9, 10], [11, 12]])

        c.shape, d.shape

        np.dot(c, d)
        # What is a matrix product?
        # https://en.wikipedia.org/wiki/Matrix_multiplication
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.matmul.html#numpy.matmul
        # matmul give the matrix product, too.
        np.matmul(a, b)

        restaurantsMatrix.shape, peopleMatrix.shape
        # However, this won't work...
        np.matmul(restaurantsMatrix, peopleMatrix)

        # The most imporant idea in this project is the idea of a linear combination.

        # Informally describe what a linear combination is and how it will relate to our resturant matrix.

        # This is for you to answer! However....https://en.wikipedia.org/wiki/Linear_combination
        # essentially you are multiplying each term by a constant and summing the results.

        # Choose a person and compute(using a linear combination) the top restaurant for them.
        # What does each entry in the resulting vector represent?

        # print(peopleKeys)
        # print(peopleValues)

        # print(restaurantsKeys)
        # print(restaurantsValues)

        restaurantsMatrix.shape, peopleMatrix.shape

        # We need to swap axis on peopleMatrix
        # newPeopleMatrix = np.swapaxes(peopleMatrix, 1, 0)

        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.swapaxes.html
        newPeopleMatrix = np.swapaxes(peopleMatrix, 0, 1)

        newPeopleMatrix.shape, restaurantsMatrix.shape
        restaurantsMatrix.shape, newPeopleMatrix.shape

        # Next compute a new matrix (M_usr_x_rest  i.e. an user by restaurant) from all people.  What does the a_ij matrix represent?
        # Let's check our answers

        results = np.matmul(restaurantsMatrix, newPeopleMatrix)

        results

        # Sum all columns in M_usr_x_rest to get optimal restaurant for all users.  What do the entry’s represent?
        # I believe that this is what John and  is asking for, sum by columns
        np.sum(results, axis=1)

        # restaurantsKeys

        # Now convert each row in the M_usr_x_rest into a ranking for each user and call it M_usr_x_rest_rank.
        # Do the same as above to generate the optimal resturant choice.
        results

        # Say that rank 1 is best

        # reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
        # Argsort returns the indices that would sort an array - https://stackoverflow.com/questions/17901218/numpy-argsort-what-is-it-doing
        # By default, argsort is in ascending order, but below, we make it in descending order and then add 1 since ranks start at 1
        sortedResults = results.argsort()[::-1] + 1
        sortedResults

        # What is the problem here?

        results.shape
        return results

    def get_result(self):
        return self.result

    def get_people(self):
        return self.people

    def get_restaurant(self):
        return self.restaurant
