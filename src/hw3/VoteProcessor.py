import numpy as np

from hw3.domain.People import People
from hw3.domain.Restaurant import Restaurant


class VoteProcessor:

    @classmethod
    def process(cls, restaurant: Restaurant, people: People):
        restaurants_matrix = restaurant.get_matrix()
        people_matrix = people.get_matrix()

        # What is a matrix product?
        # https://en.wikipedia.org/wiki/Matrix_multiplication
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.matmul.html#numpy.matmul
        # matmul give the matrix product, too.

        restaurants_matrix.shape, people_matrix.shape

        transposed_people_matrix = people_matrix.T
        M_usr_x_rest = np.matmul(restaurants_matrix, transposed_people_matrix)

        # The most important idea in this project is the idea of a linear combination.

        # Informally describe what a linear combination is and how it will relate to our restaurant matrix.
        print(
            'In mathematics, a linear combination is an expression constructed from a set of terms by multiplying '
            'each term by a constant and adding the results (e.g. a linear combination of x and y would be any '
            'expression of the form ax + by, where a and b are constants).')
        print(
            'The most important formula is to find out what multiplication of restaurant to person yields a number '
            'that represents that persons vote for the restaurant.')
        print('Novelty and desire for experience, cost and cost, distance and willingness to travel have a direct '
              'relationship with each other, so if we multiply these items per person, we can calculate their vote for each restaurant')
        print(
            'restaurant matrix is (10*4) and people matrix is also (10*4). We transposed people matrix so that it becomes 4*10 and '
            'every column in the transposed people matrix represent one person (e.g. Tom is the last column). Now the matrices are ready to be multiplied.')
        print(
            'The multiplication will yield a 10*10 matrix. Every column in the result matrix(M_usr_x_rest) represent the votes of one user.')
        print(
            'Every row in the result matrix(M_usr_x_rest) represent the votes of all users to one specific restaurant.')
        print('(e.g Jane votes to all restaurants are in the first column)')

        # print('Toms Mu Vote is : (0.77606127 * 3) + (0.6586204 * 1) + (0.14484121 * 5) +(0.1323548 * 3): 4.10807466')
        # print('Toms Flacos Vote is : (0.77606127 * 2) + (0.6586204 * 3) + (0.14484121 * 4) +(0.1323548 * 5): 4.76912258')
        # print('Jane Flacos Vote is : (0.1596993 * 2) + (0.067131344 * 3) + (0.15006726 * 4) +(0.01892 * 5): 1.215661672')
        # This is for you to answer! However....https://en.wikipedia.org/wiki/Linear_combination
        # essentially you are multiplying each term by a constant and summing the results.

        # What does each entry in the resulting vector represent?
        print('Each entry represents an individuals vote for a specific restaurant. Each column contains a')
        print(' specific individuals vote of all restaurants.')

        print('If we transpose the result matrix, every row will represent individuals votes to all restaurants')
        print('I call it M_usr_x_rest, every row will represent individuals votes to all restaurants')
        tom_votes = M_usr_x_rest.T[9]
        # Choose a person and compute(using a linear combination) the top restaurant for them.
        print('For example, Tom\'s votes for all restaurants are: ' + str(tom_votes))
        toms_highest_vote = tom_votes.argmax()

        print('His highest vote was for ' + str(restaurant.get_names()[toms_highest_vote]) + '\'s restaurant ')

        print(people.get_keys())
        print(people.get_values())

        print(restaurant.get_keys())
        print(restaurant.get_values())

        restaurants_matrix.shape, people_matrix.shape

        # We need to swap axis on people_matrix
        # newPeopleMatrix = np.swapaxes(people_matrix, 1, 0)

        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.swapaxes.html
        newPeopleMatrix = np.swapaxes(people_matrix, 0, 1)

        newPeopleMatrix.shape, restaurants_matrix.shape
        restaurants_matrix.shape, newPeopleMatrix.shape

        # Next compute a new matrix (M_usr_x_rest  i.e. a user by restaurant) from all people.  What does the a_ij matrix represent? answered above
        # Let's check our answers

        results = np.matmul(restaurants_matrix, newPeopleMatrix)

        results

        # Sum all columns in M_usr_x_rest to get optimal restaurant for all users.  What do the entry’s represent?
        # I believe that this is what John and  is asking for, sum by columns
        np.sum(results, axis=1)

        sum = np.sum(M_usr_x_rest, axis=1)
        print(
            'sum represents the sum of all the votes of all individuals to all restaurants. This array can be used to rank the restaurants based on popularity')
        # @todo: delete row: 3.0280796+ 2.27690624+2.65737312+3.32868665+2.66666108+2.99036796+1.2353793+ 4.21147962+4.21147962+1.82670446+1.95750764 = 26.17992734
        # @todo: delete col: 3.0280796 + 2.68732173+2.6270343499999997+ 3.9995968100000003+ 3.41247949 + 3.8384571999999997+ 3.79620404+ 4.70369722999999995+ 3.16979065+ 2.39525089=33.65804

        # Now convert each row in the M_usr_x_rest into a ranking for each user and call it M_usr_x_rest_rank.
        # Do the same as above to generate the optimal restaurant choice.


        # Say that rank 1 is best

        # reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
        # Argsort returns the indices that would sort an array - https://stackoverflow.com/questions/17901218/numpy-argsort-what-is-it-doing
        # By default, argsort is in ascending order, but below, we make it in descending order and then add 1 since ranks start at 1
        sortedM_usr_x_rest = M_usr_x_rest.argsort()[::-1] + 1
        sortedM_usr_x_rest
        sortedResults = results.argsort()[::-1] + 1
        sortedResults

        # What is the problem here?
        print(
            'The problem of using argsort() with numpy.ndarray is that the flattened data. What we need is to sort the restaurants per user vote.')
        print('I observed strange behavior in argsort functionality that can be addressed by calling argsort twice.')
        print('The axis 0 should be considered so that the votes of people are ranked per restaurant.')

        # Now convert each row in the M_usr_x_rest into a ranking for each user and call it M_usr_x_rest_rank.   Do the same as above to generate the optimal restaurant choice.
        M_usr_x_rest_rank = np.argsort(np.argsort(M_usr_x_rest, axis=0), axis=0)
        M_usr_x_rest_rank

        print('------------------------People with restaurant rank-------------------------')
        names = people.get_names()
        offset = '\t\t\t'
        print(offset, 'name', offset, 'score', '\t\t', 'rank')

        for i in range(0, 10):
            print(names[i])
            ith_rest_rank = M_usr_x_rest_rank.T[0:10][i]
            ith_rest_score = M_usr_x_rest.T[0:10][i]
            r = restaurant.get_names()
            for j in range(0, 10):
                print(offset, r[j], '\t\t\t', ith_rest_score[j], '\t\t', ith_rest_rank[j])

        print('----------------------------------------------------------------------')

        M_rest_x_usr_rank = np.argsort(np.argsort(sum, axis=0), axis=0)[::-1] + 1

        results.shape
        print('The computed sum can help us display the best restaurants in order based on people\'s vote')
        print('M_rest_x_usr_rank ranks restaurants according to popularity. The best restaurant is ranked as 1')
        print('------------------------Restaurants with rank-------------------------')
        names = restaurant.get_names()
        print('name', '\t\t\t', 'score', '\t\t', 'rank')
        for i in range(0, 10):
            ith_rest_rank = M_rest_x_usr_rank[i]
            print(names[i], '\t\t\t', sum[i], '\t\t', ith_rest_rank)
        print('----------------------------------------------------------------------')
        return M_usr_x_rest_rank
