# Decision making with Matrices

# This is a pretty simple assignment.  You will do something you do everyday, but today it will be with matrix manipulations.
# The problem is: you and your work friends are trying to decide where to go for lunch. You have to pick a restaurant 
# that's best for everyone.  Then you should decided if you should split into two groups so everyone is happier.
# Dispite the simplicity of the process, you will need to make decisions regarding how to process the data.
# This process was thoroughly investigated in the operation research community. 
# This approach can prove helpful on any number of decision making problems that are currently not leveraging machine learning.

# -------------------------- Assignment --------------------------

# You asked your 10 work friends to answer a survey. They gave you back the following dictionary object.
people = {'Jane': {'willingness to travel': 0.1596993,
                   'desire for new experience': 0.67131344,
                   'cost': 0.15006726,
                    'indian food':1,
                    'Mexican food':1,
                    'hipster points':3,
                   'vegetarian': 0.01892          }         }

# Transform the user data into a matrix(M_people). Keep track of column and row ids.
# Added: normalize the points for each user -- make their preferences add to 1 in the actual weights matrix you use for analysis.

# Next you collected data from an internet website. You got the following information.
# Added: make these scores /10, on a scale of 0-10, where 10 is good. So, 10/10 for distance means very close. 
resturants  = {'flacos':{'distance': 2,
                          'novelty': 3,
                          'cost': 4,
                          'vegetarian': 5,
                          'average rating': 7
                        }    }

-------- Data processing ends --------
-------- Start with 2 numpy matrices if you're not excited to do data processing atm ------ 

# Transform the restaurant data into a matrix(M_resturants) use the same column index.

# The most important idea in this project is the idea of a linear combination.
# Informally describe what a linear combination is and how it will relate to our restaurant matrix.

# Choose a person and compute(using a linear combination) the top restaurant for them.  What does each entry in the resulting vector represent.

# Next compute a new matrix (M_usr_x_rest  i.e. an user by restaurant) from all people.  What does the a_ij matrix represent?

# Sum all columns in M_usr_x_rest to get optimal restaurant for all users.  What do the entry’s represent?

------------- CLASS ENDS -----------

-------- Discuss with class mates ---------

# Now convert each row in the M_usr_x_rest into a ranking for each user and call it M_usr_x_rest_rank.   Do the same as above to generate the optimal resturant choice.
Jack scores for 
                    scores        ranking 
Tacos             74            1    
tapas              50             3   
bar                  70             2

# Why is there a difference between the two?  What problem arrives?  What does represent in the real world?

# How should you preprocess your data to remove this problem.

------------ Clustering stuff ------------

# Find  user profiles that are problematic, explain why?

# Think of two metrics to compute the disatistifaction with the group.

# Should you split in two groups today?

---- Did you understand what's going on? ---------

# Ok. Now you just found out the boss is paying for the meal. How should you adjust. Now what is best restaurant?

# Tommorow you visit another team. You have the same restaurants and they told you their optimal ordering for restaurants.  Can you find their weight matrix?