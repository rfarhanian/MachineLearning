import matplotlib.pyplot as plt
import seaborn as sns

from hw3.domain.People import People
from hw3.domain.Restaurant import Restaurant


class DataVisualizer:
    def visualize(self, results, people: People, restaurants: Restaurant):
        # first plot heatmap
        # https://seaborn.pydata.org/generated/seaborn.heatmap.html
        plot_dims = (12, 10)
        fig, ax = plt.subplots(figsize=plot_dims)
        sns.heatmap(ax=ax, data=results, annot=True)
        plt.show()

        # remember a_ij is the score for a restaurant for a person
        # x is the person, y is the restaurant

        print(people.get_keys())
        # x=0 (Jane), x=1 (Bob), x=2 (Mary), x=3 (Mike), x=4 (Alice),
        # x=5 (Skip), x=6 (Kira), x=7 (Moe), x=8 (Sara), x=9 (Tom)

        print(restaurants.get_keys())
        # y=0 (flacos), y=1 (Joes), y=2 (Poke), y=3 (Sush-shi), y=4 (Chick Fillet),
        # y=5 (Mackie Des), y=6 (Michaels), y=7 (Amaze), y=8 (Kappa), y=9 (Mu)

        # What is the problem if we want to do clustering with this matrix?

        results.shape

        # from sklearn.preprocessing import StandardScaler
        # from sklearn.decomposition import PCA

        people.get_matrix().shape
