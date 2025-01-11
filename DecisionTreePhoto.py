from sklearn.tree import plot_tree
from matplotlib import pyplot as plt

class DecisionTreePhoto:
    count = 0
    def __init__(self):
        DecisionTreePhoto.count += 1

    def photo(self,dtc,x_t_columns):
        # Plot del Decision Tree
        plt.figure(figsize=(20, 15))
        plot_tree(dtc, filled=True, feature_names=x_t_columns, rounded=True)
        # Salva l'immagine ad alta risoluzione
        plt.savefig(f'decision_tree_high_res{DecisionTreePhoto.count}.png', dpi=300, bbox_inches='tight')
        # Mostra l'immagine
        plt.show()
        DecisionTreePhoto.count += 1