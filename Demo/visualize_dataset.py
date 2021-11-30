import vhh_rd.RD as RD
import vhh_rd.Helpers as Helpers
import os
import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

"""
Creates UMAP and t-SNE visualizations of the dataset
"""

config_path = "./config/config_rd.yaml"


def main():
    rd = RD.RD(config_path)
    similarities = []
    # Compute all similarities
    features = []
    print("Loading features")
    for ft_name in os.listdir(rd.features_path):
        path = os.path.join(rd.features_path, ft_name)
        feature = Helpers.do_unpickle(path)
        features.append(feature)
    
    print("Fitting UMAP")
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(features)

    print("Plotting UMAP")
    plt.scatter(embedding[:, 0], embedding[:, 1], s = 1)
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection ({0})'.format(rd.config["MODEL"]), fontsize=12)
    umap_path = os.path.join(rd.visualizations_path, "UMAP_{0}.png".format(rd.config["MODEL"]))
    plt.savefig(umap_path)
    plt.close()
    print("Stored plot of UMAP at {0}".format(umap_path))

    print("Fitting t-SNE")
    embedding = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(np.asarray(features, dtype=np.float32))
    print("Plotting t-SNE")
    plt.scatter(embedding[:, 0], embedding[:, 1], s = 1)
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('t-SNE projection ({0})'.format(rd.config["MODEL"]), fontsize=12)
    tSNE_path = os.path.join(rd.visualizations_path, "t-SNE_{0}.png".format(rd.config["MODEL"]))
    plt.savefig(tSNE_path)
    print("Stored plot of t-SNE at {0}".format(tSNE_path))

if __name__ == "__main__":
    main()