import numpy as np

from sklearn.manifold import TSNE
# For the UCI ML handwritten digits dataset
from sklearn.datasets import load_digits

# Import matplotlib for plotting graphs ans seaborn for attractive graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns

def plot(x, colors):
    # Choosing color palette
    # https://seaborn.pydata.org/generated/seaborn.color_palette.html
    palette = np.array(sns.color_palette("pastel", 10))
    # pastel, husl, and so on

    # Create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int8)])
    # Add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([pe.Stroke(linewidth=5, foreground="w"), pe.Normal()])
        txts.append(txt)
    plt.savefig('./digits_tsne-pastel.png', dpi=120)
    return f, ax, txts


digits = load_digits()
print(digits.data.shape)
# There are 10 classes (0 to 9) with alomst 180 images in each class
# The images are 8x8 and hence 64 pixels(dimensions)

# Place the arrays of data of each digit on top of each other and store in X
X = np.vstack([digits.data[digits.target==i] for i in range(10)])
# Place the arrays of data of each target digit by the side of each other continuosly and store in Y
Y = np.hstack([digits.target[digits.target==i] for i in range(10)])

# Implementing the TSNE Function - ah Scikit learn makes it so easy!
digits_final = TSNE(perplexity=30).fit_transform(X)
# Play around with varying the parameters like perplexity, random_state to get different plots

plot(digits_final, Y)

