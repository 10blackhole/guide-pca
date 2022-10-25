import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from matplotlib.image import imread
import matplotlib.pyplot as plt

# Display the image
def display_image(my_image):
    plt.figure(figsize=[12,8])
    plt.imshow(my_image)
    plt.show()

def new_image(my_image):
    image_sum = my_image.sum(axis=2)
    return image_sum/image_sum.max()

def grayscaling_image(new_image):
    plt.figure(figsize=[12,8])
    plt.imshow(new_image, cmap=plt.cm.gray)
    plt.show()

def pca(new_image):
    pca = PCA()
    return pca.fit(new_image)
    # var_cumu = np.cumsum(pca.explained_variance_ratio_)*100
    # k = np.argmax(var_cumu > 95)

    # plt.figure(figsize=[10,5])
    # plt.title('Cumulative Explained Variance explained by the components')
    # plt.xlabel('Principal components')
    # plt.ylabel('Cumulative Explained variance')
    # plt.axvline(x=k, color="k", linestyle="--")
    # plt.axhline(y=95, color="r", linestyle="--")
    # plt.plot(var_cumu)
    # plt.grid(True)
    # plt.show()

def scree_plot(new_image):
    pca = PCA()
    pca.fit(new_image)
    var_cumu = np.cumsum(pca.explained_variance_ratio_)*100
    k = np.argmax(var_cumu > 95)

    plt.figure(figsize=[10,5])
    plt.title('Cumulative Explained Variance explained by the components')
    plt.xlabel('Principal components')
    plt.ylabel('Cumulative Explained variance')
    plt.axvline(x=k, color="k", linestyle="--")
    plt.axhline(y=95, color="r", linestyle="--")
    plt.plot(var_cumu)
    plt.grid(True)
    plt.show()
