# Pricipal Component Analysis (PCA)
<!--# Links:-->
#  [Guide To Image Reconstruction Using Principal Component Analysis](https://analyticsindiamag.com/guide-to-image-reconstruction-using-principal-component-analysis/)

Principal Component Analysis belongs to a class of linear transforms bases on statistical techniques. This method provides a powerful toll for data analysis and pattern recognition, which is often preferred in signal an imagage processing as a technique of data compression, data dimension reduction, or decorrelation. PCA is an unsipervides learning method similar to clustering. It finds patterns without prior knowledge about whether the samples come from different treatment groups or essential differences. The objective is pursued by analysing principal components where we can perceive relationships that would otherwise remain hidden in higher dimensions.
The representation processed must be such that the loss of information must be minimal after discarding the higher dimensions.
The goal of the methods is to reorient the data so that a multitude of original variables can be summarized with relatively few "factors" or "components" that capture the maximum possible information from the original variables.


The output of PCA is principal components, which are less than or equal to the number of original variables. Less, in a case when we wish to discard or reduce the dimensions in our dataset. 

The image transformation technique from colour to the gray level, i.e. the intensity of the image, can be done using most of the common algorithms. According to relation, the implementations is usually based on the weighted sum of three core colour components Red, Green, and Blue. The R, G and B matrices contain image colour components, and the ewights are determined regarding the posibilities of human preception.
The PCA method provides an alternative way to this method, where the matrix A is replaced by matrix Al where only l largest (instead of n) eigenvalues are used for its formation. A vector of reconstructed variables is then given by relation. A selected real picture P and Its three reconstructed components are obtained accordingly for each eigenvalue and presented. The comparison of the intensity of images obtained from the original image as the weighted colour sum is evaluated as the first principal component. The variance figures for each principal component are present in the eigenvalue list. These indicate the amount of variation accounted for by each component within the feature space.

### Getting Started with the Code
**Importing labraries**

```python
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
```

Here we sill using imread from matplotlib to import the image as a matrix

**Setting image path**

```python
my_image=imread("/path/.png")
print(my_image.shape)
``````

**Displaying the image**
```python
plt.figure(figsize=[12,8])
plt.imshow(my_image)
``````

The image being processed is a coloured image and hance has data in 3 channels-Red, Green, Blue. Therefore the shape of the data (shape of data)

**Processing the image**

Let us now start with our image processing. Here first, we will be grayscaling our image, and then we'll perform PCA on the matrix with all the components. We will also create and look at the scree plot (In multivariate statistics, a scree plot is a line plot of the eigenvalues of factors or principal components in an analysis) to assess how many components we could retain and how much cumulative variance they capture.

**Greyscaling the image**
```python
image_sum = my_image.sum(axis=2)
print(image_sum.shape)

new_image = image_sum/image_sum.max()
print(new_image.max())

plt.figure(figsize=[12,8])
plt.imshow(new_image, cmap=plt.cm.gray)
``````
**Creating scree plot**
```python
from sklearn.decomposition import PCA, IncrementalPCA
pca = PCA()
pca.fit(new_image)
```

**Getting the cumulative variance**
```python
var_cumu = np.cumsum(pca.explained_variance_ratio_)*100
```
**How many PCs explain 95% of the variance?**
```python
k = np.argmax(var_cumu>95)
print("Number of components explaining 95% variance: "+str(k))
print("\n")

plt.figure(figsize=[10,5])
plt.title('Cumulative Explained Variance explained by the components')
plt.xlabel('Principal components')
plt.ylabel('Cumulative Explained variance')
plt.axvline(x=k, color="k", linestyle="--")
plt.axhline(y=95, color="r", linestyle="--")
ax = plt.plot(var_cumu)
```
Now let's reconstruct the image using only (#) components and see if our reconstructed image comes out to be visually different from the original image

**Reconstructing using Inverse Transform**
```python
ipca = IncrementalPCA(n_components=k)
image_recon = ipca.inverse_transform(ipca.fit_transform(new_image))
```
**Plotting the reconstructed image**
```python
plt.figure(figsize=[12,8])
plt.imshow(image_recon, cmap=plt.cm.gray)
```
As we can observe, there is a relative difference now. We shall try with a different value of components to check if that maked a difference in the missin clariry and help capture finer details in the visuals.

**Function to reconstruct and plot image for a given number of components**
```python
def plot_at_k(k):
	ipca = IncrementalPCA(n_components=k)
	image_recon = ipca.inverse_transform(ipca.fit_transform(new_image))
	plt.imshow(image_recon, cmap=plt.cm.gray)

k = 150
plt.figure(figsize=[12,8])
plot_at_k(100)
```

We can observe that, yes, the number of principal components do make a difference!

Plotting the same for different numbers of components to compare the exact relative difference,

**Setting different amount of K**
```python
ks = [10, 25, 50, 100, 150, 250]

plt.figure(figsize=[15,9])

for i in range(6):
	plt.subplot(2,3,i+1)
	plot_at_k(ks[i])
	plt.title("Components: "+str(ks[i]))

plt.subplots_adjust(wspace=0.2, hspace=0.0)
plt.show()
``````
Using PCA for Image Reconstruction, we can also segregate between the amounts of RGB present in an image,
```python
import cv2
img = cv2.cvtColor(cv2.imread('/path/.image'), cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
``````
**Splittin into channels**
```python
blue, green, red = cv2.split(img)
``````
**Plotting the images**
```python
fig = plt.figure(figsize=(15,7.2))
fig.add_subplot(131)
plt.title("Blue Presence")
plt.imshow(blue)
fig.add_subplot(132)
plt.title("Green Presence")
plt.imshow(green)
fig.add_subplot(133)
plt.title("Red Presence")
plt.imshow(red)
plt.show()
``````
A particular image channel can also be converted into a data frame for further processing,
```python
import numpy as np
import pandas as pd
``````
**Creating dataframe from blue presence in image**
```python
blue_chnl_df = pd.DataFrame(data=blue)
blue_chnl_df
```
The data fot each color presence can also be fit and transformed to a particular number of components for checking the variance of each color presence,

**Scaling data between 0 to 1**
```python
df_blue = blue/255
df_green = green/255
df_red = red/255
```
**Setting a reduced number of components**
```python
pca_b = PCA(n_components=50)
pca_b.fit(df_blue)
trans_pca_b = pca_b.transform(df_blue)
pca_g = PCA(n_components=50)
pca_g.fit(df_green)
trans_pca_g = pca_g.transform(df_green)
pca_r = PCA(n_components=50)
pca_r.fit(df_red)
trans_pca_r = pca_r.transform(df_red)
```
**Transforming shape**
```python
print(trans_pca_b.shape)
print(trans_pca_r.shape)
print(trans_pca_g.shape)
```
**Checking variance after reduced components**
```python
print(f"Blue Channel : {sum(pca_b.explained_variance_ratio_)}")
print(f"Green Channel: {sum(pca_g.explained_variance_ratio_)}")
print(f"Red Channel  : {sum(pca_r.explained_variance_ratio_)}")
```
Output(example):
```python
Blue Channel : 0.9835704508744926
Green Channel: 0.9794100254497594
Red Channel  : 0.9763416610407115
```
We can observe that by using 50 components we can keep around 98% of the variance in the data!

**************

# [Principal Component Analysis (PCA) applied to images (pdf)](http://people.ciirc.cvut.cz/~hlavac/TeachPresEn/11ImageProc/15PCA.pdf)


**PCA, the instance of the eigen-analysis**

PCA is mathematically defined as an orthogonal linear transformation that transforms the data to a new coordinate system such that the greatest variance by some projection of the data comes to lie on the first coordinate (called the first principal component), the second greatest variance on the second coordinate, and so on.

PCA objetive is to rotate rigidly the coordinate axes of the $p$-dimensional linear space to new "natural" positions (pricipal axes) such that:
+ Coordinate axes are ordered sush that principal axis 1 corresponds to the highest variance in data, axis 2 has the next highest variance,..., and axis $p$ has the lowest variance.
+ The covariance among each pair of principal exes is zero, i.e, they are uncorralated.

## Geometric motivation, principal components
- Two- dimensional vector space of observations, $(x_1, x_2)$
- Each observation corresponds to a single point in the vector space.
- *The goal*: Find another basis of the vector space, which treats variations of fata better.
- *We sill see later*: Data points (observations) are represented in a rotated orthgonal cooridante system. The origin is the mean of the data points and the axes are provied by the eigenvectors.
- Assume a single straight line approximating best the observation in the (total) least-square sense, i.e. by minimizing the sum of perpendicular distances between data points and the line.
- The fistr pricipal direction (component) is the direction of this line. Let it be a new basis vector $z_1$.
- The second principal direction (component, basis vector) $z_2$ is a direction perpendicular to $z_1$ and minimizing the distances to data points to a correspinding straight line.
- For higher dimensional observation spces, this construction is repeated.

##Principal component analysis, introduction
- PCA is a powerful and widely used linear technique in statics, signal processing, image processing, and elsewhere.
- In statistics, PCA is a method for simplifyng a multidimensional dataset to lower dimesions for analysis, visualization or data compression.
- PCA represents the data in a new coordiante system in which **basis vectors follow modes of greatest variance in the data**.
- Thus, new **basis ectros are calculated for the particular data set**.
- The price to be pains for PCA's flecibility is in higher computational requirements as comapered to, e.g. ,the fats Fourier transform.

## Derivation, $M$-dimensional case
- Suppose a **data set** comprising $N$ observations, each of $M$ variables (dimensions). Usually $N\gg M$.
- **The aim: to reduce the dimensionality** of the data so that each observation can be usefully represented with only $L$ variables, $1\leq L\leq M$.
- Data are arranged as a set of $N$ column data vectors, each representing a single observation of $M$ variables: the $n$-th observations is a column vector $\vb{x}$





*********************
# [How to reverse PCA and reconstruct original variables from several principal components?](https://stats.stackexchange.com/questions/229092/how-to-reverse-pca-and-reconstruct-original-variables-from-several-principal-com)

# [PCA Visualization  in Julia](https://plotly.com/julia/pca-visualization/#highdimensional-pca-analysis-with--plotdataframe-kindsplom)
