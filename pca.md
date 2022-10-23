# PCA
## Links:
- [Guide To Image Reconstruction Using Principal Component Analysis](https://analyticsindiamag.com/guide-to-image-reconstruction-using-principal-component-analysis/)

Principal Component Analysis belongs to a class of linear transforms bases on statistical techniques. This method provides a powerful toll for data analysis and pattern recognition, which is often preferred in signal an imagage processing as a technique of data compression, data dimension reduction, or decorrelation. PCA is an unsipervides learning method similar to clustering. It finds patterns without prior knowledge about whether the samples come from different treatment groups or essential differences. The objective is pursued by analysing principal components where we can perceive relationships that would otherwise remain hidden in higher dimensions.
The representation processed must be such that the loss of information must be minimal after discarding the higher dimensions.
The goal of the methods is to reorient the data so that a multitude of original variables can be summarized with relatively few "factors" or "components" that capture the maximum possible information from the original variables.


The output of PCA is principal components, which are less than or equal to the number of original variables. Less, in a case when we wish to discard or reduce the dimensions in our dataset. 

The PCA method provides an alternative way to this method, where the matrix A is replaced by matrix Al where only l largest (instead of n) eigenvalues are used for its formation. A vector of reconstructed variables is then given by relation. A selected real picture P and Its three reconstructed components are obtained accordingly for each eigenvalue and presented. The comparison of the intensity of images obtained from the original image as the weighted colour sum is evaluated as the first principal component. The variance figures for each principal component are present in the eigenvalue list. These indicate the amount of variation accounted for by each component within the feature space.

### Getting Started with the Code
**Importing labraries**

	import numpy as np
	from matplotlib.image import imread
	import matplotlib.pyplot as plt

Here we sill using imread from matplotlib to import the image as a matrix

**Setting image path**

	my_image=imread("/path/.png")
	print(my_image.shape)

**Displaying the image**

	plt.figure(figsize=[12,8])
	plt.imshow(my_image)

The image being processed is a coloured image and hance has data in 3 channels-Red, Green, Blue. Therefore the sape of the data -525x700x3

**Processing the image**

Let us now start with our image processing. Here first, we will be grayscaling our image, and then we'll perfomr PCA on the matrix with all the components. We will also create and look at the scree plot to assess how many components we could retain and how much cumulative variance they capture.

**Greyscaling the image**

	image_sum = my_image.sum(axis=2)
	print(image_sum.shape)

	new_image = image_sum/image_sum.max()
	print(new_image.max())

	plt.figure(figsize=[12,8])
	plt.imshow(new_image, cmap=plt.cm.gray)

**Creating scree plot**
	
	from sklearn.decomposition import PCA, IncrementalPCA
	pca = PCA()
	pca.fit(new_image)

**Getting the cumulative variance**

	var_cumu = np.cumsum(pca.explained_variance_ratio_)*100

**How many PCs explain 95% of the variance?**

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

Now let's reconstruct the image using only 23 components and see if our reconstructed image comes out to be visually different from the original image

**Reconstructing using Inverse Transform**

	ipca = IncrementalPCA(n_components=k)
	image_recon = ipca.inverse_transform(ipca.fit_transform(new_image))

**Plotting the reconstructed image**

	plt.figure(figsize=[12,8])
	plt.imshow(image_recon, cmap=plt.cm.gray)

As we can observe, there is a relative difference now. We shall try with a different value of components to check if that maked a difference in the missin clariry and help capture finer details in the visuals.

**Function to reconstruct and plot image for a given number of components**

	def plot_at_k(k):
		ipca = IncrementalPCA(n_components=k)
		image_recon = ipca.inverse_transform(ipca.fit_transform(new_image))
		plt.imshow(image_recon, cmap=plt.cm.gray)

	k = 150
	plt.figure(figsize=[12,8])
	plot_at_k(100)

We can observe that, yes, the number of principal components do make a difference!

Plotting the same for different numbers of components to compare the exact relative difference,

**Setting different amount of K**

	ks = [10, 25, 50, 100, 150, 250]

	plt.figure(figsize=[15,9])

	for i in range(6):
		plt.subplot(2,3,i+1)
		plot_at_k(ks[i])
		plt.title("Components: "+str(ks[i]))

	plt.subplots_adjust(wspace=0.2, hspace=0.0)
	plt.show()

Using PCA for Image Reconstruction, we can also segregate between the amounts of RGB present in an image,

	import cv2
	img = cv2.cvtColor(cv2.imread('flor.jpeg'))


- [Principal Component Analysis (PCA) applied to images (pdf)](http://people.ciirc.cvut.cz/~hlavac/TeachPresEn/11ImageProc/15PCA.pdf)
- [How to reverse PCA and reconstruct original variables from several principal components?](https://stats.stackexchange.com/questions/229092/how-to-reverse-pca-and-reconstruct-original-variables-from-several-principal-com)
