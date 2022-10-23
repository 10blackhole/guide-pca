# Pricipal Component Analysis (PCA)
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

The image being processed is a coloured image and hance has data in 3 channels-Red, Green, Blue. Therefore the shape of the data (shape of data)

**Processing the image**

Let us now start with our image processing. Here first, we will be grayscaling our image, and then we'll perform PCA on the matrix with all the components. We will also create and look at the scree plot (In multivariate statistics, a scree plot is a line plot of the eigenvalues of factors or principal components in an analysis) to assess how many components we could retain and how much cumulative variance they capture.

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

Now let's reconstruct the image using only (#) components and see if our reconstructed image comes out to be visually different from the original image

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
	img = cv2.cvtColor(cv2.imread('/path/.image'), cv2.COLOR_BGR2RGB)
	plt.imshow(img)
	plt.show()

**Splittin into channels**

	blue, green, red = cv2.split(img)

**Plotting the images**

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

A particular image channel can also be converted into a data frame for further processing,

	import numpy as np
	import pandas as pd

**Creating dataframe from blue presence in image**

	blue_chnl_df = pd.DataFrame(data=blue)
	blue_chnl_df

The data fot each color presence can also be fit and transformed to a particular number of components for checking the variance of each color presence,

**Scaling data between 0 to 1**
	
	df_blue = blue/255
	df_green = green/255
	df_red = red/255

**Setting a reduced number of components**

	pca_b = PCA(n_components=50)
	pca_b.fit(df_blue)
	trans_pca_b = pca_b.transform(df_blue)
	pca_g = PCA(n_components=50)
	pca_g.fit(df_green)
	trans_pca_g = pca_g.transform(df_green)
	pca_r = PCA(n_components=50)
	pca_r.fit(df_red)
	trans_pca_r = pca_r.transform(df_red)

**Transforming shape**
	
	print(trans_pca_b.shape)
	print(trans_pca_r.shape)
	print(trans_pca_g.shape)

**Checking variance after reduced components**

	print(f"Blue Channel : {sum(pca_b.explained_variance_ratio_)}")
	print(f"Green Channel: {sum(pca_g.explained_variance_ratio_)}")
	print(f"Red Channel  : {sum(pca_r.explained_variance_ratio_)}")

Output(example):

	Blue Channel : 0.9835704508744926
	Green Channel: 0.9794100254497594
	Red Channel  : 0.9763416610407115

We can observe that by using 50 components we can keep around 98% of the variance in the data!



- [Principal Component Analysis (PCA) applied to images (pdf)](http://people.ciirc.cvut.cz/~hlavac/TeachPresEn/11ImageProc/15PCA.pdf)
- [How to reverse PCA and reconstruct original variables from several principal components?](https://stats.stackexchange.com/questions/229092/how-to-reverse-pca-and-reconstruct-original-variables-from-several-principal-com)
