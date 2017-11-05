from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pylab
from time import time
import random
from scipy.stats import multivariate_normal
from time import time

class GaussianMixtureModel:
	"""
	Gaussian Mixture Model for image segmentation.
	Clustering with EM and K-Means.
	"""
	def __init__(self, image_name, K):
		self.image = pylab.imread(image_name)
		self.pixels = np.double(self.image.reshape(self.image.shape[0]*self.image.shape[1], 3))
		self.K = K
		self.Aux = np.einsum('ij,il->ijl', self.pixels, self.pixels)

	def dist_matrix(self, centroids):
		return np.sqrt(np.sum(np.square(self.pixels[:,np.newaxis,:] - centroids), axis=2))

	def k_Means(self, num_steps=20):
		self.centroids = np.ones((self.K, 3))*np.mean(self.pixels, axis=0) + np.random.multivariate_normal(np.zeros(3), 0.8*np.eye(3), size=self.K)
		D_mat = self.dist_matrix(self.centroids)
		self.labels = np.argmin(D_mat, axis=1)
		t_0 = time()
		for i in range(num_steps):
			self.centroids = np.array(map(lambda k: np.mean(self.pixels[np.where(self.labels==k)[0],:], axis=0),range(self.K)))
			D_mat = self.dist_matrix(self.centroids)
			self.labels = np.argmin(D_mat, axis=1)
		print ('K-Means, %d Steps, Time elapsed: %.2f') % (num_steps, time()-t_0)
		self.clustered_k_Means = np.uint8(np.array([self.centroids[k,:] for k in self.labels]).reshape((640,640,3)))
		self.A = np.array([self.centroids[k,:] for k in self.labels]).reshape((640,640,3))

	def Estep(self, pi, X, means, covs):
		# conditional densities of each point xi belonging to each cluster k
		P = np.array(map(lambda k: multivariate_normal.pdf(X, mean=means[k,:], cov=covs[k,:,:]), np.arange(self.K))).T
		# conditional log pdf
		lP = np.array(map(lambda k: multivariate_normal.logpdf(X, mean=means[k,:], cov=covs[k,:,:]),np.arange(self.K)))
		C = np.dot(P, pi) # normalization constant for each element i
		#matrix of responsibilities
		R = np.divide(np.dot(P,np.diag(pi)).T,C).T
		return R,lP,P

	def Mstep(self, R, X):
		# vector of weights
		pi_vec = np.mean(R, axis=0)
		# means
		mean_vec = np.divide((np.einsum('ij,il->jl',R,X)).T, np.sum(R, axis=0, keepdims=True)).T
		# covariance matrices
		sigma_vec = np.divide(np.einsum('ij,ikl->jkl',R,self.Aux).T, np.sum(R, axis=0)).T - np.einsum('ij,il->ijl', mean_vec, mean_vec) + 7e-1*np.array(map(lambda k: np.eye(X.shape[1]), np.arange(R.shape[1])))
		return pi_vec,mean_vec,np.array(sigma_vec)

	def EM(self):
		weights = np.random.random(self.K)
		pi_vec = weights/np.sum(weights)
		X = self.pixels
		means = np.array([X[i,:] for i in range(self.K)])
		covs = np.array([np.eye(3) for i in range(self.K)])
		old = means
		S = np.random.chisquare(17, size=(X.shape[0], self.K))
		R = np.divide(S, S.sum(axis=1)[:,None])
		count = 0
		t_0 = time()
		while(True):
			pi_vec,means,covs = self.Mstep(R, X)
			R, lP, P = self.Estep(pi_vec, X, means, covs)
			count+=1
			print 'Running EM Algorithm, Step: '+str(count)
			# Stopping criteria
			if np.sum(map(lambda k: np.linalg.norm(means[k,:]-old[k,:]), range(means.shape[0]))) <= 10 and count >= 10:
				break
			old = means
		print ('EM Algorithm, %d Steps, Time elapsed: %.2f') % (count, time()-t_0)
		# return the pixel matrix of the weighted means
		N = np.dot(R, means)
		# image ready for plotting
		self.clustered_EM = np.uint8(N.reshape((640,640,3)))


im_name = 'FluorescentCells.jpg'
num_clusters = 3
model = GaussianMixtureModel(im_name, num_clusters)
model.k_Means()
model.EM()

"""
Plot clustered images.
"""

fig = plt.subplot(121)
fig.title.set_text('K-Means, %d clusters' % num_clusters)
plt.imshow(model.clustered_k_Means)
plt.axis('off')
fig = plt.subplot(122)
fig.title.set_text('EM Algorithm, %d clusters' % num_clusters)
plt.imshow(model.clustered_EM)
plt.axis('off')
plt.show()
