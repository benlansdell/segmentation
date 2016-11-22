#!/usr/bin/env python
"""
###############################
Mumford-Shah image segmentation
###############################

Based on 

"A first-order primal-dual algorithm for convex problems with applications to imaging"
Chambolle, Antonin and Pock, Thomas (2011)
Journal of Mathematical Imaging and Vision. 40(1)

"""

import numpy as np 
import cv2 
from sklearn.cluster import KMeans

def chambolle(x, y, tau, sigma, theta, K, K_star, f, res_f, res_G, n_iter = 100, eps = 1e-6):
	x_bar = x.copy()
	for n in range(n_iter):
		x_old = x 
		if (np.linalg.norm(x-x_old) < eps) and (n > 0):
			break 
		y = res_F(y + sigma*K(x_bar))
		x = res_G(x + tau*K_star(y))
		x_bar = x + theta*(x - x_old)
	return x

def grad(x):
	asdf

def div(x):
	asdf


def mumford(fn_in):
	nc = 5
	#Params from "A first-order primal-dual algorithm for convex problems with applications to imaging"
	theta = 1
	tau = 0.01
	h = 
	L2 = 8/h**2
	sigma = 1/(L2 * tau)
	lmda = 5

	#Load image
	fn_in = './jellyfish.jpg'
	img = cv2.imread(fn_in)
	#Resize a bit...
	img = cv2.resize(img, )

	#K-means clustering to get colors and for comparison 
	nx = img.shape[0]
	ny = img.shape[1]
	N = nx*ny
	X = img.reshape(N, -1, 3).squeeze()
	#Cluster image to get 'mean' clusters
	km = KMeans(n_clusters = nc).fit(X)
	centers = km.cluster_centers_
	cl = km.predict(X).reshape((nx, ny))
	#Paint the image by the clusters
	km_img = np.zeros(img.shape)
	for i in range(nx):
		for j in range(ny):
			km_img[i,j,:] = centers[cl[i,j],:]
	km_img = km_img.astype(np.uint8)
	#cv2.imshow('test', km_img)
	#cv2.waitKey()

	#Generate resolvents and such
	res_F = dykstra
	res_G = project_simplex
	K = grad 
	K_star = div

	#Generate set of images, f_l, that are the error measures for each pixel and 
	#each color, c_l, obtained from k-means
	f = 

	#Init u to live in probability simplex 
	u = res_G(np.zeros((nx, ny, nc)))
	p = res_F(grad(u))

	#Run chambolle algorithm
	u_s = chambolle(u, p, tau, sigma, theta, K, K_star, f, res_f, res_G, n_iter = 100)

	#Take argmax of u tensor to obtain segmented image
	#Paint the image by the cluster colors
	ms_img = np.zeros(img.shape)
	for i in range(nx):
		for j in range(ny):
			col = np.argmax(u_s[i,j,:])
			ms_img[i,j,:] = centers[col,:]
	ms_img = ms_img.astype(np.uint8)

	return ms_im 