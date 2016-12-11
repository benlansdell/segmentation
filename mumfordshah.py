#!/usr/bin/env python
import numpy as np 
import cv2 
from sklearn.cluster import KMeans

import argparse
from os.path import basename
import os.path 
import os 

def J1(x, h):
	return np.sum(np.linalg.norm(grad(x,h), axis = 2))

def chambolle(x, y, tau, sigma, theta, K, K_star, f, res_F, res_G, j_tv, n_iter = 100, eps = 1e-6, im_updates = 0, im_out = None, centers = None):
	x_bar = x.copy()
	x_old = x.copy()
	save_progress = im_updates > 0 and centers is not None and im_out is not None
	print('====================================================================\nIter:\tdX:\t\tJ(u):\t\tf:\t\tPrimal objective:')
	for n in range(n_iter):
		err = np.linalg.norm(x-x_old)
		ju = j_tv(x)
		fu = np.sum(f*x)
		obj = fu + ju
		print('%d\t%e\t%e\t%e\t%e'%(n, err, ju, fu, obj))
		if (err < eps) and (n > 0):
			break
		x_old = x.copy()
		y = res_F(y + sigma*K(x_bar))
		x = res_G(x - tau*(K_star(y)+f))
		x_bar = x + theta*(x - x_old)

		#If im_updates is greater than 0 then we output our progress every
		#im_updates iterations
		if save_progress and (n%im_updates) == 0:
			#Print to im_out
			ms_img = np.dot(x,centers)
			ms_img = ms_img.astype(np.uint8)
			ms_img = ms_img.astype(np.uint8)
			cv2.imwrite('%s_n_%04d.png'%(im_out,n), ms_img)
	return x

def grad(u, h):
	k = u.shape[2]
	p = np.zeros((u.shape[0], u.shape[1], 2, k))
	for i in range(k):
		p[0:-1, :, 0, i] = (u[1:, :, i] - u[0:-1, :, i])/h
		p[:, 0:-1, 1, i] = (u[:, 1:, i] - u[:, 0:-1, i])/h
	return p 

def div(p, h):
	k = p.shape[3]
	u = np.zeros((p.shape[0], p.shape[1], k))
	for i in range(k):
		#u[0:-1,:,i]  = (p[1:, :, 0, i] - p[0:-1, :, 0, i])/h
		#u[:,0:-1,i] += (p[:, 1:, 1, i] - p[:, 0:-1, 1, i])/h
		u[1:,:,i]  = (p[1:, :, 0, i] - p[0:-1, :, 0, i])/h
		u[:,1:,i] += (p[:, 1:, 1, i] - p[:, 0:-1, 1, i])/h
	return u 

def project_balls(p):
	#print 'Projection onto unit balls'
	pt = np.transpose(p, (0,1,3,2))
	n = np.linalg.norm(pt, axis = 3)
	d = np.maximum(2*n, 1)
	for i in range(pt.shape[3]):
		pt[:,:,:,i] = pt[:,:,:,i]/d
	#		pt[n>0.5,i] = pt[n>0.5,i]/(2*n[n>0.5])
	p = np.transpose(pt, (0,1,3,2))
	return p 

#Project onto intersection of unit balls
def project_balls_intersect(p):
	eps_p = 1e-6
	n_batch = 200
	#print 'Projection onto intersection of unit balls'
	p0 = p 
	k = p.shape[3]
	r = k*(k-1)/2
	#Not sure what to set this to.....
	n_iter = r
	pairs = np.zeros((r,2), dtype = int)
	c = 0
	for i in range(k-1):
		for j in range(i+1,k):
			pairs[c,:] = [i,j]
			c += 1

	def proj(x, c):
		[i,j] = pairs[c,:]
		a = x[:,:,:,i]
		b = x[:,:,:,j]
		d = np.maximum(np.linalg.norm(a-b, axis = 2),1)
		xa0 = a[:,:,0]*(1+1/d)/2 + b[:,:,0]*(1-1/d)/2
		xa1 = a[:,:,1]*(1+1/d)/2 + b[:,:,1]*(1-1/d)/2
		xb0 = a[:,:,0]*(1-1/d)/2 + b[:,:,0]*(1+1/d)/2
		xb1 = a[:,:,1]*(1-1/d)/2 + b[:,:,1]*(1+1/d)/2
		x[d>1,0,i] = xa0[d>1]
		x[d>1,1,i] = xa1[d>1]
		x[d>1,0,j] = xb0[d>1]
		x[d>1,1,j] = xb1[d>1]
		return x

	i = np.zeros((p.shape + (r,)))
	errs = np.zeros(r)
	m = 0
	err = eps_p + 1e6
	while (err > eps_p) and m < n_batch:
		print 'Batch', m, 'errors', errs
		n = 0
		while n < n_iter:
			p = proj(p0 - i[:,:,:,:,n%r], n%r)
			errs[n%r] = np.linalg.norm(p-p0)
			i[:,:,:,:,n%r] = p - (p0 - i[:,:,:,:,n%r])
			p0 = p
			n += 1
		m += 1
		err = np.max(errs)
	return p 

def project_simplex(u):
	(ny, nx, k) = u.shape

	def proj_prob(xv):
		x = np.array(xv)
		if not len(x.shape) == 1:
			return 1.
		D = x.shape[0]
		uv = np.sort(x)[::-1]
		vv = uv + np.array([1./j - np.sum(uv[0:j])/float(j) for j in range(1,D+1)])
		rho = np.max(np.where(vv > 0))
		lmbda = (1 - np.sum(uv[0:rho+1]))/float(rho+1)
		xp = np.maximum(x + lmbda, 0)
		return xp 

	#Quite slow........could be parallelized 
	for i in range(ny):
		for j in range(nx):
			u[i,j,:] = proj_prob(np.squeeze(u[i,j,:]))
	return u 

def mumford(fn_in, l=5):
	nc = 16
	#Params from [1]
	theta = 1
	tau = 0.05

	h = 1 			#Not sure this is right...
	L2 = 8/h**2
	sigma = 1/(L2 * tau)

	lmda = l		#Not sure what this should be set to....
	ny = 200

	#Test code
	#fn_in = './black_white_orange.png'
	fn_in = './butterfly.png'

	bn = basename(os.path.splitext(fn_in)[0])
	dr = './progress_frames/'
	if not os.path.exists(dr):
	    os.makedirs(dr)

	#Load image
	img = cv2.imread(fn_in)
	(iny,inx) = img.shape[0:2]
	img = cv2.resize(img, (int(inx*(ny/float(iny))), ny))

	#K-means clustering to get colors and for comparison 
	ny = img.shape[0]
	nx = img.shape[1]
	N = nx*ny
	X = img.reshape(N, -1, 3).squeeze()
	#Cluster image to get 'mean' clusters
	km = KMeans(n_clusters = nc).fit(X)
	centers = km.cluster_centers_
	cl = km.predict(X).reshape((ny, nx))
	#Paint the image by the clusters
	km_img = np.zeros(img.shape)
	for i in range(ny):
		for j in range(nx):
			km_img[i,j,:] = centers[cl[i,j],:]
	km_img = km_img.astype(np.uint8)
	cv2.imwrite('%s_kmeans.png'%bn, km_img)
	#cv2.waitKey()

	#Generate resolvents and such
	#res_F = project_balls_intersect
	res_F = project_balls
	res_G = project_simplex
	K = lambda x: grad(x, h)
	K_star = lambda x: -div(x, h)
	j_tv = lambda x: J1(x, h)

	#Generate set of images, f_l, that are the error measures for each pixel and 
	#each color, c_l, obtained from k-means
	f = np.zeros((ny, nx, nc))
	for c in range(nc):
		for i in range(ny):
			for j in range(nx):
				f[i,j,c] = lmda*(np.linalg.norm(img[i,j,:]-centers[c,:]))**2/2

	#Init u, p
	u = res_G(np.zeros((ny, nx, nc)))
	p = res_F(K(u))

	#Start with a better initial guess based on k-means clustering

	#Test
	#u = np.zeros((ny, nx, nc))
	#p = np.zeros((ny, nx, 2, nc))

	#Run chambolle algorithm
	n_iter = 300
	u_s = chambolle(u, p, tau, sigma, theta, K, K_star, f, res_F, res_G, j_tv, n_iter = n_iter,\
		im_updates = 3, im_out = '%s/%s_MS_lambda_%.02e_niter_%04d'%(dr,bn,l,n_iter), centers = centers)

	#Take argmax of u tensor to obtain segmented image
	#Paint the image by the cluster colors
	#ms_img = np.zeros(img.shape)
	#for i in range(ny):
	#	for j in range(nx):
	#		col = np.argmax(u_s[i,j,:])
	#		ms_img[i,j,:] += centers[col,:]
	ms_img = np.dot(u_s,centers)
	ms_img = ms_img.astype(np.uint8)
	cv2.imwrite('%s_MS_lambda_%.02e_niter_%04d.png'%(bn,l,n_iter), ms_img)
	return ms_img

if __name__ == '__main__':

	usage = """mumfordshah.py [input_image]

###############################
Mumford-Shah image segmentation
###############################

Based on 

[1] "A first-order primal-dual algorithm for convex problems with applications to imaging"
Chambolle, Antonin and Pock, Thomas (2011)
Journal of Mathematical Imaging and Vision. 40(1)

Intersection of convex set projection method based on 

[2] "A cyclic projection algorithm via duality"
Gaffke, Norbert and Mathar, Rudolf (1989)
Metrika. 36(1)

Unit simplex projection based on 

[3] "Projection onto the probability simplex : An efficient algorithm with a
 simple proof and an application"
Wang, Weiran and Miguel, A (2013)
arXiv:1309.1541v1

Ben Lansdell. 2016
"""

	parser = argparse.ArgumentParser()
	#parser.add_argument('fn_in', default='./jellyfish.jpg', 
	#	help='input video file, any format readable by OpenCV')
	parser.add_argument('fn_in', default='./butterfly_part.png', 
		help='input video file, any format readable by OpenCV')
	parser.add_argument('-lambda', default=1, dest='l', type = float,
		help='Regulaization term. Lower means smoother, higher means closer to image')
	args = parser.parse_args()

	#Set up multiprocessing
	print '===================================================================='
	print 'Chambolle proximal point algorithm for mumford-shah image segmentation'
	print 'Regularization (smaller = smoother): lambda =', args.l
	mumford(args.fn_in, args.l)
