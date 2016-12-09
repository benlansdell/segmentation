Mumford-Shah image segmentation
===============================

The code provides a CPU (slow) implementation of an approximation to Mumford-Shah image segmentation. 

The basic idea is to fix a set of $k$ colors we are going to segment the image with. We want to do so in a way that avoids high spatial variability while maintaining sharp transitions between regions of very different color. The solution is to use a total-variation (TV) penalty. Let $c_i$ be   

Based on:
 * "A first-order primal-dual algorithm for convex problems with applications to imaging" 
Chambolle, Antonin and Pock, Thomas (2011)
Journal of Mathematical Imaging and Vision. 40(1)

Intersection of convex set projection method based on 
 * "A cyclic projection algorithm via duality"
Gaffke, Norbert and Mathar, Rudolf (1989)
Metrika. 36(1)

Unit simplex projection based on 
 * "Projection onto the probability simplex : An efficient algorithm with a
 simple proof and an application"
Wang, Weiran and Miguel, A (2013)
arXiv:1309.1541v1

Ben Lansdell
11/22/2016
