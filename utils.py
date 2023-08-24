#%%
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn import datasets
from networkx import draw_networkx_edges
from matplotlib.pyplot import quiver

def graph_Erdos_Renyi(n_vertices= 5, p= 0.5):
	Adj = np.zeros((n_vertices, n_vertices))
	for i in range(n_vertices):
		for j in range(i):
			Adj[i,j] = np.random.binomial(1, p)
		
	return (Adj + Adj.T)

def graph_rbf(node_num, dimension, gamma=None, sparsity = 0.1, clusters= 2): ##
	# From the github of Kaige Yang
	if clusters==False or clusters == None:
		node_f=np.random.uniform(low=-0.5, high=0.5, size=(node_num, dimension))
	else:
		node_f, _=datasets.make_blobs(n_samples=node_num, n_features=dimension, centers= clusters, cluster_std=0.2, center_box=(-1,1),  shuffle=False)

	adj=rbf_kernel(node_f, gamma=gamma)

	if sparsity is not None : 
		tri_inds = np.triu_indices(node_num, k= 1) 
		thd = np.quantile(adj[tri_inds], q = 1-sparsity)
		adj[adj<=thd] = 0.0

	np.fill_diagonal(adj, 0)
	return adj, node_f

def random_walk_laplacian(Adj):
	deg = Adj.sum(axis= 1, keepdims= True)
	invDeg = np.where(deg>0, 1/deg, 0)
	return np.eye(len(Adj)) - Adj*invDeg

def Laplacian(Adj, which= "random_walk"):
	return Adj.sum(axis= 1) - Adj

def normalized_Laplacian(Adj):
	deg = Adj.sum(axis= 1)
	sqrtInvDeg = np.where(deg>0, 1/deg, 0)**0.5
	return np.eye(len(Adj)) - Adj*np.outer(sqrtInvDeg, sqrtInvDeg)

def doubly_stochastic_Adjacency(Adj):
	# TODO replace by while some on rows or on colmns is far from ones vector
	for _ in range(1000):
		deg = Adj.sum(axis= 1, keepdims= True)
		invDeg = np.where(deg>0, 1/deg, 0)
		Adj *= invDeg
		deg = Adj.sum(axis= 0)
		invDeg = np.where(deg>0, 1/deg, 0)
		Adj *= invDeg
	return Adj

def quiver_graph(graph, pos, signal, alpha= 0.2):
	assert signal.shape[1] == 2; "The signal two plot must be two dimensional !"
	norms = np.linalg.norm(signal, axis= 1, keepdims= True)
	sig_norm = signal/norms
	quiver(pos[:,0], pos[:,1], sig_norm[:,0], sig_norm[:,1], norms)
	draw_networkx_edges(graph, pos = pos, alpha= alpha)
	return None

# %%
