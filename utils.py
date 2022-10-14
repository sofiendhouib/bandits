import numpy as np

def generate_graph(n_vertices= 5, p= 0.5):
    Adj = np.zeros((n_vertices, n_vertices))
    for i in range(n_vertices):
        for j in range(i):
            Adj[i,j] = np.random.binomial(1, p)
        
    return (Adj + Adj.T)


def random_walk_laplacian(Adj):
    deg = Adj.sum(axis= 1, keepdims= True)
    invDeg = np.where(deg>0, 1/deg, 0)
    return np.eye(len(Adj)) - Adj*invDeg

def Laplacian(Adj):
    return Adj.sum(axis= 1) - Adj
