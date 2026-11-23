import os
import scipy.io
import networkx as nx
import numpy as np
import powerlaw

#load data -> change from matrix to graph in networkX
# Data: "power" network (Watts & Strogatz 1998) from the SuiteSparse Matrix Collection,
# Newman group: https://sparse.tamu.edu/Newman/power -> place power.mtx in ./data/
A = scipy.io.mmread(os.path.join("data", "power.mtx"))
G = nx.Graph(nx.from_scipy_sparse_array(A))

#basic stats of graph -> already given in paper, just check
N = G.number_of_nodes() #number of nodes: stations, generators, transformers
L = G.number_of_edges() #lineas de transmision
k_mean = 2 * L / N #av number of lineas leave a normal station
print(N, L, round(k_mean, 2))

#max conexion, used to def lines that cant connect
Gc = G if nx.is_connected(G) else G.subgraph(max(nx.connected_components(G), key=len)).copy()

#obtain distribution of degrees P(k) (how many nodes with k connection). Fit ro exponential or power
degrees = [d for _, d in Gc.degree()]
fit = powerlaw.Fit(degrees, discrete=True)
R, p = fit.distribution_compare('power_law', 'exponential')
print(R, p)

#average shortest path and max distance between two nodes
l = nx.average_shortest_path_length(Gc)
D = nx.diameter(Gc)
print(round(l, 2), D)

#clustering coefficient
C = nx.average_clustering(Gc)
print(round(C, 4))

#build same N, L graph but with random graph to compare
G_er = nx.gnm_random_graph(N, L, seed=42)
Gc_er = G_er if nx.is_connected(G_er) else G_er.subgraph(max(nx.connected_components(G_er), key=len)).copy()

l_rand = nx.average_shortest_path_length(Gc_er)
C_rand = nx.average_clustering(Gc_er)

#compare networks
print(f"Real:      l={l:.2f}   C={C:.4f}")
print(f"Aleatorio: l={l_rand:.2f}   C={C_rand:.4f}")

#compare to Watts-Strogatz
for p in [0.001, 0.01, 0.1, 1.0]:
    G_ws = nx.watts_strogatz_graph(N, k=2, p=p, seed=42)
    Gc_ws = G_ws if nx.is_connected(G_ws) else G_ws.subgraph(max(nx.connected_components(G_ws), key=len)).copy()
    print(p, round(nx.average_clustering(Gc_ws), 4), round(nx.average_shortest_path_length(Gc_ws), 2))

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4.5))
fit.plot_ccdf(ax=ax, color='k', linewidth=1.5, label='Datos reales')
fit.power_law.plot_ccdf(ax=ax, color='r', linestyle='--', label=f'Ley de potencia ($\\gamma$={fit.power_law.alpha:.2f})')
fit.exponential.plot_ccdf(ax=ax, color='g', linestyle='-', label=f'Exponencial ($\\lambda$={fit.exponential.parameter1:.3f})')
ax.set_xlabel('k')
ax.set_ylabel('$P_c(k) = P(K \\geq k)$')
ax.legend()
plt.tight_layout()
plt.savefig('degree_distribution.png')