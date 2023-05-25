# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 20:55:13 2022

@author: mhw20
"""


import networkx as nx
import math
import numpy as np
import copy
import scipy.stats
import random
import matplotlib.pyplot as plt
import multiprocessing
import copy
import itertools

"""
Creating a network with a given degree distribution (k_prior) and number of nodes.
"""

def create_log_normal_graph(num_nodes, mean_degree):
	# This does assume sigma squared = 1.
	# First, get the variance.
	variance = 0.5
	mu = np.log(mean_degree) - (0.5*variance)
	# Next, draw the values.
	sigma = np.sqrt(variance)
	a = list(np.random.lognormal(mu,sigma,num_nodes))
	print("Degrees: " + str(a))	
	drawn_network = nx.expected_degree_graph(a, seed=None, selfloops = False)
	drawn_network = connect_components(drawn_network)
	max_degree = max(np.array(list(drawn_network.degree()))[:,1])
	print("Max degree is: " + str(max_degree))	
	return drawn_network


def connect_components(disconnected_graph):
	conn_graph = copy.deepcopy(disconnected_graph)
	# A method for making a fully connected graph with the fewest possible steps.
	S = [disconnected_graph.subgraph(c).copy() for c in nx.connected_components(disconnected_graph)]
	# This is a list of all subgraphs.
	if len(S) == 1:
		print("Graph is already one component")
		return disconnected_graph
	else:
		print("Number of separate components in graph: " + str(len(S)))
	# Now, draw an edge from each component to the LCC.
	largest_cc = max(nx.connected_components(disconnected_graph), key=len)
	original_edges = len(disconnected_graph.edges())
	print("Number of original edges: " + str(original_edges))
	added_edges = 0
	for i in range(len(S)):
		node_1 = random.choice(list(largest_cc))
		node_2 = random.choice(list(S[i].nodes()))
		conn_graph.add_edge(node_1, node_2)
		added_edges += 1
	print("Number of added edges for full connection: " + str(added_edges))
	S = [conn_graph.subgraph(c).copy() for c in nx.connected_components(conn_graph)]
	if len(S) == 1:
		return conn_graph
	else: 
		print("Seemed to be ERROR in making graph connected")
		return conn_graph

def create_chain_graph(num_nodes, k):
	output_graph = nx.Graph()
	for i in range(num_nodes):
		output_graph.add_node(i)
	# Then rewire them one after another.
	for i in range(num_nodes):
		if i < num_nodes - k:
			for j in range(k):
				output_graph.add_edge(i, i + j + 1)
		else:
			for j in range(num_nodes - i + (k-1)):
				output_graph.add_edge(i, j)
	return connect_components(output_graph)
		

