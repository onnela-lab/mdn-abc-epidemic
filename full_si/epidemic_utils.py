"""
Running a randomized trial
Generate a network based on average degree and degree distribution.
Seed a contagion and allow it to spread until it reaches a base prevalence.
Treatment decreases the probability an individual gets affected when an infectious event occurs.
Randomly treat the individuals.
The parameter of interest here would be theta, which is the effect of the treatment.
"""

import networkx as nx
import argparse
import pickle
import numpy as np
import random
import copy
import os
import multiprocessing as mp
import itertools
import network_reconstruction
import bz2 
import pickle as cPickle
import time
import scipy.stats
import torch as th
import math


def dict_to_list(output_dic):
	output_list = []
	for k in output_dic.keys():
		output_list.extend(output_dic[k])
	return output_list

def get_SI_connections(network, i_list, s_list):
	# See how many edges connect i_nodes to s_nodes.
	count_SI = 0
	for i_node in i_list:
		neighbors = list(network.neighbors(i_node))
		if len(neighbors) > 0:
			for neighbor in neighbors:
				if neighbor in s_list:
					count_SI += 1 
	return count_SI	

def simulate_SI_gillespie(network, beta, i_list, time_steps):
	
	num_nodes = len(list(network.nodes()))
	i_nodes = copy.deepcopy(i_list)
	s_nodes = []
	for node in network.nodes():
		if not (node in i_nodes):
			s_nodes.append(node) 
	
	node_transitions = []
	for node in list(network.nodes()):
		if node in i_nodes:
			node_transitions.append(0.0)
		else: # If node is susceptible, transition probability is beta times number of infected neighbors.
			neighbors = network.neighbors(node)
			m = 0
			for neighbor in neighbors:
				if neighbor in i_nodes:
					m += 1
			node_transitions.append(m*beta)
	node_probabilities = []
	for node in list(network.nodes()):
		node_probabilities.append(node_transitions[node]/sum(node_transitions))
	time = 0
	
	# Things we need to keep track of.
	SI_connections = [] # Number of SI connections at each time.
	total_infection_events = 0 
	infection_times = [time_steps] * num_nodes
	infected_count = [] # Number of infected, corresponding to each event time.
	event_times = [] # Time of all events.
	for infected in i_list:
		infection_times[infected] = 0
	
	while True:
		rate = sum(node_transitions)
		tau = np.random.exponential(scale = 1/rate)
		
		# Time advances by tau
		time = time + tau	
	
		multi_draw = np.random.multinomial(1,node_probabilities)
		selected_node = list(multi_draw).index(1)

		# These metrics are calculated at each event.
		SI_connections.append(get_SI_connections(network, i_nodes, s_nodes))
		infected_count.append(len(i_nodes))
		event_times.append(time)

		if selected_node in s_nodes:
			i_nodes.append(selected_node)
			s_nodes.remove(selected_node)
			node_transitions[selected_node] = 0.0
			# And also mark down the recovery time.
			infection_times[selected_node] = time
			total_infection_events += 1
		else:
			if selected_node in i_nodes:
				print("ERROR: Infected nodes should not be transitioning")
			else:
				print("ERROR: Node of unknown status")
		# Recalculate transition rates.
		node_transitions = []
		for node in list(network.nodes()):
			if node in i_nodes:
				node_transitions.append(0.0)
			else:
				neighbors = network.neighbors(node)
				m = 0
				for neighbor in neighbors:
					if neighbor in i_nodes:
						m+=1
				node_transitions.append(m*beta)
		if len(i_nodes) == num_nodes: # One everyone is infected, the simulation just ends.
			break
		if sum(node_transitions) < 0.00001:
			break		
		# Next, recalculate transition probabilities
		node_probabilities = []
		for node in list(network.nodes()):
			node_probabilities.append(node_transitions[node]/sum(node_transitions))
			
	return{"i_times": infection_times, "tot_i": total_infection_events, "event_times": event_times, "infected_count": infected_count, "SI_connections": SI_connections, "final_time": time}				
	


def sample_nm(size, true_network, initial_list, time_steps, gamma_params):
	# Given parameters, we sample a number of epidemics, given size.
	# Note that this uses a gamma prior as in Bu 2020.
   
	sample = []
	# Assume that the gamma prior parameters is of form [a_beta, b_beta, a_gamma, b_gamma]. 
	if len(gamma_params)!=2:
		print("ERROR: gamma_params should be of form [a_beta, b_beta]")
	beta_draws = np.random.gamma(shape = gamma_params[0], scale = 1/gamma_params[1], size = size)
	beta_draws = np.reshape(beta_draws, (size,1))
	parameters = beta_draws

	# Uniform prior not being used.
	#parameters = np.random.uniform(low = 0, high = 1, size = (size,2))
	for i in range(size):
		num_nodes = len(list(true_network.nodes()))
		op = simulate_SI_gillespie(true_network, parameters[i][0],initial_list, time_steps)
		output = op["i_times"]
		sample.append(output)
	return {
		'theta': parameters,
		'output': np.array(sample),
	}
"""
Functions to convert epidemic simulation output (a list of lists, with infection times for each individual)
into other forms.
"""

def output_to_prevalences_single(num_nodes, output_vec):
	num_infected = 0 
	for i in range(len(output_vec)):
		if output_vec[i] == 1: # If this individual was infected.. it should have an infection time less than the greatest.
			num_infected += 1
	
	return {"prevalence": num_infected/len(output_vec)}

"""
Functions for making sure that the "original" epidemic run is considered typical.
"""

def get_typical_values_gillespie(network, beta, i_list, time_steps):
	all_prev = []
	for i in range(400):
		epidemic_output = simulate_SI_gillespie(network, beta, i_list, time_steps)
		prevalences = output_to_prevalences_single(len(list(network.nodes())), epidemic_output["i_times"])
		all_prev.append(prevalences["prevalence"])
	return {"p_mean": np.mean(np.array(all_prev)), "p_sd": np.std(np.array(all_prev))}	
def is_typical(proposed_output_prevalences, typical_values):
	
	if np.abs(proposed_output_prevalences["prevalence"] - typical_values["p_mean"]) > 1 * typical_values["p_sd"]:
		return False
	return True

"""
Function for creating a summary statistic.
"""
def summarize_network(network):
	degrees = []
	for node in network:
		degrees.append(network.degree())
	return degrees

def compressed_pickle(title, data):
        with bz2.BZ2File(title + ".pbz2", "w") as f:
                cPickle.dump(data, f)

# Pickle a file and then compress it into a file with extension
# Load any compressed pickle file
def decompress_pickle(file):
        data = bz2.BZ2File(file, "rb")
        data = cPickle.load(data)
        return data

