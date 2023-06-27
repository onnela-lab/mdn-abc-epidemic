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
import _pickle as cPickle

#################
# Basic utilities
#################
def get_output_len(obs_times):
	# Based on times of observations, get an output length.
	total_observations = 0
	for k in obs_times.keys():
		total_observations += len(obs_times[k])
	return total_observations

def dict_to_list(output_dic):
	# Breaking down a dict's values into one ordered list.	
	output_list = []
	for k in output_dic.keys():
		output_list.extend(output_dic[k])
	return output_list

################
# Epidemic
################

def generate_test_times(network, cadences, adherences, initial_list, time_steps):
	# Given cadences and adherences, generate a dictionary of test times 
	# that contains the times of testing for each node.
	test_times = {}
	num_nodes = len(list(network.nodes))
	for node in range(num_nodes):
		test_times[node] = []
	for node in initial_list: # The sources are known.
		test_times[node].append(0)

	for node in range(num_nodes):
		start_day = int(node)%7 # Assign each node a day of the week for testing.
		proposed_days = [start_day]
		next_day = start_day + cadences[node]
		while next_day < time_steps:
			proposed_days.append(next_day)
			next_day = next_day + cadences[node]
		for day in proposed_days:
			if random.uniform(0,1) <= adherences[node]:
				test_times[node].append(day)
	return test_times	

def simulate_SIR_gillespie(network, beta, gamma, i_list, test_times, time_steps):

	num_nodes = len(list(network.nodes()))
	node_status = {}
	for i in range(num_nodes):
		if i in i_list:
			node_status[i] = "i"
		else:
			node_status[i] = "s"

	node_transitions = []
	for node in list(network.nodes()):
		if node_status[node] == "i":
			node_transitions.append(gamma)
		elif node_status[node] == "r":
			node_transitions.append(0.0)
		else: # If node is susceptible, transition probability is beta times number of infected neighbors.
			neighbors = network.neighbors(node)
			m = 0
			for neighbor in neighbors:
				if node_status[neighbor] == "i":
					m += 1
			node_transitions.append(m*beta)
	node_probabilities = []
	for node in list(network.nodes()):
		node_probabilities.append(node_transitions[node]/sum(node_transitions))
	time = 0
	# Things we need to keep track of.
	total_infection_events = 0
	total_recovery_events = 0
	infection_times = [time_steps] * num_nodes
	recovery_times = [time_steps] * num_nodes
	for infected in i_list:
		infection_times[infected] = 0
	# Gillespie simulation begins here.
	while time < time_steps:
		rate = sum(node_transitions)
		tau = np.random.exponential(scale = 1/rate)
		# Time advances by tau at each step.
		time = time + tau

		multi_draw = np.random.multinomial(1,node_probabilities)
		selected_node = list(multi_draw).index(1)

		if node_status[selected_node] == "i":
			node_status[selected_node] = "r"
			recovery_times[selected_node] = time
			total_recovery_events += 1
		elif node_status[selected_node] == "s":
			node_status[selected_node] = "i"
			infection_times[selected_node] = time
			total_infection_events += 1
		else:
			if node_status[selected_node] == "r":
				print("ERROR: Recovered nodes should not be transitioning")
			else:
				print("ERROR: Node of unknown status")
		# Recalculate transition rates.
		node_transitions = []
		for node in list(network.nodes()):
			if node_status[node] == "i":
				node_transitions.append(gamma)
			elif node_status[node] == "r":
				node_transitions.append(0.0)
			else:
				neighbors = network.neighbors(node)
				m = 0
				for neighbor in neighbors:
					if node_status[neighbor] == "i":
						m+=1
				node_transitions.append(m*beta)
		# End the epidemic if no more transitions are possible.
		if total_recovery_events == num_nodes:
			break
		if sum(node_transitions) < 0.00001:
			break
		# Lastly, recalculate transition probabilities for the next iteration.
		node_probabilities = []
		for node in list(network.nodes()):
 			node_probabilities.append(node_transitions[node]/sum(node_transitions))
	
	# Finally, let's make the output_dic, the same as time_inference.
	output_dic = {}
	for node in list(network.nodes()):
		output_dic[node] = []
	
	# The input of test_times has nodes as keys, and their test times as values.
	positive_results = 0
	negative_results = 0
	for node in test_times.keys():
		for t in test_times[node]:
			if (infection_times[node] < t) and (recovery_times[node]>t): # We test positive if we were infected before t, but we recovered after t.
				output_dic[node].append(1)
				positive_results += 1
			else:
				output_dic[node].append(0)
				negative_results += 1
	output_vec = dict_to_list(output_dic) # The vector of test results that is our main data.
	return{"results": output_vec, "i_times": infection_times, "r_times": recovery_times, "tot_i": total_infection_events, "tot_r": total_recovery_events, "positive_results": positive_results, "negative_results": negative_results}

def get_trajectories(i_times, r_times, time_steps): 
	# Returns the number of infected per time step.
	# Allows for posterior predictive plots of infection curves.
	res_pt = 1000
	times = np.linspace(0.001,time_steps, num = res_pt)
	infected_count = []
	for t in times:
		i_at_t = 0
		for node in range(len(i_times)):
			if i_times[node] < t and r_times[node] > t:
				i_at_t += 1
		infected_count.append(i_at_t)
	return{"times": times, "infected": infected_count}

def sample_nm(size, true_network, initial_list, test_times, time_steps, prior_params, prior_dist):
	# Given parameters, we sample a number of epidemics, given size.
	# Allows for prior to be either gamma distributed or uniform.   
	sample = []
	times = []
	# Draw from prior for parameters (uniform prior).
	if prior_dist == "uniform":
		parameters = np.random.uniform(low = prior_params[0], high = prior_params[1], size = (size,2))
	elif prior_dist == "gamma": 
		# Here, the parameters should be of form [a_beta, b_beta, a_gamma, b_gamma]
		beta_draws = np.random.gamma(shape = prior_params[0], scale = 1/prior_params[1], size = size)
		gamma_draws = np.random.gamma(shape = prior_params[2], scale = 1/prior_params[3], size = size)
		beta_draws = np.reshape(beta_draws, (size,1))
		gamma_draws = np.reshape(gamma_draws, (size,1))
		parameters = np.concatenate((beta_draws, gamma_draws), axis=1)
	else:
		print("Unknown prior type")
	for i in range(size):
		num_nodes = len(list(true_network.nodes()))
		op = simulate_SIR_gillespie(true_network, parameters[i][0], parameters[i][1], initial_list, test_times, time_steps)
		# op has two elements. The first is the output that we'll use to train our NN. The second is the true infection times, used for cosmetic purposes.
		sample.append(op["results"])
		times.append(op["i_times"])
	return {
		'theta': parameters,
		'output': np.array(sample),
		'times': np.array(times)
	}
##################
# Functions for making sure 
# that the "original" epidemic run is considered typical, 
# at least in terms of total infections.
# Typically does not matter; this just helps us avoid really "extreme"
# epidemics.
##################
def get_typical_values(network, beta, gamma, i_list, test_times, time_steps):
	# Run 100 trials, and get means and standard errors for infection times..
	all_prev = []
	for i in range(400):
		epidemic_output = simulate_SIR_gillespie(network, beta, gamma, i_list, test_times, time_steps)
		all_prev.append(epidemic_output["tot_i"])
	# Now that we have prevalences for both arms for some number of trials, let's get the prevalences and 
	return {"p_mean": np.mean(np.array(all_prev)), "p_sd": np.std(np.array(all_prev))}	

def is_typical(proposed_output_prevalence, typical_values):
	# We judge if a run is typical if the prevalence is within one sd of the means of previous trials.
	if np.abs(proposed_output_prevalence - typical_values["p_mean"]) > 1 * typical_values["p_sd"]:
		return False
	return True



###################
# A few utilities for
# pickling things more efficiently.
###################
def compressed_pickle(title, data):
	with bz2.BZ2File(title + ".pbz2", "w") as f:
	    cPickle.dump(data, f)

# Pickle a file and then compress it into a file with extension
# Load any compressed pickle file
def decompress_pickle(file):
	data = bz2.BZ2File(file, "rb")
	data = cPickle.load(data)
	return data

