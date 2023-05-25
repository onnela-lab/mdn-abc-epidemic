# Code that generates the initial, "true" epidemic. Only need to run once, to generate the appropriate files.

import epidemic_utils
import networkx 
import os
import pickle
import numpy as np
import argparse
import sys
import network_reconstruction

num_nodes = 100
network_type = "LN"
prior_param = [2,4] # parameters for the prior.
beta = 0.15
time_steps = 100
check_typical = True
mean_degree = 4

	
"""
	num_nodes and avg_deg are the number of nodes and the average degree of the generated network.
	network_type is a string that tells us what kind of network to generate: "ER" or "BA"
	initial_infected_amount is a value 0-1 that is the amount of population infected at time 0
	beta is the probability of transmission on contact
"""
	
	
# Generate a file to keep the data from the initial epidemic.
path_to_output = r"true_epidemic/"
if not os.path.exists(path_to_output):
	os.makedirs(path_to_output)
		
# First, generate and store the "true" network via reconstruction.
# If regeneration run, just load an old network.
if network_type == "BA":
	if mean_degree%2 != 0:
		print("ERROR: BA graph has m = mean_degree/2. Choosing an odd mean degree causes issues.")
	m = int(mean_degree/2)
	true_network = network_reconstruction.connect_components(networkx.barabasi_albert_graph(num_nodes, m))
elif network_type == "ER":
	prob = mean_degree/num_nodes
	true_network = networkx.erdos_renyi_graph(num_nodes, prob)
	# Then make sure the network is fully connected.	
	true_network = network_reconstruction.connect_components(true_network)
elif network_type == "LN":
	true_network = network_reconstruction.create_log_normal_graph(num_nodes, mean_degree)	
	true_network = network_reconstruction.connect_components(true_network)
elif network_type == "chain":
	true_network = network_reconstruction.create_chain_graph(num_nodes, k = 2)

network_file = open(path_to_output + "true_network.pkl", "wb")
pickle.dump(true_network,network_file)
network_file.close()
print("Generated and stored true network")
# Just pick one node as the origin.
initial_number = 1
i_list = [0]
initial_infected_amount = initial_number
print("Seeded " + str(initial_number) + " infected individuals")
initial_file = open(path_to_output + "true_initial_list.pkl", "wb")
pickle.dump(i_list, initial_file)
initial_file.close()


# Next, generate and store the "true" epidemic process. Retry until it fits "typical" values
true_epidemic_output = epidemic_utils.simulate_SI_gillespie(true_network, beta, i_list, time_steps)
if check_typical:
	print("Checking typical-ness of epidemic")
	# We may want to make sure the true epidemic output is "typical"
	true_epidemic_prevalences = epidemic_utils.output_to_prevalences_single(num_nodes, true_epidemic_output["i_times"])
	num_checks = 0
	max_checks = 100
	typical_vals = epidemic_utils.get_typical_values_gillespie(true_network, beta, i_list, time_steps)
	while epidemic_utils.is_typical(true_epidemic_prevalences, typical_vals) == False:
		# If our current trial is not typical, rerun it.
		true_epidemic_output = epidemic_utils.simulate_SI_gillespie(true_network, beta, i_list, time_steps)
		true_epidemic_prevalences = epidemic_utils.output_to_prevalences_single(num_nodes, true_epidemic_output["i_times"])
			
		num_checks += 1
		if num_checks > max_checks:
			print("Failed to find typical epidemic after " + str(max_checks) + " iterations.")
			break
print("Took " + str(num_checks) + " trials to get a suitable run.")
true_epidemic_nn_input = true_epidemic_output["i_times"]

max_time = np.max(true_epidemic_output["i_times"])
print("The last individual infected was at time " + str(max_time))

total_infected = true_epidemic_output["tot_i"]
print("Total number of people infected by the end: " + str(total_infected) + " out of " + str(num_nodes))

print("'True' Epidemic finished, storing results")
epidemic_file = open(path_to_output + "true_epidemic.pkl", "wb")
pickle.dump(true_epidemic_output, epidemic_file)
epidemic_file.close()

nn_input_file = open(path_to_output + "true_nn_input.pkl", "wb")
pickle.dump(true_epidemic_nn_input, nn_input_file)
nn_input_file.close()
	
param_dic = {"num_nodes": num_nodes, "mean_degree": mean_degree, "network_type": network_type, 
			 "initial_infected_amount": initial_infected_amount, "prior_params": prior_param,
			 "beta": beta, "time_steps": time_steps}
param_file = open(path_to_output + "params.pkl", "wb")
pickle.dump(param_dic, param_file)
param_file.close()
	
