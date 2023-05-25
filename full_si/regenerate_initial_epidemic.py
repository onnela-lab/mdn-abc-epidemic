# Code that REgenerates the initial, "true" epidemic. 
# Requires the initial epidemic to have been run already -- this really just reruns it, knowing all parameters
# Also knows the initial values.

import epidemic_utils
import networkx 
import os
import pickle
import numpy as np
import argparse


    
"""
num_nodes and avg_deg are the number of nodes and the average degree of the generated network.
network_type is a string that tells us what kind of network to generate: "ER" or "BA"
amount_vaccinated is a value 0-1 that is the amount of population that is vaccinated
initial_infected_amount is a value 0-1 that is the amount of population infected at time 0
beta is the probability of transmission on contact
vaccine_efficacy is the efficacy of the vaccine
time_steps is the number of time steps (how long the trial is.)
"""
# Generate a file to keep the data from the initial epidemic.
path_to_output = r"true_epidemic/"
path_to_original_epidemic = r"true_epidemic/"
if not os.path.exists(path_to_output):
    os.makedirs(path_to_output)
    
# Load in the original "true" network
true_network_file = open(path_to_original_epidemic + "true_network.pkl", "rb")
true_network = pickle.load(true_network_file)
true_network_file.close()
    
# Load in beta, vaccine efficacies, and time_steps from the original epidemic.
params_file = open(path_to_original_epidemic + "params.pkl", "rb")
params_dic = pickle.load(params_file)
beta = params_dic["beta"]
gamma = params_dic["gamma"]
time_steps = params_dic["time_steps"]
num_nodes = params_dic["num_nodes"]
params_file.close()
    
# Load in the initial list
initial_file = open(path_to_original_epidemic + "true_initial_list.pkl", "rb")
i_list = pickle.load(initial_file)
initial_file.close()
    
# Next, generate and store the "true" epidemic process. Retry until it fits "typical" values
true_epidemic_output = epidemic_utils.simulate_SI_epidemic(true_network, spread_type, beta, sus_efficacy, inf_efficacy, v_list, i_list, time_steps)
if check_typical:
    print("Checking typical-ness of epidemic")
    # We may want to make sure the true epidemic output is "typical"
    true_epidemic_prevalences = epidemic_utils.output_to_prevalences_single(num_nodes, v_list, true_epidemic_output)
    num_checks = 0
    max_checks = 100
    typical_vals = epidemic_utils.get_typical_values(true_network, spread_type, beta, sus_efficacy, inf_efficacy, v_list, i_list, time_steps)
    while epidemic_utils.is_typical(true_epidemic_prevalences, typical_vals) == False:
        # If our current trial is not typical, rerun it.
        true_epidemic_output = epidemic_utils.simulate_SI_epidemic(true_network, spread_type, beta, sus_efficacy, inf_efficacy, v_list, i_list, time_steps)
        true_epidemic_prevalences = epidemic_utils.output_to_prevalences_single(num_nodes, v_list, true_epidemic_output)
            
        num_checks += 1
        if num_checks > max_checks:
            print("Failed to find typical epidemic after " + str(max_checks) + " iterations.")
            break       
    
print("'True' Epidemic finished, storing results")
epidemic_file = open(path_to_output + "true_epidemic.pkl", "wb")
pickle.dump(true_epidemic_output, epidemic_file)
epidemic_file.close()

param_dic = {"num_nodes": num_nodes, "avg_deg": params_dic["avg_deg"], "network_type": params_dic["network_type"], "amount_vaccinated": params_dic["amount_vaccinated"],
             "initial_infected_amount": params_dic["initial_infected_amount"], "spread_type": spread_type, "beta": beta, "sus_efficacy": sus_efficacy, "inf_efficacy": inf_efficacy,
             "time_steps": time_steps}
param_file = open(path_to_output + "params.pkl", "wb")
pickle.dump(param_dic, param_file)
param_file.close()
