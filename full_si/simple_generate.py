# Code that generates the simulated data we require.
# Reads in the vaccinated lists (for now, don't necessarily start the epidemic with the same seeds)
# For now, don't allow for command line arguments, but this will be implemented in the future.


import os
import sys
import epidemic_utils
import pickle
from tqdm import tqdm
import argparse
import numpy as np

# Generate the data in batches...

job_id = sys.argv[1]
mode = sys.argv[2]
num_samples = int(sys.argv[3])

regen_prior = None
batch_size = 100
	
print("Generating " + str(num_samples) + " trials for " + mode)
	
# Define our paths.
path_to_output = r"data/" # Where we'll store our data.
if not os.path.exists(path_to_output):
	os.makedirs(path_to_output)
output_name = path_to_output + mode + "_data_" + str(job_id)
output_file = path_to_output + mode + "_data_" + str(job_id) + ".pkl"
path_to_original_epidemic = r"true_epidemic/" # Where we'll get our network and vaccinated list.
	

# Load in the original "true" network
true_network_file = open(path_to_original_epidemic + "true_network.pkl", "rb")
true_network = pickle.load(true_network_file)
true_network_file.close()

# Load in the initial list
initial_file = open(path_to_original_epidemic + "true_initial_list.pkl", "rb")
i_list = pickle.load(initial_file)
initial_file.close()
	
# Load in beta and time_steps from the original epidemic.
params_file = open(path_to_original_epidemic + "params.pkl", "rb")
params_dic = pickle.load(params_file)
beta = params_dic["beta"]
time_steps = params_dic["time_steps"]
gamma_params = params_dic["prior_params"]
params_file.close()
# Now, let's draw our samples.
samples = {}
times = []
curr_samples = 0

while curr_samples < num_samples:
	# M: use a min operator here -- we generate args.batch_size per iteration
	# but on the last iteration we might not have that many samples left to generate.
	curr_batch_size = min(batch_size, num_samples - curr_samples)
			
	# Need to then sample some epidemics.. Basically return a sample from
	# the function in the paper, with thetas drawn from a U(0,1)
	# Which network we use (true vs. observed) depends on the "missing" flag
	
	sample = epidemic_utils.sample_nm(curr_batch_size, true_network, i_list, time_steps, gamma_params)
			
	for key, value in sample.items(): # sample comes out with a 'theta' and an 'output' and a 'time'
		samples.setdefault(key, []).append(value)
	curr_samples += curr_batch_size
samples = {key: np.concatenate(value, axis=0) for key, value in samples.items()}
epidemic_utils.compressed_pickle(output_name, samples)	
print("Data generation finished for " + mode)
