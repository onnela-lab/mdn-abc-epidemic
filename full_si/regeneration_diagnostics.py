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
import torch as th
import matplotlib.pyplot as plt
import seaborn as sns
# Generate the data in batches...

num_samples = int(sys.argv[1])
batch_size = 100
path_to_original_epidemic = "true_epidemic/"
print("Reenerating " + str(num_samples))
# Load in the original "true" network
true_network_file = open(path_to_original_epidemic + "true_network.pkl", "rb")
true_network = pickle.load(true_network_file)
true_network_file.close()
# Load in the initial list
initial_file = open(path_to_original_epidemic + "true_initial_list.pkl", "rb")
i_list = pickle.load(initial_file)
initial_file.close()
# Load in initial results
nn_input_file = open(path_to_original_epidemic + "true_nn_input.pkl", "rb")
true_nn_input = pickle.load(nn_input_file)
nn_input_file.close()

# Load in beta and time_steps from the original epidemic.
params_file = open(path_to_original_epidemic + "params.pkl", "rb")
params_dic = pickle.load(params_file)
beta = params_dic["beta"]
time_steps = params_dic["time_steps"]
normalization = params_dic["normalization"]
gamma_params = params_dic["prior_params"]
params_file.close()
# Now, let's draw our samples.
samples = {}
times = []
curr_samples = 0

# Load in our compressor.
path_to_compressor = "models/compressor.pt"
compressor = th.load(path_to_compressor)
compressor.eval()
orig_features = compressor(th.Tensor(true_nn_input))
num_features = orig_features.size(dim=0)
features = np.zeros((num_samples,num_features))
distances = []
while curr_samples < num_samples:
	# Need to then sample some epidemics.. Basically return a sample from
	# the function in the paper, with thetas drawn from a U(0,1)
	# Which network we use (true vs. observed) depends on the "missing" flag
	sample = epidemic_utils.simulate_SI_gillespie(true_network,beta,i_list,time_steps)
	# This above operation basically makes it so samples has keys 'theta' and 'output', in a list of lists.
	sample_nn_input = epidemic_utils.normalize(sample["i_times"])
	sample_features = compressor(th.Tensor(sample_nn_input))
	features[curr_samples,:] = sample_features.detach().numpy()
	#print(str(sample_features[0]) + " " + str(sample_features[39]))
	dist = (sample_features - orig_features).pow(2).sum(axis=0).sqrt().detach().numpy() 
	distances.append(float(dist)) 
	curr_samples += 1

fig, ax = plt.subplots()
for i in range(num_features):
	sns.kdeplot(features[:,i], alpha = 0.1)
plt.title("Distribution of summary statistic features")
plt.savefig("diagnostics/feature_distributions.png")

fig,ax = plt.subplots()
plt.hist(x = distances, bins = 100)
plt.savefig("diagnostics/true_distance.png")
fig,ax = plt.subplots()
sns.kdeplot(distances)
plt.savefig("diagnostics/true_distance_density.png")


distances_2 = []
beta_2 = 0.5
curr_samples = 0
while curr_samples < num_samples:
	sample = epidemic_utils.simulate_SI_gillespie(true_network,beta_2,i_list,time_steps)
	curr_samples += 1
	sample_nn_input = epidemic_utils.normalize(sample["i_times"])
	sample_features = compressor(th.Tensor(sample_nn_input))
	dist = (sample_features - orig_features).pow(2).sum(axis=0).sqrt().detach().numpy() 
	distances_2.append(float(dist)) 

distances_3 = []
beta_3 = 0.05
curr_samples = 0
while curr_samples < num_samples:
	sample = epidemic_utils.simulate_SI_gillespie(true_network,beta_3,i_list,time_steps)
	curr_samples += 1
	sample_nn_input = epidemic_utils.normalize(sample["i_times"])
	sample_features = compressor(th.Tensor(sample_nn_input))
	dist = (sample_features - orig_features).pow(2).sum(axis=0).sqrt().detach().numpy() 
	distances_3.append(float(dist))

distances_4 = []
beta_4 = 0.10
curr_samples = 0
while curr_samples < num_samples:
	sample = epidemic_utils.simulate_SI_gillespie(true_network,beta_4,i_list,time_steps)
	curr_samples += 1
	sample_nn_input = epidemic_utils.normalize(sample["i_times"])
	sample_features = compressor(th.Tensor(sample_nn_input))
	dist = (sample_features - orig_features).pow(2).sum(axis=0).sqrt().detach().numpy() 
	distances_4.append(float(dist))

fig,ax = plt.subplots()
sns.kdeplot(distances, color = "blue", label = "beta = 0.08 (true)")
sns.kdeplot(distances_2, color = "green", label = "beta = 0.5")
sns.kdeplot(distances_3, color = "red", label = "beta = 0.05")
sns.kdeplot(distances_4, color = "gold", label = "beta = 0.10")
plt.xlabel("Distance")
plt.legend()
plt.savefig("diagnostics/distance_comparison.png")

distances = np.sort(distances)
distances_2 = np.sort(distances_2)
distances_3 = np.sort(distances_3)
distances_4 = np.sort(distances_4)
fig,ax = plt.subplots()
cdf_1 = np.arange(1,len(distances)+1)/float(len(distances))
cdf_2 = np.arange(1,len(distances_2)+1)/float(len(distances_2))
cdf_3 = np.arange(1,len(distances_3)+1)/float(len(distances_3))
cdf_4 = np.arange(1,len(distances_4)+1)/float(len(distances_4))
plt.step(distances, cdf_1, color = "blue", label = "beta = 0.08 (true)")
plt.step(distances_2, cdf_2, color = "green", label = "beta = 0.5")
plt.step(distances_3, cdf_3, color = "red", label = "beta = 0.05")
plt.step(distances_4, cdf_4, color = "gold", label = "beta = 0.10")
ax.set_xscale('log')
plt.xlabel("Distance")
plt.ylabel("Empirical CDF")
plt.legend()
plt.savefig("diagnostics/cdf_distance_comparison.png")

small_d = distances[:int(num_samples/10)]
small_d_2 = distances_2[:int(num_samples/10)]
small_d_3 = distances_3[:int(num_samples/10)]
small_d_4 = distances_4[:int(num_samples/10)]
fig,ax = plt.subplots()
sns.kdeplot(distances, color = "blue", label = "beta = 0.08 (true)")
sns.kdeplot(distances_2, color = "green", label = "beta = 0.5")
sns.kdeplot(distances_3, color = "red", label = "beta = 0.05")
sns.kdeplot(distances_4, color = "gold", label = "beta = 0.10")
plt.xlabel("Distance")
plt.title("Distance comparison, best 10 percent of runs")
plt.legend()
plt.savefig("diagnostics/distance_comparison_small.png")
