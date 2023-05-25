import numpy as np
import torch as th
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import epidemic_utils
import os
import argparse
import sys
import statistics
import scipy
import random
import copy
from scipy import stats

percentile_accepted = float(sys.argv[1])


path_to_original_epidemic = r"true_epidemic/"

# Load in the original "true" network
true_network_file = open(path_to_original_epidemic + "true_network.pkl", "rb")
true_network = pickle.load(true_network_file)
true_network_file.close()

# Load in beta, recovery coefficient, and time_steps from the original epidemic.
params_file = open(path_to_original_epidemic + "params.pkl", "rb")
params_dic = pickle.load(params_file)
true_beta = params_dic["beta"]
true_gamma = params_dic["gamma"]
time_steps = params_dic["time_steps"]
num_nodes = params_dic["num_nodes"]
prior_params = params_dic["prior_params"]
params_file.close()

# Load in the initial list and test times
initial_file = open(path_to_original_epidemic + "true_initial_list.pkl", "rb")
i_list = pickle.load(initial_file)
initial_file.close()

test_times_file = open(path_to_original_epidemic + "test_times.pkl", "rb")
test_times = pickle.load(test_times_file)
test_times_file.close()

# Load in the models
path_to_compressor = "models/compressor.pt"
compressor = th.load(path_to_compressor)
compressor.eval()

path_to_mdn = "models/mdn.pt"
mdn = th.load(path_to_mdn)
mdn.eval()


# Load in the ABC data
print("Drawing all samples from training set")
data_path = "data/"
train_data_path = data_path + "training_data.pbz2"
training_samples = epidemic_utils.decompress_pickle(train_data_path)
num_training_samples = len(training_samples["output"])
theta = training_samples["theta"]
training_betas = theta[:,0]
training_gammas = theta[:,1]
training_features = compressor(th.Tensor(training_samples["output"]))

percentiles_of_interest = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95]
beta_coverages = {}
gamma_coverages = {}
for percentile in percentiles_of_interest:
	beta_coverages[percentile] = 0
	gamma_coverages[percentile] = 0
coverage_iterations = 5000
beta_bounds = []
gamma_bounds = []
print("Calculating coverages drawn from true value only.")
results = {"beta": {"means": [], "medians": [], "ranks": [], "percentiles": []}, "gamma": {"means": [], "medians": [], "ranks": [], "percentiles": []}}
for i in range(coverage_iterations):
	print("Iteration: " + str(i) + " out of " + str(coverage_iterations))
	samp = epidemic_utils.simulate_SIR_gillespie(true_network, true_beta, true_gamma, i_list, test_times,time_steps)
	trial_features = compressor(th.Tensor(samp["results"]))
	dist = (training_features - trial_features).pow(2).sum(axis=1).sqrt().detach().numpy()
	euclidean_distances = list(dist)
	
	accepted_thetas = []
	percentile_value = np.percentile(np.array(euclidean_distances), float(percentile_accepted))
	for i in range(num_training_samples):
		if euclidean_distances[i] <= percentile_value:
			accepted_thetas.append([training_betas[i],training_gammas[i]])
	accepted_thetas = np.array(accepted_thetas)
	accepted_betas = accepted_thetas[:,0]
	accepted_gammas = accepted_thetas[:,1]
	results["beta"]["means"].append(np.mean(accepted_betas))
	results["beta"]["medians"].append(np.median(accepted_betas))
	results["gamma"]["means"].append(np.mean(accepted_gammas))
	results["gamma"]["medians"].append(np.median(accepted_gammas))

	for percentile in percentiles_of_interest:
		beta_lower_bound = np.percentile(accepted_betas, (100-percentile)/2)
		beta_upper_bound = np.percentile(accepted_betas, percentile + (100-percentile)/2)
		if true_beta < beta_upper_bound and true_beta > beta_lower_bound:
			beta_coverages[percentile] += 1/coverage_iterations
		beta_bounds.append([beta_lower_bound, beta_upper_bound])
		gamma_lower_bound = np.percentile(accepted_gammas, (100-percentile)/2)
		gamma_upper_bound = np.percentile(accepted_gammas, percentile + (100-percentile)/2)
		if true_gamma < gamma_upper_bound and true_gamma > gamma_lower_bound:
			gamma_coverages[percentile] += 1/coverage_iterations
		gamma_bounds.append([gamma_lower_bound, gamma_upper_bound])
	# Get ranks and percentiles.
	beta_rank = sum(accepted_betas < true_beta)
	results["beta"]["ranks"].append(beta_rank)
	gamma_rank = sum(accepted_gammas < true_gamma)
	results["gamma"]["ranks"].append(gamma_rank)
	beta_percentile = stats.percentileofscore(accepted_betas, true_beta)
	results["beta"]["percentiles"].append(beta_percentile)
	gamma_percentile = stats.percentileofscore(accepted_gammas, true_gamma)
	results["gamma"]["percentiles"].append(gamma_percentile)

fig,ax = plt.subplots()
plt.scatter(np.array(percentiles_of_interest)/100, list(beta_coverages.values()))
x = np.linspace(0,1,1000)
y = x
plt.plot(x,y)
plt.xlabel("Nominal Coverage")
plt.ylabel("Empirical Coverage")
plt.title("Nominal vs Empirical Coverage, Beta")
plt.savefig("abc/ci_coverage_beta.png")
print("95 percent coverage for beta for MDN-ABC is: " + str(beta_coverages[95]))

fig,ax = plt.subplots()
plt.scatter(np.array(percentiles_of_interest)/100, list(gamma_coverages.values()))
x = np.linspace(0,1,1000)
y = x
plt.plot(x,y)
plt.title("Nominal vs Empirical Coverage, Gamma")
plt.xlabel("Nominal Coverage")
plt.ylabel("Empirical Coverage")
plt.savefig("abc/ci_coverage_gamma.png")
print("95 percent coverage for gamma for MDN-ABC is: " + str(gamma_coverages[95]))



beta_coverage_file = open("abc/coverages_beta.pkl","wb")
pickle.dump(beta_coverages, beta_coverage_file)
beta_coverage_file.close()
gamma_coverage_file = open("abc/coverages_gamma.pkl","wb")
pickle.dump(gamma_coverages, gamma_coverage_file)
gamma_coverage_file.close()
bounds_file = open("abc/coverage_bounds.pkl","wb")
pickle.dump({"beta": beta_bounds, "gamma": gamma_bounds}, bounds_file)
bounds_file.close()
results_file = open("abc/coverage_mean_medians.pkl", "wb")
pickle.dump(results, results_file)
results_file.close()
