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
time_steps = params_dic["time_steps"]
num_nodes = params_dic["num_nodes"]
prior_params = params_dic["prior_params"]
params_file.close()

# Load in the initial list and test times
initial_file = open(path_to_original_epidemic + "true_initial_list.pkl", "rb")
i_list = pickle.load(initial_file)
initial_file.close()
# Load in the models
path_to_compressor = "models/compressor.pt"
compressor = th.load(path_to_compressor)
compressor.eval()

path_to_mdn = "models/mdn.pt"
mdn = th.load(path_to_mdn)
mdn.eval()


# Load in the ABC data
print("SI Epidemic, coverage calculation... Drawing all samples from training set")
data_path = "data/"
train_data_path = data_path + "training_data.pbz2"
training_samples = epidemic_utils.decompress_pickle(train_data_path)
num_training_samples = len(training_samples["output"])
theta = training_samples["theta"]
training_betas = theta[:,0]
training_features = compressor(th.Tensor(training_samples["output"]))

percentiles_of_interest = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95]
mdn_beta_coverages = {}
true_beta_coverages = {}
for percentile in percentiles_of_interest:
	mdn_beta_coverages[percentile] = 0
	true_beta_coverages[percentile] = 0
coverage_iterations = 5000
mdn_beta_bounds = []
true_beta_bounds = []
print("Calculating coverages drawn from true value only.")
results = {"mdn_abc": {"means": [], "medians": [], "ranks": [], "percentiles": []}, "analytical": {"means": [], "medians": [], "ranks": [], "percentiles": []}}
for i in range(coverage_iterations):
	print("Iteration: " + str(i) + " out of " + str(coverage_iterations))
	samp = epidemic_utils.simulate_SI_gillespie(true_network, true_beta, i_list, time_steps)
	trial_features = compressor(th.Tensor(samp["i_times"]))
	dist = (training_features - trial_features).pow(2).sum(axis=1).sqrt().detach().numpy()
	euclidean_distances = list(dist)
	
	accepted_thetas = []
	percentile_value = np.percentile(np.array(euclidean_distances), float(percentile_accepted))
	for i in range(num_training_samples):
		if euclidean_distances[i] <= percentile_value:
			accepted_thetas.append(training_betas[i])
	accepted_thetas = np.array(accepted_thetas)
	accepted_betas = accepted_thetas
	results["mdn_abc"]["means"].append(np.mean(accepted_betas))
	results["mdn_abc"]["medians"].append(np.median(accepted_betas))

	# Calculate the truth as well.
	samp_n_E = samp["tot_i"]
	event_times = samp["event_times"]
	samp_beta_hat_denom = 0
	for k in range(len(event_times)):
		if k == 0:
			samp_beta_hat_denom += samp["SI_connections"][0] * event_times[0]
		else:
			samp_beta_hat_denom += samp["SI_connections"][k] * (event_times[k] - event_times[k-1])
	samp_beta_mle = (samp_n_E - 1)/samp_beta_hat_denom
	samp_beta_a = prior_params[0] + samp_n_E - 1
	samp_beta_b = prior_params[1] + (samp_n_E - 1)/samp_beta_mle
	samp_true_draws = np.random.gamma(shape = samp_beta_a, scale = 1/samp_beta_b, size = 5000) 
	results["analytical"]["means"].append(np.mean(samp_true_draws))
	results["analytical"]["medians"].append(np.mean(samp_true_draws))
	
	# Get the percentiles.
	for percentile in percentiles_of_interest:
		mdn_beta_lower_bound = np.percentile(accepted_betas, (100-percentile)/2)
		mdn_beta_upper_bound = np.percentile(accepted_betas, percentile + (100-percentile)/2)
		if true_beta < mdn_beta_upper_bound and true_beta > mdn_beta_lower_bound:
			mdn_beta_coverages[percentile] += 1/coverage_iterations
		mdn_beta_bounds.append([mdn_beta_lower_bound, mdn_beta_upper_bound])
		true_beta_lower_bound = np.percentile(samp_true_draws, (100-percentile)/2)
		true_beta_upper_bound = np.percentile(samp_true_draws, percentile + (100-percentile)/2)
		if true_beta < true_beta_upper_bound and true_beta > true_beta_lower_bound:
			true_beta_coverages[percentile] += 1/coverage_iterations
		true_beta_bounds.append([true_beta_lower_bound, true_beta_upper_bound])
		
	# Get ranks and percentiles.
	mdn_beta_rank = sum(accepted_betas < true_beta)
	results["mdn_abc"]["ranks"].append(mdn_beta_rank)
	true_beta_rank = sum(samp_true_draws < true_beta)
	results["analytical"]["ranks"].append(true_beta_rank)
	mdn_beta_percentile = stats.percentileofscore(accepted_betas, true_beta)
	results["mdn_abc"]["percentiles"].append(mdn_beta_percentile)
	true_beta_percentile = stats.percentileofscore(samp_true_draws, true_beta)
	results["analytical"]["percentiles"].append(true_beta_percentile)

fig,ax = plt.subplots()
plt.scatter(np.array(percentiles_of_interest)/100, list(mdn_beta_coverages.values()), color = "blue", alpha = 0.6)
plt.scatter(np.array(percentiles_of_interest)/100, list(true_beta_coverages.values()), color = "orange", alpha = 0.3)
x = np.linspace(0,1,1000)
y = x
plt.plot(x,y)
plt.xlabel("Nominal Coverage")
plt.ylabel("Empirical Coverage")
plt.title("Nominal vs Empirical Coverage, Beta")
plt.savefig("abc/ci_coverage_beta.png")
print("95 percent coverage for beta for MDN-ABC is: " + str(mdn_beta_coverages[95]))




mdn_abc_beta_coverage_file = open("abc/coverages_mdn_abc.pkl","wb")
pickle.dump(mdn_beta_coverages, mdn_abc_beta_coverage_file)
mdn_abc_beta_coverage_file.close()
analytical_coverage_file = open("abc/coverages_analytical.pkl","wb")
pickle.dump(true_beta_coverages, analytical_coverage_file)
analytical_coverage_file.close()
bounds_file = open("abc/coverage_bounds.pkl","wb")
pickle.dump({"mdn_abc": mdn_beta_bounds, "analytical": true_beta_bounds}, bounds_file)
bounds_file.close()
results_file = open("abc/coverage_mean_medians.pkl", "wb")
pickle.dump(results, results_file)
results_file.close()
