import numpy as np
import torch as th
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import epidemic_utils
import os
import argparse
import sys
import scipy.stats as st
import statistics
import scipy
import random
import copy
import pandas

save_pdf = True
calculate_coverages = True
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

true_run_file = open(path_to_original_epidemic + "true_epidemic.pkl", "rb")
true_run = pickle.load(true_run_file)
true_run_file.close()

nn_input_file = open(path_to_original_epidemic + "true_nn_input.pkl","rb")
true_nn_input = pickle.load(nn_input_file)
nn_input_file.close()


# Load in the accepted densities and recalculate the density.
path_to_abc = r"abc/"
accepted_file = open(path_to_abc + "abc_draws.pkl", "rb")
accepted_draws = pickle.load(accepted_file)
og_accepted_thetas = accepted_draws["thetas"]
uc_file = open(path_to_abc + "uc_draws.pkl","rb")
uc_accepted_thetas = pickle.load(uc_file)

accepted_file.close()
uc_file.close()


# Load in the models
path_to_compressor = "models/compressor.pt"
compressor = th.load(path_to_compressor)
compressor.eval()

path_to_mdn = "models/mdn.pt"
mdn = th.load(path_to_mdn)
mdn.eval()

orig_features = compressor(th.Tensor(true_nn_input))

# Load in the ABC data
print("Drawining all samples from training set")
data_path = "data/"
train_data_path = data_path + "training_data.pbz2"
training_samples = epidemic_utils.decompress_pickle(train_data_path)
num_training_samples = len(training_samples["output"])
theta = training_samples["theta"]
training_betas = theta[:,0]
training_features = compressor(th.Tensor(training_samples["output"]))

def event_times_to_trajectory(event_times, time_vector):
	# Given the event times, we now define a trajectory as the count of how many individuals have been infected by each timepoint.
	trajectory = []
	for t in time_vector:
		x = [i for i in event_times if i<t]
		trajectory.append(len(x))
	return trajectory

# First plot some posterior predictive trajectories.
traj_number = 25
trajectories = []
time_sample = np.linspace(0,50,1000)
for i in range(traj_number):
	num_abc = og_accepted_thetas.shape[0]
	param_draw = og_accepted_thetas[random.randint(0,num_abc-1),:]
	j_samp = epidemic_utils.simulate_SI_gillespie(true_network, param_draw[0], i_list, time_steps)
	trajectory = event_times_to_trajectory(j_samp["i_times"], time_sample)
	trajectories.append(trajectory)
true_trajectory = event_times_to_trajectory(true_run["i_times"], time_sample)
fig,ax = plt.subplots()
plt.plot(time_sample,true_trajectory, color = "blue")
for i in range(traj_number):
	plt.plot(time_sample,trajectories[i], color = "grey", alpha = 0.3)
plt.savefig("abc/posterior_trajectories.png")
print("Trajectories plotted")

whisker_metrics = []
true_whisker_metrics = []
whisker_num = 10
metrics = ["tot_i", "final_time"]
metric_dict = {key: [] for key in metrics}
violin_data = np.array([])
norm_params = []
for w in range(whisker_num):
	print("Iteration: " + str(w) + " out of " + str(whisker_num))
	samp = epidemic_utils.simulate_SI_gillespie(true_network, true_beta, i_list, time_steps)
	trial_features = compressor(th.Tensor(samp["i_times"]))
	dist = (orig_features - trial_features).pow(2).sum().sqrt().detach().numpy()
	euclidean_distances = list(dist)

	accepted_thetas = []
	percentile_value = np.percentile(np.array(euclidean_distances), float(percentile_accepted))
	for i in range(num_training_samples):
		if euclidean_distances[i] <= percentile_value:
			accepted_thetas.append([training_betas[i]])
	accepted_thetas = np.array(accepted_thetas)
	accepted_betas = accepted_thetas[:,0]
	norm_params.append([np.mean(accepted_betas), np.std(accepted_betas)])

	sample_beta_025 = np.percentile(accepted_betas, 2.5)
	sample_beta_mean = np.mean(accepted_betas)
	sample_beta_975 = np.percentile(accepted_betas,97.5)
	whisker_metrics.append([sample_beta_025, sample_beta_mean, sample_beta_975])

	# Next, find the true posterior parameters for this run.
	n_E = samp["tot_i"]
	event_times = samp["event_times"]
	beta_hat_denom = 0
	for i in range(len(event_times)):
		if i == 0:
			beta_hat_denom += samp["SI_connections"][0] * event_times[0]
		else:
			beta_hat_denom += samp["SI_connections"][i] * (event_times[i] - event_times[i-1])
	beta_mle = (n_E - 1)/beta_hat_denom
	beta_a = prior_params[0] + n_E - 1
	beta_b = prior_params[1] + (n_E-1)/beta_mle
	true_draws = np.random.gamma(shape = beta_a,scale = 1/beta_b,size = sample_size)
	true_beta_025 = np.percentile(true_draws,2.5)
	true_beta_mean = np.mean(true_draws)
	true_beta_975 = np.percentile(true_draws,97.5)
	true_whisker_metrics.append([true_beta_025, true_beta_mean, true_beta_975])

	combined_points = np.concatenate((np.array(accepted_thetas),np.reshape(true_draws,(sample_size,1)))) # All the points, from ABC and truth.
	# Next, make an array that keeps track of which source each came from.
	sources = np.reshape(["MDN-ABC"] * len(accepted_betas) + ["Gold Standard"] * sample_size, (len(accepted_betas) + sample_size,1))
	# Lastly, make an array that keeps track of which run this is.
	run = np.reshape([w+1] * (len(accepted_betas) + sample_size), (len(accepted_betas) + sample_size,1))
	curr_data = np.concatenate((combined_points,sources,run), axis = 1)
	if violin_data.shape[0] == 0:
		violin_data = curr_data
	else:	
		violin_data = np.concatenate((violin_data,curr_data))

if calculate_coverages:
	# Generating pooled posterior.
	all_accepted = []
	percentiles_of_interest = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
	coverages = {}
	true_draws_coverages = {}
	for percentile in percentiles_of_interest:
		coverages[percentile] = 0
		true_draws_coverages[percentile] = 0
	coverage_iterations = 5000
	for i in range(coverage_iterations):
		print("Iteration: " + str(i) + " out of " + str(coverage_iterations))
		samp = epidemic_utils.simulate_SI_gillespie(true_network, true_beta, i_list, time_steps)
		trial_features = compressor(th.Tensor(samp["i_times"]))
		dist = (trial_features-orig_features).pow(2).sum().sqrt().detach().numpy()
		euclidean_distances = list(dist)

		accepted_thetas = []
		percentile_value = np.percentile(np.array(euclidean_distances), float(percentile_accepted))
		for i in range(num_training_samples):
			if euclidean_distances[i] <= percentile_value:
				accepted_thetas.append([training_betas[i]])
		accepted_thetas = np.array(accepted_thetas)
		accepted_betas = accepted_thetas[:,0]
		for percentile in percentiles_of_interest:
			lower_bound = np.percentile(accepted_betas, (100 - percentile)/2)
			upper_bound = np.percentile(accepted_betas, percentile + (100 - percentile)/2)
			if true_beta < upper_bound and true_beta > lower_bound:
				coverages[percentile] += 1/coverage_iterations	
		n_E = samp["tot_i"]
		event_times = samp["event_times"]
		beta_hat_denom = 0
		for i in range(len(event_times)):
			if i == 0:
				beta_hat_denom += samp["SI_connections"][0] * event_times[0]
			else:
				beta_hat_denom += samp["SI_connections"][i] * (event_times[i] - event_times[i-1])
		beta_mle = (n_E - 1)/beta_hat_denom
		beta_a = prior_params[0] + n_E - 1
		beta_b = prior_params[1] + (n_E-1)/beta_mle
		true_draws = np.random.gamma(shape = beta_a,scale = 1/beta_b,size = len(accepted_betas))	
		for percentile in percentiles_of_interest:
			lower_bound = np.percentile(true_draws, (100 - percentile)/2)
			upper_bound = np.percentile(true_draws, percentile + (100 - percentile)/2)
			if true_beta < upper_bound and true_beta > lower_bound:
				true_draws_coverages[percentile] += 1/coverage_iterations	
	fig,ax = plt.subplots()
	plt.scatter(np.array(percentiles_of_interest)/100, list(coverages.values()))
	plt.scatter(np.array(percentiles_of_interest)/100, list(true_draws_coverages.values()), color = "gold")
	x = np.linspace(0,1,1000)
	y = x
	plt.plot(x,y)
	plt.title("Nominal vs Empirical coverage " + str(coverage_iterations) + " iterations")
	plt.savefig("abc/ci_coverage.png")
	print("95 percent coverage for MDN-ABC is: " + str(coverages[95]))
	print("95 percent coverage for true posterior is: " + str(true_draws_coverages[95]))
else:
	print("NOT calculating coverages")
metric_dict_file = open("abc/metric_dict.pkl", "wb")
pickle.dump(metric_dict, metric_dict_file)
metric_dict_file.close()

if calculate_coverages:
	coverage_file = open("abc/coverages.pkl", "wb")
	pickle.dump({"MDN-ABC": coverages, "truth": true_draws_coverages} ,coverage_file)
	coverage_file.close()
print("Finished with violin plots and whisker processing.")



