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

sample_size = int(sys.argv[1])
percentile_accepted = float(sys.argv[2])


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
params_file.close()

# Load in the initial list and test times
initial_file = open(path_to_original_epidemic + "true_initial_list.pkl", "rb")
i_list = pickle.load(initial_file)
initial_file.close()

test_times_file = open(path_to_original_epidemic + "test_times.pkl", "rb")
test_times = pickle.load(test_times_file)
test_times_file.close()

results_file = open(path_to_original_epidemic + "true_epidemic.pkl", "rb")
true_results = pickle.load(results_file)
results_file.close()

# Load in the accepted densities and recalculate the density.
path_to_abc = r"abc/"
accepted_file = open(path_to_abc + "abc_draws.pkl", "rb")
accepted_draws = pickle.load(accepted_file)
accepted_thetas = accepted_draws["thetas"]
beta_kde = sns.kdeplot(list(accepted_thetas[:,0]))
gamma_kde = sns.kdeplot(list(accepted_thetas[:,1]))


# Calculate Median, and 5% and 95% values of the parameters.
x,y = beta_kde.get_lines()[0].get_data()
beta_cdf = scipy.integrate.cumtrapz(y,x,initial = 0)
beta_median = x[np.abs(beta_cdf - 0.5).argmin()]
beta_095 = x[np.abs(beta_cdf - 0.95).argmin()] # Upper
beta_05 = x[np.abs(beta_cdf - 0.05).argmin()] # Lower

x,y = gamma_kde.get_lines()[0].get_data()
gamma_cdf = scipy.integrate.cumtrapz(y,x,initial = 0)
gamma_median = x[np.abs(gamma_cdf - 0.5).argmin()]
gamma_095 = x[np.abs(gamma_cdf - 0.95).argmin()]
gamma_05 = x[np.abs(gamma_cdf - 0.05).argmin()]


# Load in the models
path_to_compressor = "models/compressor.pt"
compressor = th.load(path_to_compressor)
compressor.eval()

path_to_mdn = "models/mdn.pt"
mdn = th.load(path_to_mdn)
mdn.eval()


# Load in the ABC data
print("Drawining all samples from training set")
data_path = "data/"
train_data_path = data_path + "training_data.pbz2"
training_samples = epidemic_utils.decompress_pickle(train_data_path)
num_training_samples = len(training_samples["output"])
theta = training_samples["theta"]
training_betas = theta[:,0]
training_gammas = theta[:,1]
training_features = compressor(th.Tensor(training_samples["output"]))

whisker_metrics = []
whisker_num = 25
metrics = ["tot_i", "tot_r", "positive_results", "negative_results"]
metric_dict = {key: [] for key in metrics}
for i in range(whisker_num):
	print("Iteration: " + str(i) + " out of " + str(whisker_num))
	samp = epidemic_utils.simulate_SIR_gillespie(true_network, true_beta, true_gamma, i_list, test_times, time_steps)
	trial_features = compressor(th.Tensor(samp["results"]))
	dist = (training_features - trial_features).pow(2).sum(axis=1).sqrt().detach().numpy()
	euclidean_distances = list(dist)

	accepted_thetas = []
	percentile_value = np.percentile(np.array(euclidean_distances), float(percentile_accepted))
	for i in range(num_training_samples):
		if euclidean_distances[i] <= percentile_value:
			accepted_thetas.append([training_betas[i], training_gammas[i]])
	accepted_thetas = np.array(accepted_thetas)
	

	accepted_betas = accepted_thetas[:,0]
	accepted_gammas = accepted_thetas[:,1]

	sample_beta_025 = np.percentile(accepted_betas, 2.5)
	sample_beta_mean = np.mean(accepted_betas)
	sample_beta_975 = np.percentile(accepted_betas,97.5)
	sample_gamma_025 = np.percentile(accepted_gammas,2.5)
	sample_gamma_mean = np.mean(accepted_gammas)
	sample_gamma_975 = np.percentile(accepted_gammas,97.5)
	whisker_metrics.append([sample_beta_025, sample_beta_mean, sample_beta_975, sample_gamma_025, sample_gamma_mean, sample_gamma_975])

	# Now, use the accepted values to simulate more and look at summary statistics.
	all_samps = []
	for j in range(sample_size):
		num_abc = accepted_thetas.shape[0]
		param_draw = accepted_thetas[random.randint(0,num_abc-1),:]
		j_samp = epidemic_utils.simulate_SIR_gillespie(true_network, param_draw[0], param_draw[1], i_list, test_times, time_steps)
		all_samps.append(copy.deepcopy(j_samp))
	for metric in metrics:
		metric_vals = []
		for sample in all_samps:
			metric_vals.append(sample[metric])
		metric_dict[metric].append([np.percentile(metric_vals,2.5), np.mean(metric_vals), np.percentile(metric_vals,97.5), samp[metric]])		

print(metric_dict)

x_coords = np.array(list(range(whisker_num))) + 1
# Now look at each metric and generate a plot.
for metric in metrics:
	m_array = np.array(metric_dict[metric])
	fig,ax = plt.subplots()
	ax.set_xlim(0, whisker_num + 1)
	ax.set_ylim(0, 1.1*max(m_array[:,2]))
	plt.scatter(x_coords, m_array[:,1])
	plt.scatter(x_coords, m_array[:,3], marker = "x", color = "green")
			
	for i in range(whisker_num):
		x_loc = i+1
		plt.vlines(x = x_loc, ymin = m_array[i][0], ymax = m_array[i][2])
		plt.hlines(y = m_array[i][0], xmin = x_loc-0.05, xmax = x_loc+0.05)
		plt.hlines(y = m_array[i][2], xmin = x_loc-0.05, xmax = x_loc+0.05)
	plt.savefig("abc/whisker_metric_" + metric + ".png")

metric_dict_file = open("abc/whisker_metric_dict.pkl", "wb")
pickle.dump(metric_dict, metric_dict_file)
metric_dict_file.close()
print("Dumped metric_dict, with values for metrics of interest")


fig, ax = plt.subplots()
ax.set_xlim(0.001, time_steps)
true_traj = epidemic_utils.get_trajectories(true_results["i_times"], true_results["r_times"], time_steps)
ax.plot(true_traj["times"], true_traj["infected"], color = "dodgerblue", label = "Original Epidemic")
traj_data = {"true_times": true_traj["times"], "true_infected": true_traj["infected"], "times": [], "infected": []}
repetitions = 150
avg_traj = []
for r in range(repetitions):
	num_abc = accepted_thetas.shape[0]
	param_draw = accepted_thetas[random.randint(0,num_abc-1),:]
	samp = epidemic_utils.simulate_SIR_gillespie(true_network, param_draw[0], param_draw[1], i_list, test_times, time_steps)
	samp_traj = epidemic_utils.get_trajectories(samp["i_times"], samp["r_times"], time_steps)
	traj_data["times"].append(samp_traj["times"])
	traj_data["infected"].append(samp_traj["infected"])
	plt.plot(samp_traj["times"], samp_traj["infected"], color = "black", alpha = 0.05)
	if len(avg_traj) == 0:
		avg_traj = np.array(samp_traj["infected"])
	else:
		avg_traj = avg_traj + np.array(samp_traj["infected"])
avg_traj = avg_traj/repetitions
ax.plot(true_traj["times"], avg_traj, color = "crimson", label = "Posterior Predictive Average")
ax.legend()
plt.xlabel("Time")
plt.ylabel("Number infected")
plt.savefig("abc/infected_trajectories.png")
plt.savefig("abc/infected_trajectories.pdf")
traj_file = open("abc/traj_data.pkl", "wb")
pickle.dump(traj_data, traj_file)
traj_file.close()
print("traj_file dumped.")
