"""
Simple script for doing the MDN-compressed ABC, once the models have been trained
"""

import matplotlib.pyplot as plt
import pickle
import epidemic_utils
import torch as th
import numpy as np
import epidemic_utils
from tqdm import tqdm
import scipy.stats as st
import seaborn as sns
import os
import time
import argparse
import sys
import statistics
import scipy
import copy
import random
import pandas

# How much of our training set to accept as ABC samples.
percentile_accepted = float(sys.argv[1])
print("Accepting " + str(percentile_accepted) + " percent.")
"""
Load the compressor
"""
	
path_to_compressor = "models/compressor.pt"
compressor = th.load(path_to_compressor)
compressor.eval()

path_to_mdn = "models/mdn.pt"
mdn = th.load(path_to_mdn)
mdn.eval()
	
"""
Load our original epidemic information
"""
params_file = open("true_epidemic/params.pkl", "rb")
params_dic = pickle.load(params_file)
num_nodes = params_dic["num_nodes"]
true_beta = params_dic["beta"]
true_gamma = params_dic["gamma"]
time_steps = params_dic["time_steps"]
prior_params = params_dic["prior_params"]
params_file.close()
	
path_to_original_epidemic = "true_epidemic/"
	
# Load in the original "true" network
true_network_file = open(path_to_original_epidemic + "true_network.pkl", "rb")
true_network = pickle.load(true_network_file)
true_network_file.close()

# Load in original test times.
true_test_times_file = open(path_to_original_epidemic + "test_times.pkl", "rb")
true_test_times = pickle.load(true_test_times_file)
true_test_times_file.close()


results_file = open(path_to_original_epidemic + "true_epidemic.pkl", "rb")
true_results = pickle.load(results_file)
results_file.close()
	
# Load in the initial list
initial_file = open(path_to_original_epidemic + "true_initial_list.pkl", "rb")
i_list = pickle.load(initial_file)
initial_file.close()
	
# Lastly, let's get the original features, from our compressor.
orig_features = compressor(th.Tensor(true_results["results"]))

orig_mdn = mdn(th.Tensor(true_results["results"]))


"""
Reload our training samples (training samples can be used for our draw.)
"""
data_path = "data/"
train_data_path = data_path + "training_data.pbz2"
	
training_samples = epidemic_utils.decompress_pickle(train_data_path)
	  
	
num_training_samples = len(training_samples["output"])
print("Training samples reloaded.")

"""
Now, let's do basic rejection sampling for a while.
First, draw a bunch of samples, do the simulation, and record the euclidean distances of features from original run.
"""
	
sampled_betas = []
sampled_gammas = []
sampled_times = []
euclidean_distances = []

accepted_times = [] 
	
print("Drawing " + str(num_training_samples) + " samples from training set")
	
theta = training_samples["theta"]
# Get the first and second columns of theta.
sampled_betas.extend(theta[:,0])
sampled_gammas.extend(theta[:,1])
 
op_features = compressor(th.Tensor(training_samples["output"]))
dist = (op_features-orig_features).pow(2).sum(axis = 1).sqrt().detach().numpy()
euclidean_distances = list(dist)
		
# And then truncate down to the size we need
sampled_betas = sampled_betas[:int(num_training_samples)]
sampled_gammas = sampled_gammas[:int(num_training_samples)]
sampled_times = training_samples["times"][:int(num_training_samples)]
euclidean_distances = euclidean_distances[:int(num_training_samples)]
		
"""
Now, extract the best % of the runs and plot ABC results.
"""
if not os.path.exists("abc"):
	os.makedirs("abc")

# percentile_accepted came in as an argument.
accepted_thetas = []
accepted_times = [] # For looking at inference on true infection times.
# Grab the first percentile
percentile_value = np.percentile(np.array(euclidean_distances), float(percentile_accepted))
	
for i in range(int(num_training_samples)):
	if euclidean_distances[i] <= percentile_value:
		accepted_thetas.append([sampled_betas[i],sampled_gammas[i]])
		accepted_times.append(sampled_times[i])
accepted_thetas = np.array(accepted_thetas)
	
fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0.001,0.999)
ax.set_ylim(0.001,0.999)
sns.kdeplot(list(accepted_thetas[:,0]), list(accepted_thetas[:,1]),ax = ax)
plt.axvline(true_beta, color = "black", alpha = 0.6)
plt.axhline(true_gamma, color = "black", alpha = 0.6)
plt.scatter(list(accepted_thetas[:,0]), list(accepted_thetas[:,1]), color = "g", marker = ".", alpha = 0.1)
plt.title("Contours and Scatterplot of MDN-Compressed ABC,\n Spreading Rate and Recovery Rate")
plt.ylabel("Gamma")
plt.xlabel("Beta")
plt.savefig('abc/joint_inference_big.png')

fig, ax = plt.subplots()
ax.set_xlim(0.001,0.4)
ax.set_ylim(0.001,0.2)
sns.kdeplot(list(accepted_thetas[:,0]), list(accepted_thetas[:,1]),ax = ax)
plt.axvline(true_beta, color = "black", linestyle = "-.", alpha = 0.6)
plt.axhline(true_gamma, color = "black", linestyle = "-.", alpha = 0.6)
plt.scatter(list(accepted_thetas[:,0]), list(accepted_thetas[:,1]), color = "g", marker = ".", alpha = 0.1)
plt.title("Contours and Scatterplot of MDN-Compressed ABC,\n Spreading Rate and Recovery Rate")
plt.ylabel("Gamma")
plt.xlabel("Beta")
ax.axis("equal")
plt.savefig('abc/joint_inference_small.png')
plt.savefig('abc/joint_inference_small.pdf')

"""
Plots below
"""
fig, ax = plt.subplots()
ax.set_xlim(0.001,0.999)
beta_kde = sns.kdeplot(list(accepted_thetas[:,0]))
x,y = beta_kde.get_lines()[0].get_data()
beta_cdf = scipy.integrate.cumtrapz(y,x,initial = 0)
beta_median = x[np.abs(beta_cdf - 0.5).argmin()]
plt.vlines(beta_median,0,y[np.abs(beta_cdf-0.5).argmin()], color = "tab:cyan")
plt.axvline(true_beta, color = "r", alpha = 0.3, linestyle = "-.")
plt.title("MDN-Compressed ABC,\n Spreading coefficient beta")
plt.ylabel("Posterior Density")
plt.xlabel("Beta")
plt.savefig('abc/beta_big.png')

fig, ax = plt.subplots()
ax.set_xlim(0.001,0.4)
sns.kdeplot(list(accepted_thetas[:,0]))
plt.axvline(true_beta, color = "black", alpha = 0.6, linestyle = "-.")
plt.title("MDN-Compressed ABC,\n Spreading coefficient beta")
plt.ylabel("Posterior Density")
plt.xlabel("Beta")
plt.savefig('abc/beta_small.png')
plt.savefig('abc/beta_small.pdf')

fig, ax = plt.subplots()
ax.set_xlim(0.001,0.999)
gamma_kde = sns.kdeplot(list(accepted_thetas[:,1]))
x,y = gamma_kde.get_lines()[0].get_data()
gamma_cdf = scipy.integrate.cumtrapz(y,x,initial = 0)
gamma_median = x[np.abs(gamma_cdf - 0.5).argmin()]
plt.vlines(gamma_median,0,y[np.abs(gamma_cdf-0.5).argmin()], color = "tab:cyan")
plt.axvline(true_gamma, color = "r", alpha = 0.3, linestyle = "-.")
plt.title("MDN-Compressed ABC,\n Recovery coefficient gamma")
plt.ylabel("Posterior Density")
plt.xlabel("Gamma")
plt.savefig('abc/gamma_big.png')


fig, ax = plt.subplots()
ax.set_xlim(0.001,0.4)
sns.kdeplot(list(accepted_thetas[:,1]))
#plt.vlines(gamma_median,0,y[np.abs(gamma_cdf-0.5).argmin()], color = "tab:cyan")
plt.axvline(true_gamma, color = "black", alpha = 0.6, linestyle = "-.")
plt.title("MDN-Compressed ABC,\n Recovery coefficient gamma")
plt.ylabel("Posterior Density")
plt.xlabel("Gamma")
plt.savefig('abc/gamma_small.png')
plt.savefig('abc/gamma_small.pdf')


#Dump the draws.

path_to_output = r"abc/" # Where we'll store our data.
if not os.path.exists(path_to_output):
	os.makedirs(path_to_output)
dump_content = {"thetas": accepted_thetas,
				"distances": euclidean_distances}
output_file = open(path_to_output + "abc_draws.pkl", "wb")
pickle.dump(dump_content, output_file)
output_file.close()

#Also, dump the density estimates for later use.
beta_kde_file = open(path_to_output + "beta_density.pkl", "wb")
gamma_kde_file = open(path_to_output + "gamma_density.pkl", "wb")
pickle.dump(beta_kde, beta_kde_file)
pickle.dump(gamma_kde, gamma_kde_file)
beta_kde_file.close()
gamma_kde_file.close()
	


# Print some results:
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


print("Estimated density median for beta is " + str(beta_median) + ", with error of " + str((beta_median - true_beta)/true_beta))
beta_median_calculated = statistics.median(accepted_thetas[:,0])
print("Actual median for accepted betas is " + str(beta_median_calculated) + ", with error of " + str((beta_median_calculated - true_beta)/true_beta))

print("Estimated density median for gamma is " + str(gamma_median) + ", with error of " + str((gamma_median - true_gamma)/true_gamma))
gamma_median_calculated = statistics.median(accepted_thetas[:,1])
print("Actual median for accepted gammas is " + str(gamma_median_calculated) + ", with error of " + str((gamma_median_calculated - true_gamma)/true_gamma))


# Have let's round each value in accepted_times.
accepted_times = np.array(accepted_times)
rounded_accepted_times = np.zeros((accepted_times.shape[0], accepted_times.shape[1]))
for i in range(accepted_times.shape[0]):
	for j in range(accepted_times.shape[1]):
		if accepted_times[i,j] <= time_steps:
			rounded_accepted_times[i,j] = int(np.round(accepted_times[i,j]))
		else:
			print("An event went past observation time; truncating time from " + str(accepted_times[i,j]))
			rounded_accepted_times[i,j] = int(time_steps)


# Finally, do a heatmap of the infection times.
# Accepted times are in accepted_times.
true_i_times = true_results["i_times"]
true_r_times = true_results["r_times"]
for i in range(len(true_i_times)):
	if true_i_times[i] > time_steps:
		print("Had to truncate infection time " + str(true_i_times[i]))
		true_i_times[i] = time_steps
for i in range(len(true_r_times)):
	if true_r_times[i] > time_steps:
		print("Had to truncate recovery time " + str(true_r_times[i]))
		true_r_times[i] = time_steps
nodes_to_show = min(num_nodes, 55)
print(rounded_accepted_times.shape)
print(rounded_accepted_times)
print("Max time:")
print(np.max(rounded_accepted_times))
accepted_times_flip = np.swapaxes(accepted_times, 0, 1)
rounded_accepted_times_flip = np.swapaxes(rounded_accepted_times,0,1)
takes = np.zeros((nodes_to_show, time_steps + 1))
fig, ax = plt.subplots()
for node in range(nodes_to_show):
	total_per_timestep = [0] * (time_steps+1)
	for iteration in rounded_accepted_times_flip[node,:]:
		total_per_timestep[int(iteration)] += 1
	for time_step in range(time_steps+1):
		takes[node,time_step] = total_per_timestep[time_step]
plt.imshow(takes, cmap = "hot", vmax = int(0.2*len(accepted_times)))
# Plot the true times of infection.
plt.scatter(true_i_times[:nodes_to_show], list(range(nodes_to_show)), facecolors = "none", edgecolors = "deeppink", s = 3)
plt.scatter(true_r_times[:nodes_to_show], list(range(nodes_to_show)), facecolors = "none", edgecolors = "limegreen", s= 3)
# Plot the true test times.
n = []
tt = []
for node in true_test_times:
	if node < nodes_to_show:
		for time in true_test_times[node]:
			tt.append(time)
			n.append(node)
plt.scatter(tt, n, c = "lightblue", s = 1, marker = "v")		
plt.title("Heatmap of accepted times")
plt.xlabel("Time of infection")
plt.ylabel("Node")
plt.savefig('abc/time_heatmap.png', dpi = 1200)

# Plot with ranked times (ordering by first infected to last infected)
fig,ax = plt.subplots()
order = np.array(true_i_times[:nodes_to_show]).argsort() # order tells us which nodes go first.
ordered_takes = np.zeros((nodes_to_show, time_steps+1))
ordered_r_times = [0]*nodes_to_show
for i in range(nodes_to_show):
	ordered_takes[i,:] = takes[order[i],:] # Each row of ordered takes draws from takes.
	ordered_r_times[i] = true_r_times[order[i]] # Each element of the new r_times draws from true_r_times
implot = plt.imshow(ordered_takes,cmap = "hot", vmax = int(0.2*len(accepted_times)))
tit_copy_s = copy.deepcopy(true_i_times[:nodes_to_show])
tit_copy_s.sort()
plt.scatter(ordered_r_times[:nodes_to_show], list(range(nodes_to_show)), facecolors = "none", edgecolors = "limegreen", s= 3)
plt.scatter(tit_copy_s[:nodes_to_show], list(range(nodes_to_show)), facecolors = "none", edgecolors = "deeppink", s= 3)
plt.title("Heatmap of accepted times, ranked by true time of infection")
plt.xlabel("Time of infection")
plt.ylabel("Rank")
plt.savefig('abc/time_heatmap_ranked.png', dpi = 1200)

# We can calculate the coverages (for posterior predictive) too.
# Calculate 95% interval coverage.
coverages = []
percentiles = [50, 60, 70, 75, 80, 85, 90, 95]
for percentile in percentiles:
	upper_percentile = 100-((100-percentile)/2)
	lower_percentile = (100-percentile)/2
	cover_times = 0
	for node in range(num_nodes):
		node_times = accepted_times_flip[node,:]
		upper = np.percentile(node_times, upper_percentile)
		lower = np.percentile(node_times, lower_percentile)
		if true_i_times[node] >= lower and true_i_times[node] <= upper:
			cover_times += 1
	coverages.append(cover_times*100/num_nodes)
fig,ax = plt.subplots()
plt.scatter(percentiles, coverages, color = "royalblue")
plt.plot(percentiles, percentiles, alpha = 0.3, color = "limegreen")
plt.title("Nominal coverage of various confidence intervals, \nposterior predictives for infection times")
plt.xlabel("Nominal coverage")
plt.ylabel("Empirical coverage")
plt.savefig('abc/coverages.png')



# Lastly, generate some extra "original epidemics", and see how they vary.
# And make their violin plots.
check_iterations = 10
orig_check = []
pos_results = []
beta_violin_data = np.array([])
gamma_violin_data = np.array([])

for w in range(check_iterations):
	iter_output = epidemic_utils.simulate_SIR_gillespie(true_network, true_beta, true_gamma, i_list, true_test_times, time_steps)

	iter_features = compressor(th.Tensor(iter_output["results"]))

	iter_dist = (op_features-iter_features).pow(2).sum(axis = 1).sqrt().detach().numpy()
	iter_distances = list(iter_dist)
		
	# percentile_accepted came in as an argument.
	iter_accepted_thetas = []
	iter_accepted_times = [] # For looking at inference on true infection times.
	# Grab the first percentile
	iter_percentile_value = np.percentile(np.array(iter_distances), float(percentile_accepted))
	 
	for i in range(int(num_training_samples)):
		if iter_distances[i] <= iter_percentile_value:
			iter_accepted_thetas.append([sampled_betas[i],sampled_gammas[i]])
			iter_accepted_times.append(sampled_times[i])
	iter_accepted_thetas = np.array(iter_accepted_thetas)
	orig_check.append(iter_accepted_thetas)
	pos_results.append(iter_output["positive_results"]) # Track how many positive results.

	# For the violin plots
	iter_accepted_betas = np.reshape(np.array(iter_accepted_thetas[:,0]), (len(iter_accepted_thetas[:,0]),1))
	iter_accepted_gammas = np.reshape(np.array(iter_accepted_thetas[:,1]), (len(iter_accepted_thetas[:,1]),1))
	run = np.reshape([w+1] * len(iter_accepted_betas), (len(iter_accepted_betas),1)) # Get a column vector of runs.
	curr_data_beta = np.concatenate((iter_accepted_betas,run),axis=1)
	curr_data_gamma = np.concatenate((iter_accepted_gammas,run),axis=1)
	if beta_violin_data.shape[0] == 0:
		beta_violin_data = curr_data_beta
		gamma_violin_data = curr_data_gamma
	elif w < 15:
		beta_violin_data = np.concatenate((beta_violin_data, curr_data_beta))
		gamma_violin_data = np.concatenate((gamma_violin_data, curr_data_gamma))

fig,ax = plt.subplots()
ax.set_xlim(0.001,true_beta*3)
beta_kde = sns.kdeplot(list(accepted_thetas[:,0]), color = "dodgerblue")
for i in range(check_iterations):
	sns.kdeplot(list(orig_check[i][:,0]), color = "grey", alpha = 0.2)
plt.axvline(true_beta, color = "r", alpha = 0.3, linestyle = "-.")
plt.title("MDN-Compressed ABC,\n Spreading coefficient beta")
plt.ylabel("Posterior Density")
plt.xlabel("Beta")
plt.savefig('abc/orig_comparison_beta.png')

fig,ax = plt.subplots()
ax.set_xlim(0.001,true_gamma*3)
beta_kde = sns.kdeplot(list(accepted_thetas[:,1]), color = "dodgerblue")
for i in range(check_iterations):
	sns.kdeplot(list(orig_check[i][:,1]), color = "grey", alpha = 0.2)
plt.axvline(true_gamma, color = "r", alpha = 0.3, linestyle = "-.")
plt.title("MDN-Compressed ABC,\n Recovery coefficient gamma")
plt.ylabel("Posterior Density")
plt.xlabel("Gamma")
plt.savefig('abc/orig_comparison_gamma.png')


# Now also plot the same figures, but using color coding.
fig,ax = plt.subplots()
ax.set_xlim(0.001,true_beta*3)
beta_kde = sns.kdeplot(list(accepted_thetas[:,0]), color = "dodgerblue")
for i in range(check_iterations):
	color_choice = "grey"
	if pos_results[i] >= np.percentile(pos_results, 80):
		color_choice = "maroon"
	elif pos_results[i] >= np.percentile(pos_results, 60):
		color_choice = "red"
	elif pos_results[i] >= np.percentile(pos_results,40):
		color_choice = "orangered"
	elif pos_results[i] >= np.percentile(pos_results, 20):
		color_choice = "orange"
	else:
		color_choice = "gold"
	sns.kdeplot(list(orig_check[i][:,0]), color = color_choice, alpha = 0.2)
plt.axvline(true_beta, color = "r", alpha = 0.3, linestyle = "-.")
plt.title("MDN-Compressed ABC,\n Spreading coefficient beta")
plt.ylabel("Posterior Density")
plt.xlabel("Beta")
plt.savefig('abc/orig_comparison_beta_colored.png')


fig,ax = plt.subplots()
ax.set_xlim(0.001,true_gamma*3)
beta_kde = sns.kdeplot(list(accepted_thetas[:,1]), color = "dodgerblue")
for i in range(check_iterations):
	color_choice = "grey"
	if pos_results[i] >= np.percentile(pos_results, 80):
		color_choice = "maroon"
	elif pos_results[i] >= np.percentile(pos_results, 60):
		color_choice = "red"
	elif pos_results[i] >= np.percentile(pos_results,40):
		color_choice = "orangered"
	elif pos_results[i] >= np.percentile(pos_results, 20):
		color_choice = "orange"
	else:
		color_choice = "gold"
	sns.kdeplot(list(orig_check[i][:,1]), color = color_choice, alpha = 0.2)
plt.axvline(true_gamma, color = "r", alpha = 0.3, linestyle = "-.")
plt.title("MDN-Compressed ABC,\n Recovery coefficient gamma")
plt.ylabel("Posterior Density")
plt.xlabel("Gamma")
plt.savefig('abc/orig_comparison_gamma_colored.png')


# Plot violin plots.
beta_violin_df = pandas.DataFrame(data = beta_violin_data, columns = ["Beta", "Instance"])
beta_violin_df = beta_violin_df.astype({"Beta": float, "Instance": int})
gamma_violin_df = pandas.DataFrame(data = gamma_violin_data, columns = ["Gamma", "Instance"])
gamma_violin_df = gamma_violin_df.astype({"Gamma": float, "Instance": int})
fig,ax = plt.subplots()
sns.violinplot(data = beta_violin_df, x = "Instance", y = "Beta")
plt.axhline(true_beta, color = "black", linestyle = "-.")
plt.savefig("abc/violin_plots_beta.pdf") 
fig,ax = plt.subplots()
sns.violinplot(data = gamma_violin_df, x = "Instance", y = "Gamma")
plt.axhline(true_gamma, color = "black", linestyle = "-.")
plt.savefig("abc/violin_plots_gamma.pdf") 

v_file_beta = open("abc/violin_data_beta.pkl","wb")
pickle.dump(beta_violin_df, v_file_beta)
v_file_beta.close()
v_file_gamma = open("abc/violin_data_gamma.pkl", "wb")
pickle.dump(gamma_violin_df, v_file_gamma)
v_file_gamma.close()

# Plot 95 percentile whisker plots.
x_coords = np.array(list(range(check_iterations))) + 1
fig,ax = plt.subplots()
max_val = 0
means = []
for i in range(check_iterations):
	x_loc = i+1
	lower = np.percentile(orig_check[i][:,0],2.5)
	mean = np.mean(orig_check[i][:,0])
	means.append(mean)
	upper = np.percentile(orig_check[i][:,0],97.5)
	if upper > max_val:
		max_val = upper
	plt.vlines(x = x_loc, ymin = lower, ymax = upper)
	plt.hlines(y = lower , xmin = x_loc-0.1, xmax = x_loc+0.1)
	plt.hlines(y = upper, xmin = x_loc-0.1, xmax = x_loc + 0.1)
plt.axhline(y = true_beta, linestyle = "-.", color = "black")
plt.scatter(x_coords, means) 
plt.xlabel("Instance")
plt.ylabel("Beta")
ax.set_xlim(0,check_iterations + 1)
ax.set_ylim(0, 1.1*max_val)
plt.savefig("abc/whisker_beta.png")

fig,ax = plt.subplots()
max_val = 0
means = []
for i in range(check_iterations):
	x_loc = i+1
	lower = np.percentile(orig_check[i][:,1],2.5)
	mean = np.mean(orig_check[i][:,1])
	means.append(mean)
	upper = np.percentile(orig_check[i][:,1],97.5)
	if upper > max_val:
		max_val = upper
	plt.vlines(x = x_loc, ymin = lower, ymax = upper)
	plt.hlines(y = lower , xmin = x_loc-0.05, xmax = x_loc+0.05)
	plt.hlines(y = upper, xmin = x_loc-0.05, xmax = x_loc + 0.05)
plt.axhline(y = true_gamma, linestyle = "-.", color = "black")
plt.scatter(x_coords, means) 
plt.xlabel("Instance")
plt.ylabel("Gamma")
ax.set_xlim(0,check_iterations + 1)
ax.set_ylim(0, 1.1*max_val)
plt.savefig("abc/whisker_gamma.png")
# This commented out area is for drawing pooled draws.

orig_check_file = open("abc/orig_check.pkl", "wb")
pickle.dump(orig_check, orig_check_file)
orig_check_file.close()


# Get posterior predictive trajectories.

fig,ax = plt.subplots()
ax.set_xlim(0.001,time_steps)
true_traj = epidemic_utils.get_trajectories(true_results["i_times"], true_results["r_times"], time_steps)
ax.plot(true_traj["times"], true_traj["infected"], color = "dodgerblue", label = "Original Epidemic")
traj_data = {"true_times": true_traj["times"], "true_infected": true_traj["infected"], "times": [], "infected": []}
repetitions = 150
avg_traj = []
for r in range(repetitions):
	num_abc = accepted_thetas.shape[0]
	param_draw = accepted_thetas[random.randint(0,num_abc-1),:]
	samp = epidemic_utils.simulate_SIR_gillespie(true_network, param_draw[0], param_draw[1], i_list, true_test_times, time_steps)
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


