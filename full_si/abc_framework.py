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
import pandas

save_pdf = True
percentile_accepted = int(sys.argv[1])
print("Accepting " + str(percentile_accepted * 100) + " percent.")
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
# Get the same parameters as the original epidemic (for beta and number of nodes).
# In the future, consider storing these with the initial run and loading them in.
params_file = open("true_epidemic/params.pkl", "rb")
params_dic = pickle.load(params_file)
num_nodes = params_dic["num_nodes"]
true_beta = params_dic["beta"]
time_steps = params_dic["time_steps"]
prior_params = params_dic["prior_params"]
params_file.close()
 
path_to_original_epidemic = "true_epidemic/"
    
# Load in the original "true" network
true_network_file = open(path_to_original_epidemic + "true_network.pkl", "rb")
true_network = pickle.load(true_network_file)
true_network_file.close()

"""
No missingness in this nm_sampled_sir case    
# Load in the observed network
observed_network_file = open(path_to_original_epidemic + "observed_network.pkl", "rb")
observed_network = pickle.load(observed_network_file)
observed_network_file.close()
"""

results_file = open(path_to_original_epidemic + "true_epidemic.pkl", "rb")
true_results = pickle.load(results_file)
results_file.close()
    
# Load in the initial list
initial_file = open(path_to_original_epidemic + "true_initial_list.pkl", "rb")
i_list = pickle.load(initial_file)
initial_file.close()

# Load in the combined inputs, for use in the compressor and MDN.
nn_input_file = open(path_to_original_epidemic + "true_nn_input.pkl", "rb")
true_nn_input = pickle.load(nn_input_file)
nn_input_file.close()

    
# Lastly, let's get the original features, from our compressor.
orig_features = compressor(th.Tensor(true_nn_input))

orig_mdn = mdn(th.Tensor(true_nn_input))
# Plot the MDN compressed approximations.
res_pt = 1000
thetas = np.linspace(0.001,1,num = res_pt)
probs = np.empty(res_pt)
mdn_marginal_beta = []
for i in range(len(thetas)):
    mdn_marginal_beta.append(float(np.e**orig_mdn.log_prob(th.Tensor([thetas[i]]))))
fig,ax = plt.subplots()
plt.plot(thetas,mdn_marginal_beta)
plt.axvline(true_beta, color = "b", alpha = 0.6)
plt.savefig("abc/mdn_beta.png")
"""
Reload our training samples (training samples can be used for our draw.)
"""
# Get the priors.
prior_beta_a = prior_params[0]
prior_beta_b = prior_params[1]


# Calculate the MLE
n_E = true_results["tot_i"]
event_times = true_results["event_times"]
# MLE for beta
beta_hat_denom = 0
for i in range(len(event_times)):
    if i == 0:
        beta_hat_denom += true_results["SI_connections"][0] * event_times[0]
    else:
        beta_hat_denom += true_results["SI_connections"][i]*(event_times[i] - event_times[i-1])
beta_mle = (n_E - 1)/beta_hat_denom

print("MLE for beta is: " + str(beta_mle) +", compared to true value of " + str(true_beta))

# Next, calculate posterior params.
beta_a = prior_params[0] + n_E - 1
beta_b = prior_params[1] + (n_E - 1)/beta_mle

data_path = "data/"
train_data_path = data_path + "training_data.pbz2"
    
training_samples = epidemic_utils.decompress_pickle(train_data_path)
    
num_training_samples = len(training_samples["output"])
print("Training samples reloaded.")
print("Length of input: " + str(num_training_samples)) 
"""
Now, let's do basic rejection sampling for a while.
First, draw a bunch of samples, do the simulation, and record the euclidean d  istances of features from original run.
"""
    
sampled_betas = []
euclidean_distances = []
    
print("Drawing " + str(num_training_samples) + " samples from training set")
    
theta = training_samples["theta"]
# Get the first and second columns of theta.
sampled_betas.extend(theta[:,0])
 
op_features = compressor(th.Tensor(training_samples["output"]))
dist = (op_features-orig_features).pow(2).sum(axis = 1).sqrt().detach().numpy()
euclidean_distances = list(dist)

 
"""
Now, extract the best % of the runs and plot ABC results.
"""
if not os.path.exists("abc"):
    os.makedirs("abc")

# percentile_accepted came in as an argument.
accepted_thetas = []
accepted_distances = []
# Grab the first percentile
percentile_value = np.percentile(np.array(euclidean_distances), float(percentile_accepted))
min_disc = np.min(np.array(euclidean_distances))
print("Threshold is: " + str(percentile_value))
print("Minimum discrepancy is: " + str(min_disc))
for i in range(int(num_training_samples)):
    if euclidean_distances[i] <= percentile_value:
        accepted_thetas.append([sampled_betas[i]])
        accepted_distances.append(euclidean_distances[i])
accepted_thetas = np.array(accepted_thetas)

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
# Plotting the golden standard and the prior.
x = np.linspace(0,1,2000)
ax.plot(x,scipy.stats.gamma.pdf(x,beta_a, scale = 1/beta_b), color = "gold", linestyle = "-.", alpha = 0.5)
ax.plot(x,scipy.stats.gamma.pdf(x,prior_beta_a, scale = 1/prior_beta_b), color = "orange", linestyle = "-.", alpha = 0.5)
plt.title("MDN-Compressed ABC and Golden Standard,\n Spreading coefficient beta")
plt.ylabel("Density")
plt.xlabel("Beta")
plt.savefig('abc/beta_big.png')
if save_pdf:
    plt.savefig('abc/beta_big.pdf')

    
fig, ax = plt.subplots()
ax.set_xlim(0.001,0.3)
sns.kdeplot(list(accepted_thetas[:,0]))
plt.vlines(beta_median,0,y[np.abs(beta_cdf-0.5).argmin()], color = "tab:cyan")
plt.axvline(true_beta, color = "r", alpha = 0.3, linestyle = "-.")
plt.title("MDN-Compressed ABC,\n Spreading coefficient beta")
plt.ylabel("Posterior Density")
plt.xlabel("Beta")
x = np.linspace(0,0.3,1000)
ax.plot(x,scipy.stats.gamma.pdf(x,beta_a, scale = 1/beta_b), color = "gold", linestyle = "-.", alpha = 0.5)
plt.savefig('abc/beta_small.png')
if save_pdf:
    plt.savefig('abc/beta_small.pdf')

fig,ax = plt.subplots()
ax.set_xlim(0.001, 0.999)
hist = plt.hist(list(accepted_thetas[:,0]), bins = 15, density = True, alpha = 0.1, label = "MDN-ABC")
plt.axvline(true_beta, color = "black",linestyle = "-.")
x = np.linspace(0,1,2000)
golden = ax.plot(x,scipy.stats.gamma.pdf(x,beta_a, scale = 1/beta_b), color = "gold", linestyle = "-.", label = "Golden Standard")
prior = ax.plot(x,scipy.stats.gamma.pdf(x,prior_beta_a, scale = 1/prior_beta_b), color = "orange", linestyle = "-.", alpha = 0.5, label = "Prior")
ax.legend()
plt.title("MDN-Compressed ABC and Golden Standard,\n Spreading coefficient beta")
plt.ylabel("Density")
plt.xlabel("Beta")
plt.savefig('abc/beta_hist.png')
if save_pdf:
    plt.savefig('abc/beta_hist.pdf')
#Dump the draws.

fig,ax = plt.subplots()
ax.set_xlim(0.001, 0.3)
hist = plt.hist(list(accepted_thetas[:,0]), bins = 15, density = True, alpha = 0.1, label = "MDN-ABC")
plt.axvline(true_beta, color = "black",linestyle = "-.")
x = np.linspace(0,0.3,2000)
golden = ax.plot(x,scipy.stats.gamma.pdf(x,beta_a, scale = 1/beta_b), color = "gold", linestyle = "-.", label = "Golden Standard")
prior = ax.plot(x,scipy.stats.gamma.pdf(x,prior_beta_a, scale = 1/prior_beta_b), color = "orange", linestyle = "-.", alpha = 0.5, label = "Prior")
ax.legend()
plt.title("MDN-Compressed ABC and Golden Standard,\n Spreading coefficient beta")
plt.ylabel("Density")
plt.xlabel("Beta")
plt.savefig('abc/beta_hist_small.png')
if save_pdf:
    plt.savefig('abc/beta_hist_small.pdf')
path_to_output = r"abc/" # Where we'll store our data.



if not os.path.exists(path_to_output):
    os.makedirs(path_to_output)
dump_content = {"thetas": accepted_thetas,
                "distances": accepted_distances}
output_file = open(path_to_output + "abc_draws.pkl", "wb")
pickle.dump(dump_content, output_file)
output_file.close()

    


# Print some results:
# Calculate Median, and 5% and 95% values of the parameters.
x,y = beta_kde.get_lines()[0].get_data()
beta_cdf = scipy.integrate.cumtrapz(y,x,initial = 0)
beta_median = x[np.abs(beta_cdf - 0.5).argmin()]
beta_095 = x[np.abs(beta_cdf - 0.95).argmin()] # Upper
beta_05 = x[np.abs(beta_cdf - 0.05).argmin()] # Lower


print("Estimated density median for beta is " + str(beta_median) + ", with error of " + str((beta_median - true_beta)/true_beta))
beta_median_calculated = statistics.median(accepted_thetas[:,0])
print("Actual median for accepted betas is " + str(beta_median_calculated) + ", with error of " + str((beta_median_calculated - true_beta)/true_beta))


# Looking at uncompressed
uncompressed_distances = np.sqrt(((np.array(true_nn_input) - np.array(training_samples["output"]))**2).sum(axis = 1))
accepted_thetas_uc = []
accepted_distances_uc = []
# Grab the first percentile
percentile_value_uc = np.percentile(np.array(uncompressed_distances), float(percentile_accepted))
min_disc_uc = np.min(np.array(uncompressed_distances))
print("Threshold is: " + str(percentile_value_uc))
print("Minimum discrepancy is: " + str(min_disc_uc))
for i in range(int(num_training_samples)):
    if uncompressed_distances[i] <= percentile_value_uc:
        accepted_thetas_uc.append([sampled_betas[i]])
        accepted_distances_uc.append(uncompressed_distances[i])
accepted_thetas_uc = np.array(accepted_thetas_uc)
uc_file = open(path_to_output + "uc_draws.pkl","wb")
pickle.dump(accepted_thetas_uc, uc_file)
uc_file.close()


fig, ax = plt.subplots()
ax.set_xlim(0.001,0.999)
sns.kdeplot(list(accepted_thetas[:,0]), color = "blue", alpha = 0.5, label = "MDN-ABC")
sns.kdeplot(list(accepted_thetas_uc[:,0]), color = "red", alpha = 0.5, label = "Non-compressed ABC")
true_val = plt.axvline(true_beta, color = "black", linestyle = "-.")
# Plotting the golden standard and the prior.
x = np.linspace(0,1,2000)
ax.plot(x,scipy.stats.gamma.pdf(x,beta_a, scale = 1/beta_b), color = "gold", linestyle = "-.", label = "Golden Standard")
ax.plot(x,scipy.stats.gamma.pdf(x,prior_beta_a, scale = 1/prior_beta_b), color = "green", linestyle = "-.", alpha = 0.5, label = "Prior")
ax.legend()
plt.title("MDN-Compressed ABC, non-Compressed ABC, and Golden Standard,\n Spreading coefficient beta")
plt.ylabel("Density")
plt.xlabel("Beta")
plt.savefig('abc/beta_uc.png')
if save_pdf:
    plt.savefig('abc/beta_uc.pdf')

fig, ax = plt.subplots()
ax.set_xlim(0.001,0.5)
sns.kdeplot(list(accepted_thetas[:,0]), color = "blue", alpha = 0.5, label = "MDN-ABC")
sns.kdeplot(list(accepted_thetas_uc[:,0]), color = "red", alpha = 0.5, label = "Non-compressed ABC")
true_val = plt.axvline(true_beta, color = "black", linestyle = "-.")
# Plotting the golden standard and the prior.
x = np.linspace(0,1,2000)
ax.plot(x,scipy.stats.gamma.pdf(x,beta_a, scale = 1/beta_b), color = "gold", linestyle = "-.", label = "Golden Standard")
ax.plot(x,scipy.stats.gamma.pdf(x,prior_beta_a, scale = 1/prior_beta_b), color = "green", linestyle = "-.", alpha = 0.5, label = "Prior")
ax.legend()
plt.title("MDN-Compressed ABC, non-Compressed ABC, and Golden Standard,\n Spreading coefficient beta")
plt.ylabel("Density")
plt.xlabel("Beta")
plt.savefig('abc/beta_uc_small.png')
if save_pdf:
    plt.savefig('abc/beta_uc_small.pdf')


fig,ax = plt.subplots(1,2)
fig.set_size_inches(12.8,4.8)
ax[0].set_xlim(0.001, 0.5)
hist = ax[0].hist(list(accepted_thetas[:,0]), bins = 15, density = True, alpha = 0.1, label = "MDN-ABC")
ax[0].axvline(true_beta, color = "black",linestyle = "-.")
x = np.linspace(0,0.5,2000)
golden = ax[0].plot(x,scipy.stats.gamma.pdf(x,beta_a, scale = 1/beta_b), color = "gold", linestyle = "-.", label = "Golden Standard")
prior = ax[0].plot(x,scipy.stats.gamma.pdf(x,prior_beta_a, scale = 1/prior_beta_b), color = "green", linestyle = "-.", alpha = 0.5, label = "Prior")
ax[0].legend()
ax[0].set_title("MDN-Compressed ABC and Golden Standard,\n Spreading coefficient beta")
ax[0].set(ylabel = "Density", xlabel = "Beta")
ax[1].set_xlim(0.001,0.5)
sns.kdeplot(list(accepted_thetas[:,0]), ax = ax[1], color = "blue", alpha = 0.5, label = "MDN-ABC")
sns.kdeplot(list(accepted_thetas_uc[:,0]), ax = ax[1], color = "red", alpha = 0.5, label = "Non-compressed ABC")
true_val = ax[1].axvline(true_beta, color = "black", linestyle = "-.")
# Plotting the golden standard and the prior.
x = np.linspace(0,1,2000)
ax[1].plot(x,scipy.stats.gamma.pdf(x,beta_a, scale = 1/beta_b), color = "gold", linestyle = "-.", label = "Golden Standard")
ax[1].plot(x,scipy.stats.gamma.pdf(x,prior_beta_a, scale = 1/prior_beta_b), color = "green", linestyle = "-.", alpha = 0.5, label = "Prior")
ax[1].legend()
ax[1].set_title("MDN-Compressed ABC, non-Compressed ABC, and Golden Standard,\n Spreading coefficient beta")
ax[1].set(ylabel = "Density", xlabel = "Beta")
plt.savefig('abc/hist_and_uc_combined.png')
if save_pdf:
    plt.savefig('abc/hist_and_uc_combined.pdf')


# And get the whisker plots.
whisker_metrics = {"mdn_beta": [], "true_beta": []}
whisker_num = 10
metrics = ["tot_i", "final_time"]
metric_dict = {key:[] for key in metrics}
violin_data = np.array([])
sample_size  = 10000
for w in range(whisker_num):
	print("Iteration: " + str(w) + " out of " + str(whisker_num))
	samp = epidemic_utils.simulate_SI_gillespie(true_network, true_beta, i_list, time_steps)
	trial_features = compressor(th.Tensor(samp["i_times"]))
	dist = (op_features - trial_features).pow(2).sum(axis=1).sqrt().detach().numpy()
	euclidean_distances = list(dist)
	
	trial_accepted_thetas = []
	percentile_value = np.percentile(np.array(euclidean_distances),float(percentile_accepted))
	for i in range(num_training_samples):
		if euclidean_distances[i] <= percentile_value:
			trial_accepted_thetas.append([sampled_betas[i]])
	trial_accepted_thetas = np.array(trial_accepted_thetas)
	trial_accepted_betas = trial_accepted_thetas[:,0]
	
	whisker_metrics["mdn_beta"].append([np.percentile(trial_accepted_betas,2.5), np.mean(trial_accepted_betas), np.percentile(trial_accepted_betas,97.5)])
	
	samp_n_E = samp["tot_i"]
	event_times = samp["event_times"]
	samp_beta_hat_denom = 0
	for i in range(len(event_times)):
		if i == 0:
			samp_beta_hat_denom += samp["SI_connections"][0] * event_times[0]
		else:
			samp_beta_hat_denom += samp["SI_connections"][i] * (event_times[i] - event_times[i-1])
	samp_beta_mle = (samp_n_E - 1)/samp_beta_hat_denom
	samp_beta_a = prior_params[0] + samp_n_E - 1
	samp_beta_b = prior_params[1] + (samp_n_E - 1)/samp_beta_mle
	samp_true_draws = np.random.gamma(shape = samp_beta_a, scale = 1/samp_beta_b, size = sample_size)
	whisker_metrics["true_beta"].append([np.percentile(samp_true_draws,2.5), np.mean(samp_true_draws), np.percentile(samp_true_draws,97.5)])
	
	combined_points = np.concatenate((np.array(trial_accepted_thetas), np.reshape(samp_true_draws,(sample_size, 1))))
	sources = np.reshape(["MDN-ABC"] * len(trial_accepted_betas) + ["Gold Standard"] * sample_size, (len(trial_accepted_betas) + sample_size, 1))
	run = np.reshape([w+1] * (sample_size + len(trial_accepted_betas)), (len(trial_accepted_betas) + sample_size,1))
	curr_data = np.concatenate((combined_points, sources, run), axis = 1)
	if violin_data.shape[0] == 0:
		violin_data = curr_data
	else:
		violin_data = np.concatenate((violin_data,curr_data))
# Generate whisker plots
for key in whisker_metrics.keys():
	whisker_metrics[key] = np.array(whisker_metrics[key])
x_coords = np.array(list(range(whisker_num))) + 1
fig,ax = plt.subplots()
ax.set_xlim(0,whisker_num + 1)
ax.set_ylim(0 , 1.1 * max(whisker_metrics["mdn_beta"][:,2]))
plt.scatter(x_coords ,whisker_metrics["mdn_beta"][:,1], label = "MDN-ABC")
plt.scatter(x_coords + 0.2, whisker_metrics["true_beta"][:,1], color = "gold", label = "Gold Standard")
for i in range(whisker_num):
	x_loc = i+1
	plt.vlines(x = x_loc, ymin = whisker_metrics["mdn_beta"][i][0], ymax = whisker_metrics["mdn_beta"][i][2])
	plt.hlines(y = whisker_metrics["mdn_beta"][i][0], xmin = x_loc -0.05, xmax = x_loc + 0.05)
	plt.hlines(y = whisker_metrics["mdn_beta"][i][2], xmin = x_loc - 0.05, xmax = x_loc + 0.05)	
	plt.vlines(x = x_loc + 0.2, ymin = whisker_metrics["true_beta"][i][0], ymax = whisker_metrics["true_beta"][i][2], color = "gold")
	plt.hlines(y = whisker_metrics["true_beta"][i][0], xmin = x_loc+0.2 -0.05, xmax = x_loc +0.2+ 0.05, color = "gold")
	plt.hlines(y = whisker_metrics["true_beta"][i][2], xmin = x_loc +0.2- 0.05, xmax = x_loc +0.2+ 0.05, color = "gold")
plt.axhline(y = true_beta, linestyle = "-.", color = "black")
plt.xlabel("Instance")
plt.ylabel("Beta")
ax.legend()
plt.title("95% Credible Intervals, Beta")
plt.savefig("abc/whisker_beta.png")
plt.savefig("abc/whisker_beta.pdf")

# Generate violin plots
violin_df = pandas.DataFrame(data = violin_data, columns = ["Beta", "Source", "Instance"])
violin_df = violin_df.astype({"Beta": float, "Source": str, "Instance": int})
fig,ax = plt.subplots()
sns.violinplot(data = violin_df, x = "Instance", y = "Beta", hue = "Source", split = True)
plt.axhline(true_beta, color = "black", linestyle = "-.")
plt.savefig("abc/violin_plots.png")
plt.savefig("abc/violin_plots.pdf")

# Do a combined plot of violin plot and posteriors.

prior_beta_a = prior_params[0]
prior_beta_b = prior_params[1]
n_E = true_results["tot_i"]
event_times = true_results["event_times"]
beta_hat_denom = 0
for i in range(len(event_times)):
	if i == 0:
		beta_hat_denom += true_results["SI_connections"][0] * event_times[0]
	else:
		beta_hat_denom += true_results["SI_connections"][i] * (event_times[i] - event_times[i-1])
beta_mle = (n_E - 1)/beta_hat_denom
beta_a = prior_params[0] + n_E - 1
beta_b = prior_params[1] + (n_E - 1)/beta_mle
fig,ax = plt.subplots(1,2)
fig.set_size_inches(12, 4.8)
ax[0].set_xlim(0.001,0.5)
sns.kdeplot(list(accepted_thetas[:,0]), ax = ax[0], color = "blue", alpha = 0.5, linestyle = "-.", label = "MDN-ABC")
sns.kdeplot(list(accepted_thetas_uc[:,0]), ax = ax[0], color = "orange", alpha = 0.5, linestyle = "-.", label = "Non-compressed ABC")
true_val = ax[0].axvline(true_beta, color = "black", linestyle = "-.")
# Plotting the golden standard and the prior.
x = np.linspace(0,1,2000)
ax[0].plot(x,scipy.stats.gamma.pdf(x,beta_a, scale = 1/beta_b), color = "red", linestyle = "-", alpha = 0.5, label = "Gold Standard")
ax[0].plot(x,scipy.stats.gamma.pdf(x,prior_beta_a, scale = 1/prior_beta_b), color = "green", alpha = 0.5, label = "Prior")
ax[0].legend()
ax[0].set(ylabel = "Density", xlabel = "Beta")
#ax[1].set_ylim(0.001,0.35)
sns.violinplot(data=violin_df, ax = ax[1], x = "Instance", y = "Beta", hue = "Source", split = True)
true_val = ax[1].axhline(true_beta, color = "black", linestyle = "-.")
plt.savefig('abc/violin_and_posteriors_combined.png')
plt.savefig('abc/violin_and_posteriors_combined.pdf')


print("Storing data.")
violin_data_file = open("abc/violin_data.pkl", "wb")
pickle.dump(violin_df, violin_data_file)
violin_data_file.close()

whisker_metrics_file = open("abc/whisker_metrics.pkl", "wb")
pickle.dump(whisker_metrics, whisker_metrics_file)
whisker_metrics_file.close()
