"""
Script for training the neural nets for vaccine inference
"""

import torch as th
import epidemic_utils
import pickle
import nn_utils
import os
import argparse
import numpy as np
import time
start_time = time.time()
# Read the number of nodes from our original epidemic parameter dictionary.
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
print("Using device: " + str(device))


params_file = open("true_epidemic/params.pkl", "rb")
params_dic = pickle.load(params_file)    
num_nodes = params_dic["num_nodes"]
params_file.close()


true_output_file = open("true_epidemic/true_epidemic.pkl", "rb")
true_output = pickle.load(true_output_file)
len_output = len(true_output["results"]) # When doing time inference, there's actually two elements to the output (the output and the times)	
	 
num_components = 2
batch_size = 2000
num_features = 15 
num_param = 2
compressor_layers = [len_output,300, 200, 100, 60, 40, 30, num_features]
mdn_layers = [num_features, 15, 15, num_components]
learning_rate = 10e-5
patience = 10
max_epochs = 500

print("Training Neural Net")
print("Batch size: " + str(batch_size))
print("Number of features: " + str(num_features))
print("Number of beta components for joint posterior: " + str(num_components))
print("Number of parameters of inference: " + str(num_param))
print("Compressor layers: " + str(compressor_layers))
print("MDN layers: " + str(mdn_layers))
print("Learning rate: " + str(learning_rate))
print("Patience: " + str(patience))    

"""
Loading Data
"""
    
    
# Define paths to our training, test, and validation sets.
data_path = "data/"
train_data_path = data_path + "training_data.pbz2"
val_data_path = data_path + "validation_data.pbz2"

    
paths = {"train": train_data_path, "validation": val_data_path}
datasets = {}
    
    
for key, path in paths.items():
    # M: Read in the pickled data from the sample.
    samples = epidemic_utils.decompress_pickle(path) 
    data = th.as_tensor(samples["output"]).float()
    params = th.as_tensor(samples['theta']).float()
    data.to(device)
    params.to(device)
    dataset = th.utils.data.TensorDataset(data, params)
    datasets[key] = dataset
print("Data loaded")
    
    
    
    
# Put it into a dataloader.
data_loaders = {key: th.utils.data.DataLoader(dataset, batch_size, shuffle=True)
                for key, dataset in datasets.items()}
    
"""
Defining loss function
"""
    
def evaluate_negative_log_likelihood(theta: th.Tensor, dist: th.distributions.Distribution) \
    -> th.Tensor:
    """
        Evaluate the negative log likelihood of a minibatch to minimize the expected posterior entropy
        directly (just taking mean for the Monte Carlo Estimate)
    """
    loss = - dist.log_prob(theta)
    return loss.mean()
        
loss_function = evaluate_negative_log_likelihood
    
"""
Defining Neural Nets
"""
# Number of features and number of components are passed in through the argument.
    
print("Defining neural networks")
    
# The relevant neural nets are defined in nn_utils.
# But since our dimension is now high, let's get ourselves 50 nodes in the two hidden layers for now.
compressor = nn_utils.DenseStack(compressor_layers, th.nn.Tanh())
module = nn_utils.MixtureDensityNetwork_multi_param(compressor, mdn_layers, num_param, th.nn.Tanh())
    
"""
Optimization and Training
"""
    
# Define an optimizer and a scheduler.
optimizer = th.optim.Adam(module.parameters(), learning_rate)
scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience // 2)
    
print("Beginning training")
    
# Run the training.
epoch = 0
best_validation_loss = float('inf')
num_bad_epochs = 0
while num_bad_epochs < patience and (max_epochs is None or epoch < max_epochs):
    # Run one epoch using minibatches.
    train_loss = 0
    for step, (x, theta) in enumerate(data_loaders['train']):
        # M: We enumerate and get a step, an x, and a theta for each 10 x 3 datapoint.
            
        # Get the output of module, which is a distribution from the MDN fitting.
        y = module(x.float()) 
        # Needed to apply the float() function here since we "expect object of type float
            
        # The loss function is obtained from evaluate_negative_log_likelihood.
        loss: th.Tensor = loss_function(theta, y)
        # There's an error here where we're trying to plug in something of our batch size, but we expect something of size 2.
        assert not loss.isnan()
            
        optimizer.zero_grad() # Clear the gradients from last step.
        loss.backward() # Compute derivative of loss wrt to parameters.
        optimizer.step() # Take a step based on the gradients.
            
        # Extract the loss value as a float, to keep a running sum for this epoch.
        train_loss += loss.item()
        
    # Get the average training loss.
    train_loss /= step + 1
        
    # Evaluate the validation loss and update the learning rate if required.
    # The validation loss is calculated by sticking the training set (x's and corresponding thetas)
    # into the module (that has gone one step of optimization), and then getting the EPE.
    validation_loss = sum(loss_function(theta, module(x)).mean().item() for x, theta
                              in data_loaders['validation']) / len(data_loaders['validation'])
    scheduler.step(validation_loss)
        
    # Update the best validation loss.
    # M: An epoch is "bad" if our validation loss did not improve (get lower).
    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        num_bad_epochs = 0
    else:
        num_bad_epochs += 1
        
    epoch += 1
    print("epoch %3d: train loss = %.3f; validation loss = %.3f \nbest validation loss = %.3f; number bad epochs = %d / %d" % (epoch, train_loss, validation_loss,
                  best_validation_loss, num_bad_epochs, patience))
    
print("Training complete, saving results")
    
# Save the results.
path_to_models = r"models/" # Where we'll store our data.
if not os.path.exists(path_to_models):
    os.makedirs(path_to_models)
    
compressor_path = path_to_models + "compressor.pt"
th.save(compressor, compressor_path)
    
mdn_path = path_to_models + "mdn.pt"
th.save(module, mdn_path)

execution_time = (time.time() - start_time)
print('Execution time in seconds: ' + str(execution_time))
