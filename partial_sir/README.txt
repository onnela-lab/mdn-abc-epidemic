Partially observed SIR

Files to run:
run_initial_epidemic.py
  - Generate and store a simulated SIR epidemic with weekly testing cadence.
simple_generate.py --mode --number_samples
  - Generate and store simulated datasets for neural network training
  - For best results, parallelize
  - unify.py includes basic framework for gathering datasets generated in parallel.
gpu_train_nn.py
  - Train an MDN using training data; validate using validation dataset.
abc_framework.py --acceptance_percentage
  - Rejection ABC that resamples from training set and uses MDN for MDN-ABC.
  - Accepts the best <acceptance_percentage> percent of samples. Our paper uses 0.02 percent.

Utility files:
network_reconstruction.py
  - Functions for generating networks
epidemic_utils.py
  - Functions for generating epidemics and sampling several epidemics simultaneously.
nn_utils.py
  - Definitions of modules for neural network training.
