import sys
import pickle
import os
import numpy as np
import time
import epidemic_utils

mode = str(sys.argv[1])
num_threads = int(sys.argv[2])

# Now, gather the samples
# Note that samples are stored as a dictionary. 
# For each key in the dictionary, we just keep appending.

all_samples = "undefined"

start_time = time.time()
# First, find all the output files
for i in range(int(num_threads)):
    iu = i+1
    file_path = "data/" + mode + "_data_" + str(iu) + ".pbz2"
    if os.path.isfile(file_path): 
        ind_samp = epidemic_utils.decompress_pickle(file_path)
        if all_samples == "undefined":
            all_samples = {}
            for k in list(ind_samp.keys()):
                all_samples[k] = ind_samp[k].tolist() # Get it t a list for faster processing. 
        else:
            # Now, we need to appendi for each key (theta, output, array)
            for k in list(all_samples.keys()):
                all_samples[k].extend(ind_samp[k].tolist()) # Use an extension, since we want to just keep attaching
        print("Loaded in file at " + file_path)
    else:
        print("ERROR: Failed to find file at " + file_path)
num_samples = len(all_samples[list(all_samples.keys())[0]])
print("Gathered a total of " + str(num_samples) + " samples")

print("Converting back to numpy")
for k in list(all_samples.keys()):
    all_samples[k] = np.asarray(all_samples[k]) 

# Finally, dump our new file out.
unified_path = "data/" + str(mode) + "_data"
epidemic_utils.compressed_pickle(unified_path, all_samples)
total_time = time.time()-start_time
print("Total time elapsed: " + str(total_time))


print("Print metrics:")
print(all_samples)
print(len(list(all_samples.keys())))
for k in all_samples.keys():
        print(all_samples[k].shape)
delete = True
if delete:
    print("Samples gathered, deleting")
    for i in range(int(num_threads)):
        iu = i+1
        file_path = "data/" + mode + "_data_" + str(iu) + ".pbz2"
        if os.path.isfile(file_path):
            os.remove(file_path) # Delte the file once we've processed it.
        else:
            print("ERROR: Failed to find (and so could not delete) file at " + file_path)
 
