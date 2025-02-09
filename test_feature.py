import numpy as np

# Specify the path to the saved feature file
feature_path = '/data_new/rilyn/ckpt_best/fH_features/fH_47331008_compressed_128f.mov.npy'

# Load the feature from disk
fH = np.load(feature_path)

# Print out the shape and data type
print("Feature shape:", fH.shape)
print("Feature data type:", fH.dtype)

# Optionally, print some summary statistics
print("Minimum value:", fH.min())
print("Maximum value:", fH.max())
print("Mean value:", fH.mean())
