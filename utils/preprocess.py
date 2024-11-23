import numpy as np


def preprocess_features(features):
    # Normalize the feature matrix row-wise using 

    # Compute the norm of each row
    row_norms = np.linalg.norm(features, ord = 1, axis = 1, keepdims = True)
    # Set norm to 1 if the row norm is equal to 0
    row_norms[row_norms == 0] = 1
    # Normalize each row
    normalized_f_matrix = features/row_norms

    return normalized_f_matrix


