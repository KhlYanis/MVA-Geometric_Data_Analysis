import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import scipy.io as sio


"""Class to build the adjacency matrix of our
        dataset"""

class Adj_matrix : 

    def __init__(self, df, subjectIDs, root_folder):
        super().__init__()

        self.root_folder = root_folder
        # Setup the DataFrame and the subject IDs
        self.df = df
        self.subjectIDs = subjectIDs

        self.nb_subjects = len(subjectIDs)

    def retrieve_connectivity_matrices(self, kind = 'correlation'):
        # Initialize the list to stock the connectivity matrices
        self.all_networks = []
        # Get the path of the folder where the connectivity matrices are stored
        cm_folder = os.path.join(self.root_folder, 'data', 'Connectivity_matrices')

        for subjectID in self.subjectIDs :
            # Get the path of the connectivity matric of subject i
            cm_path = os.path.join(cm_folder, 'sub_' + subjectID + '_' + kind + '.mat')
            # Retrieve the connectivity matrix
            c_matrix = sio.loadmat(cm_path)['connectivity']
            # Append the connectivity matrix to the list
            self.all_networks.append(c_matrix)

    def get_feature_vectors(self):
        # Retrieve the connectivity matrices
        self.retrieve_connectivity_matrices()

        # Get the superior triangular part of the matrix
        idx = np.triu_indices_from(self.all_networks[0], 1)

        # Apply the Fisher transformation on each connectivity matrix
        norm_networks = [np.arctanh(mat) for mat in self.all_networks]
        # Vectorization of the network
        vec_networks = [mat[idx] for mat in norm_networks]
        # Build the connectivity network
        matrix = np.vstack(vec_networks)

        return matrix

    def score_mat_on_phenotypic_attr(self):
        # Initialize the matrix
        self.score_mat = np.zeros((self.nb_subjects, self.nb_subjects))

        for i, subject_id in tqdm(enumerate(self.subjectIDs), total = self.nb_subjects, desc = "Building Adjacency matrix"):
            # Row associated to subject "i"
            row_i = self.df[self.df['SUB_ID'] == int(subject_id)]

            for j in range(i + 1, self.nb_subjects):
                score = 0.0
                # Retrieve the ID of subject "j"
                SUB_ID_j = self.subjectIDs[j]
                # Row associated to subject "j"
                row_j = self.df[self.df['SUB_ID'] == int(SUB_ID_j)]

                if row_i['SITE_ID'].values == row_j['SITE_ID'].values :
                    score += 1.0

                if row_i['SEX'].values == row_j['SEX'].values :
                    score += 1.0

                self.score_mat[i, j] = score
                # Since the score matrix is symmetric
                self.score_mat[j, i] = score

        return self.score_mat

    
    ## TO BE COMPLETED
    def feature_selection(self, n_features):
        return 
    
    def compute_similarity_value(self):
        return
    
    def compute_adjacency_matrix(self):
        return
    