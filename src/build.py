import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import scipy.io as sio
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE
from scipy.spatial import distance


"""Class to build the adjacency matrix of our
        dataset"""

class Adj_matrix : 

    def __init__(self, subjectIDs, root_folder):
        super().__init__()

        self.root_folder = root_folder

        # Setup the subject IDs
        self.subjectIDs = subjectIDs
        self.nb_subjects = len(subjectIDs) 

        ## PHENOTYPIC DATA
        # Setup the path of the phenotype DataFrame
        path_to_data = os.path.join(root_folder, 'data', 'ABIDE_dataset',
                                          'ABIDE_pcp', 'Phenotypic_V1_0b_preprocessed1.csv')
        
        # Retrieve the complete phenotype DataFrame
        self.pheno_df_all = pd.read_csv(path_to_data)

        # Build the DataFrame for the relevant subjects
        self.df = self.extract_subjects(self.subjectIDs)

    def extract_subjects(self, subjectIDs):
        # Convert the subjectIDs from str to int 
        id_list = list(map(int, subjectIDs))

        # Get the DataFrame for the relevant subjects
        df = self.pheno_df_all[self.pheno_df_all['SUB_ID'].isin(id_list)]

        return df

    def retrieve_connectivity_matrices(self, kind = 'correlation'):
        # Initialize the list to stock the connectivity matrices
        self.all_networks = []
        # Get the path of the folder where the connectivity matrices are stored
        cm_folder = os.path.join(self.root_folder, 'data', 'Connectivity_matrices')

        for subjectID in tqdm(self.subjectIDs, total = self.nb_subjects, desc = "Retrieving the connectivity matrices") :
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
        feat_v_matrix = np.vstack(vec_networks)

        return feat_v_matrix

    def score_mat_on_phenotypic_attr(self):
        # Initialize the matrix
        self.score_mat = np.zeros((self.nb_subjects, self.nb_subjects))

        for i, subject_id in tqdm(enumerate(self.subjectIDs), total = self.nb_subjects, desc = "Building Score matrix"):
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

    
    def feature_selection(self, feat_vectors, n_features):

        # Initialize the Ridge Classifier
        estimator = RidgeClassifier()
        selector = RFE(estimator, n_features_to_select = n_features, step = 100, verbose = 1)

        X = feat_vectors
        Y = self.df['DX_GROUP']
        selector = selector.fit(X, Y.ravel())

        # Selection of the features
        features = selector.transform(feat_vectors)

        return features
    
    def compute_similarity_value(self):

        # Get the feature vectors
        feature_vectors = self.get_feature_vectors()

        # Get the similarity matrices with the $n_features$ most relevant features
        red_sim_mat = self.feature_selection(feature_vectors, n_features = 100)

        # Compute the correlation between each of them 
        distv = distance.pdist(red_sim_mat, metric='correlation')

        # Convert to a square symmetric distance matrix
        dist = distance.squareform(distv)
        sigma = np.mean(dist)
        # Get affinity from similarity matrix
        sparse_graph = np.exp(- dist ** 2 / (2 * sigma ** 2))

        return sparse_graph
    
    def compute_adjacency_matrix(self):
        # Retrieve the score matrix on the phenotypic features 
        print("Computing the score matrix on the phenotypic features ...")
        score = self.score_mat_on_phenotypic_attr()
        print("DONE")

        # Retrieve the correlation matrix on the similarities
        print("Computing the correlation matrix on the similarities ...")
        sim_matrix_corr = self.compute_similarity_value()
        print("DONE")

        return score * sim_matrix_corr
    