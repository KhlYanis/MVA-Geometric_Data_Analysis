import pandas as pd
import numpy as np
from tqdm import tqdm

"""Class to build the adjacency matrix of our
        dataset"""

class Adj_matrix : 

    def __init__(self, df, subjectIDs):
        super().__init__()

        # Setup the DataFrame and the subject IDs
        self.df = df
        self.subjectIDs = subjectIDs

        self.nb_subjects = len(subjectIDs)

    def score_mat_on_phenotypic_attr(self , threshold):
        # Initialize the matrix
        self.score_mat = np.zeros((self.nb_subjects, self.nb_subjects))

        for i, subject_id in tqdm(enumerate(self.subjectIDs), total = self.nb_subjects, desc = "Building Adjacency matrix"):
            # Row associated to subject "i"
            row_i = self.df[self.df['SUB_ID'] == subject_id]
            for j in range(i + 1, self.nb_subjects):
                score = 0
                # Retrieve the ID of subject "j"
                SUB_ID_j = self.subjectIDs[j]
                # Row associated to subject "j"
                row_j = self.df[self.df['SUB_ID'] == SUB_ID_j]


                if row_i['SITE_ID'] == row_j['SITE_ID'] :
                    score += 1

                if row_i['SEX'] == row_j['SEX'] :
                    score += 1

                self.score_mat[i, j] = score
                self.score_mat[j, i] = score

        return self.score_mat
    