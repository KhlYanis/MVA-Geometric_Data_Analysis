import pandas as pd
import numpy as np
import os


class PhenoDataProcessing :

    def __init__(self, root_folder):
        super().__init__()

        # Setup the path of the phenotype DataFrame
        self.path_to_data = os.path.join(root_folder, 'data', 'ABIDE_dataset',
                                          'ABIDE_pcp', 'Phenotypic_V1_0b_preprocessed1.csv')

        # Retrieve the complete phenotype DataFrame
        self.pheno_df_all = pd.read_csv(self.path_to_data)

    def extract_subjects(self, subjectIDs):
        # Convert the subjectIDs from str to int 
        id_list = list(map(int, subjectIDs))

        # Get the DataFrame for the relevant subjects
        df = self.pheno_df_all[self.pheno_df_all['SUB_ID'].isin(id_list)]

        return df
    
    def get_subject_labels(self, subjectIDs, column_names):
        # Convert the subjectIDs from str to int 
        id_list = list(map(int, subjectIDs))

        # Setup the label dictionary
        label_dict = {}

        for id in subjectIDs:
            label_dict[id] = {}
            for c_name in column_names :
                label_dict[id][c_name] = self.pheno_df_all[self.pheno_df_all['SUB_ID'] == int(id)][c_name].values[0]

        return label_dict
