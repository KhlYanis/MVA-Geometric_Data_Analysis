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
    
    def get_subject_labels(self, subjectIDs):
        # Convert the subjectIDs from str to int 
        id_list = list(map(int, subjectIDs))

        # Setup the label dictionary
        label_dict = {}

        for id in id_list:
            label_dict[id] = self.pheno_df_all[self.pheno_df_all['SUB_ID'] == id]['DX_GROUP'].values

        return label_dict
    