from nilearn.datasets import fetch_abide_pcp
from nilearn import plotting
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy
from sklearn.preprocessing import OneHotEncoder

from data.ABIDEProcessing import get_subjectIDs, fetch_filenames, \
                             get_timeseries, get_subject_connectivity, ROOT_FOLDER

from utils.visualize import visualization
from data.PhenotypicData import PhenoDataProcessing

from src.build import Adj_matrix
from src.config import general_settings
from src.pipeline_ext import DataPipeline, TrainTestPipeline
from src.pipeline_ext2 import DataPipeline as DataPipeline_ext
from src.pipeline_ext2 import TrainTestPipeline as TrainTestPipeline_ext
from src.gcn import ChebGCN
from src.gcn_ext import ChebGCN_ext

import seaborn as sns



# Get the subjects IDs
subjectIDs = get_subjectIDs()
# Get the filename associated to each subject
fileNames = fetch_filenames(subject_IDs = subjectIDs)

## Get the time series 
time_series = get_timeseries(subjectIDs, fileNames)


## Build the connectivity matrix for each subject
for i in tqdm(range(len(subjectIDs)), total = len(subjectIDs), desc = "Saving the connectivity matrices"):
    _ = get_subject_connectivity(time_series[i], subjectID = subjectIDs[i])


pdp = PhenoDataProcessing(root_folder = ROOT_FOLDER)

## Extract the phenotypic Dataframe associated to the 871 studied patients
df = pdp.extract_subjects(subjectIDs = subjectIDs)

# same weight as in the original paper
adj = Adj_matrix(subjectIDs = subjectIDs, root_folder = ROOT_FOLDER, sort_var = ['SITE_ID', 'SEX'])
adjacency_matrix = adj.compute_adjacency_matrix(nb_features = 2000)

# matrix full of 1
adjacency_matrix_full = np.ones((871, 871))

args = general_settings()
data_pipe = DataPipeline(args)

feature_vector_2000 = adj.get_feature_vectors()
feature_vector_rfe_2000 = adj.feature_selection(feature_vector_2000, 2000, method="rfe")

input_dim = 2000
args.hidden_dim = 20
args.num_layers = 5
args.dropout_rate = 0.3
args.n_epoch = 100

# Reproduction model paper
data_dict = data_pipe.build_data_dict(df, feature_vector_rfe_2000, adjacency_matrix)
chebgcn = ChebGCN(args, in_features = input_dim, out_features = 1, adjacency_matrix = data_dict["adjacency_matrix"])
Train_test_pipe = TrainTestPipeline(args, data_dict, chebgcn, "chebGCN.pt", ROOT_FOLDER)
kfold_accuracies = Train_test_pipe.NNTrainMiniBatchKFold(data_pipe, 128)
kfold_accuracies

# Extension with phenotypic measure SEX and SITE
data_pipe = DataPipeline_ext(args)
extra_features = df[['SITE_ID']]
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(extra_features)
extra_features_tensor = torch.cat((torch.tensor(encoded_features, dtype=torch.float32),torch.tensor(df[['SEX']].values, dtype=torch.float32)), dim=1)
data_dict6 = data_pipe.build_data_dict(df, feature_vector_rfe_2000, adjacency_matrix, extra_features_tensor=extra_features_tensor)
chebgcn = ChebGCN_ext(args, in_features = input_dim, out_features = 1, extra_features = extra_features_tensor, adjacency_matrix = data_dict6["adjacency_matrix"])
Train_test_pipe = TrainTestPipeline_ext(args, data_dict6, chebgcn, "chebGCN.pt", ROOT_FOLDER)
kfold_accuracies6 = Train_test_pipe.NNTrainMiniBatchKFold(data_pipe, 128)
kfold_accuracies6

# Extension with all phenotypic measure
data_pipe = DataPipeline_ext(args)
extra_features_tensor = torch.tensor(df[['DSM_IV_TR','AGE_AT_SCAN','SEX','EYE_STATUS_AT_SCAN','anat_cnr','anat_efc','anat_fber','anat_fwhm','anat_qi1','anat_snr','func_efc','func_fber','func_fwhm','func_dvars','func_outlier','func_quality','func_mean_fd','func_num_fd','func_perc_fd','func_gsr']].values, dtype=torch.float32)
data_dict9 = data_pipe.build_data_dict(df, feature_vector_rfe_2000, adjacency_matrix, extra_features_tensor=extra_features_tensor)
chebgcn = ChebGCN_ext(args, in_features = input_dim, out_features = 1, extra_features = extra_features_tensor, adjacency_matrix = data_dict9["adjacency_matrix"])
Train_test_pipe = TrainTestPipeline_ext(args, data_dict9, chebgcn, "chebGCN.pt", ROOT_FOLDER)
kfold_accuracies9 = Train_test_pipe.NNTrainMiniBatchKFold(data_pipe, 128)
kfold_accuracies9

# Idem but complete graph
data_pipe = DataPipeline_ext(args)
extra_features_tensor = torch.tensor(df[['DSM_IV_TR','AGE_AT_SCAN','SEX','EYE_STATUS_AT_SCAN','anat_cnr','anat_efc','anat_fber','anat_fwhm','anat_qi1','anat_snr','func_efc','func_fber','func_fwhm','func_dvars','func_outlier','func_quality','func_mean_fd','func_num_fd','func_perc_fd','func_gsr']].values, dtype=torch.float32)
data_dict10 = data_pipe.build_data_dict(df, feature_vector_rfe_2000, adjacency_matrix_full, extra_features_tensor=extra_features_tensor)
chebgcn = ChebGCN_ext(args, in_features = input_dim, out_features = 1, extra_features = extra_features_tensor, adjacency_matrix = data_dict10["adjacency_matrix"])
Train_test_pipe = TrainTestPipeline_ext(args, data_dict10, chebgcn, "chebGCN.pt", ROOT_FOLDER)
kfold_accuracies10 = Train_test_pipe.NNTrainMiniBatchKFold(data_pipe, 128)
kfold_accuracies10


plt.figure(figsize=(12,5))
data = {"Validation accuracy": kfold_accuracies + kfold_accuracies6 + kfold_accuracies9 + kfold_accuracies10, "": ["GCN only"]*len(kfold_accuracies) + ["GCN and FC layer \n(SEX and SITE_ID)"]*len(kfold_accuracies6) + ["GCN and FC layer \n(all phenotypic variables)"]*len(kfold_accuracies9)+ ["GCN (complete graph)\n and FC layer \n(all phenotypic variables)"]*len(kfold_accuracies10)}
sns.boxplot(x="", y="Validation accuracy", data=data, palette="muted")
plt.show()