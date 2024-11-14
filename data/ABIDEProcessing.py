import nilearn
from nilearn.datasets import fetch_abide_pcp
from nilearn import connectome
import os
import numpy as np
from pathlib import Path
import scipy.io as sio
from tqdm import tqdm

## ROOT_FOLDER
ROOT_FOLDER = Path(__file__).parent.parent.absolute()

## DATA_FOLDER : Where we can find the imaging dataset
DATA_FOLDER = os.path.join(ROOT_FOLDER, 'data\\ABIDE_dataset\\ABIDE_pcp\\cpac\\filt_noglobal\\')
## Path for the phenotypic dataset (.csv)
phenotypic_data_path = os.path.join(ROOT_FOLDER, 'data\\ABIDE_dataset\\ABIDE_pcp\\Phenotypic_V1_0b_preprocessed1.csv')



def get_subjectIDs(fileName = "subject_IDs.txt"):
    """Get the IDs of the subjects that are studied in the ABIDE dataset"""
    data_path = os.path.join(ROOT_FOLDER, fileName)

    with open(data_path, "r") as file :
        contenu = file.readlines()

    id_list = [ligne.strip() for ligne in contenu]
    
    return id_list



def fetch_filenames(subject_IDs):
    """Get the filename for each studied patient"""
    import glob
    
    # Initialize the list of filenames
    filenames = []

    for idx in subject_IDs :
        os.chdir(DATA_FOLDER)
        try: 
            filenames.append(glob.glob('*' + idx + '_rois_ho.1D'))
        except IndexError:
            filenames.append('N/A')

    return filenames


def get_timeseries(subject_IDs, FileNames):
    """
        subject_IDs : list of the subject IDs
        FileNames : list of the subjects associated filenames

    returns:
        time_series  : list of timeseries arrays, each of shape (timepoints x regions)
    """

    timeseries = []

    for _, filename in tqdm(enumerate(FileNames), total = len(FileNames), desc = "Loading time series") : 
        timeseries.append(np.loadtxt(DATA_FOLDER + filename[0], skiprows=0))
    
    return timeseries

def get_subject_connectivity(time_series, subjectID, kind = 'correlation', save = True):
    """
    time_series : timeseries table for a certain subject [timepoints, region]
    subjectID : the subject ID
    kind : type of connectivity to be used (In our case, we use the correlation)
    save : flag to save the connectivity matrix somewhere
    save_path : path to save the connectivity matrix
    """
    # Create a directory to save the connectivity matrices if not already existing
    save_directory = os.path.join(ROOT_FOLDER, 'data', 'Connectivity_matrices')

    if os.path.exists(save_directory) == False :
        os.makedirs(save_directory)

    # Now, we can compute the connectivity matrix for the subject
    conn_measure = connectome.ConnectivityMeasure(kind = kind)
    connectivity_mat = conn_measure.fit_transform([time_series])[0]

    if save :
        subject_file = os.path.join(save_directory, 'sub_' + subjectID + '_' + kind + '.mat' )
        sio.savemat(subject_file, {'connectivity' : connectivity_mat})

    return connectivity_mat

