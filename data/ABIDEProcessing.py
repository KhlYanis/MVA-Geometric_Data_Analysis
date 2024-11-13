import nilearn
from nilearn.datasets import fetch_abide_pcp
from nilearn import connectome
import os
import numpy as np
from pathlib import Path

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

    for idx, filename in enumerate(FileNames) : 
        print(f"Reading time series for subject {subject_IDs[idx]}")
        timeseries.append(np.loadtxt(DATA_FOLDER + filename[0], skiprows=0))
    
    return timeseries