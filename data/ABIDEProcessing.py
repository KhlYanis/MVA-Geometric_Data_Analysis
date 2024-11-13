import nilearn
from nilearn.datasets import fetch_abide_pcp
from nilearn import connectome
import os
import numpy as np

## ROOT_FOLDER : To change 
ROOT_FOLDER = 'c:\\Users\\yanis\\OneDrive\\Bureau\\M2 MVA\\S1\\Geometric Data Analysis\\MVA-Geometric_Data_Analysis\\'
## DATA_FOLDER : Where we can find the imaging dataset
DATA_FOLDER = os.path.join(ROOT_FOLDER, 'data\\ABIDE_dataset\\ABIDE_pcp\\cpac\\filt_noglobal\\')

##
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


def get_timeseries(FileNames):
    """
        FileNames : list of the subjects associated filenames

    returns:
        time_series  : list of timeseries arrays, each of shape (timepoints x regions)
    """

    timeseries = []

    for _, filename in enumerate(FileNames) : 
        timeseries.append(np.loadtxt(DATA_FOLDER + filename[0], skiprows=0))
    
    return timeseries