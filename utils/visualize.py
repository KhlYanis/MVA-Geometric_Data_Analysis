import matplotlib.pyplot as plt
from nilearn import plotting
import os
import scipy.io as sio

class visualization :

    def __init__(self, subjectIDs, time_series, root_folder):
        super(visualization).__init__()

        # Setup the subjectID and time series list
        self.subjectIDs = subjectIDs
        self.ts = time_series

        # Setup the path to retrieve the connectivity matrices
        self.cv_matrix_path = os.path.join(root_folder, 'data', 'Connectivity_matrices')

    def plot_ts(self, idx):
        # Retrieve the time series for the subject "idx"
        subject_ts = self.ts[idx]

        # Plot the time series
        plt.plot(subject_ts[:, :], alpha = 0.5)
        plt.ylim((-400, 400))
        plt.title(f"Subject #{self.subjectIDs[idx]} - All brain regions")
        plt.grid()
        plt.show()

    def plot_connectivity_matrix(self, idx):

        # Retrieve the connectivity matrix
        connectivity_matrix = sio.loadmat(self.cv_matrix_path + '\\sub_' 
                                          + self.subjectIDs[idx] + '_correlation.mat')['connectivity']
        
        plotting.plot_matrix(connectivity_matrix, figure=(7, 7), vmin = -1, vmax = 1,
                title=f"Connectivity matrix - Subject #{self.subjectIDs[idx]}", cmap='viridis')
        plt.show()


        
        