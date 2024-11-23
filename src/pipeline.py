import torch
import torch.nn as nn
import numpy as np
from utils.preprocess import preprocess_features
from tqdm import tqdm
import torch.nn.functional as func
from torchmetrics import Accuracy



class DataPipeline :
    def __init__(self, args,
                 nb_patients = 871):
        super(DataPipeline).__init__()

        self.nb_patients = nb_patients
        self.args = args


    def get_set_idx(self, train_ratio = 0.80, val_ratio = 0.10, test_ratio = 0.10, shuffle = True, seed = 42):

        assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "The ratios should sum to 1."

        # Generate all the indexes
        idx = np.arange(self.nb_patients)

        if shuffle:
            if seed is not None:
                np.random.seed(seed)
            np.random.shuffle(idx)

        # Compute the size of training and validation sets
        train_size = int(train_ratio * self.nb_patients)
        val_size = int(val_ratio * self.nb_patients)

        # Get the indexes associated to each set
        train_idx = idx[:train_size]
        val_idx = idx[train_size:train_size + val_size]
        test_idx = idx[train_size + val_size:]

        return train_idx, val_idx, test_idx
    
    def get_labels(self, set):
        # Retrieve the labels of the dataset and put them in {0, 1}
        labels = torch.tensor(set['DX_GROUP'].values, dtype = torch.float32) - 1.0
        return labels
    
    def build_data_dict(self, df, feature_mat, adjacency_matrix):
        train_idx, val_idx, test_idx = self.get_set_idx()

        data_dict = {
            "train_idx" : train_idx,
            "val_idx" : val_idx,
            "test_idx" : test_idx,
            "labels" : self.get_labels(df),
            "inputs" : {
                "raw_input" : torch.tensor(feature_mat, dtype = torch.float32),
                "raw_normalized_input" : preprocess_features(feature_mat)
            },
            "adjacency_matrix" : torch.tensor(adjacency_matrix, dtype = torch.float32)
        }

        return data_dict


class TrainTestPipeline :
    def __init__(self, args, data_dict, model, modelFileName, f_vect_type = "raw_inputs"):
        super(TrainTestPipeline).__init__()

        # Setup the Pipeline device
        self.device = "cuda" if args.use_cuda else "cpu"

        self.args = args
        self.data_dict = data_dict
        self.f_vect_type = "raw_inputs"

        # Retrieve the number of patients & features
        self.N, self.nb_features = self.data_dict["adjacency_matrix"].size()

        # Setup the model 
        self.model = model.to(self.device)
        self.modelFileName = modelFileName

    def get_set_matrix(self, set_id):
        # Initialize a zero matrix of size [N, nb_features]
        mat = torch.zeros([self.N, self.nb_features], dtype = torch.float32).to(self.device)

        # This matrix has the feature vector at rows in set_id, 0 otherwise
        mat[set_id] = self.data_dict["inputs"][self.f_vect_type][set_id]

        return mat


    def NNTrain(self):

        ## Get the matrix associated to training and validation set
        ## + Setup the metrics tensors  :

        # Train set
        train_idx = self.data_dict["train_idx"]
        self.train_set = self.get_set_matrix(set_id = train_idx)
        self.trainLOSS = torch.zeros([self.args.n_epochs])
    
        # Validation set
        val_idx = self.data_dict["val_idx"]
        self.val_set = self.get_set_matrix(set_id = val_idx)
        self.valLoss = torch.zeros([self.args.n_epochs])
        self.valAccuracy = torch.zeros([self.args.n_epochs])

        # Use Adam as an optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.args.lr)
        # Use BCELoss as loss function
        self.loss_fn = nn.BCEWithLogitsLoss()

        for epoch in tqdm(range(self.args.n_epoch)):

            #######################################
            ######### ---- TRAINING ---- ##########
            ####################################### 
            self.model.train()
            # Set the gradients to zero
            self.optimizer.zero_grad()

            # Forward the data through the neural network
            train_logits = self.model(self.train_set)

            # Compute the loss then backpropagate
            loss = self.loss_fn(train_logits[train_idx],self.data_dict["labels"][train_idx])
            loss.backward()

            # Save the training loss
            self.trainLOSS[epoch] = loss.item()

            # Update the network parameters 
            self.optimizer.step()

            #######################################
            ######## ---- EVALUATION ---- #########
            ####################################### 
            self.model.eval()

            with torch.no_grad():
                # Forward the validation data through the neural network
                val_logits = self.model(self.val_set)

                # Compute and save the loss on the validation set
                val_loss = self.loss_fn(val_logits[val_idx], self.data_dict["labels"][val_idx])
                self.valLoss[epoch] =  val_loss
                
                # Pass through a sigmoid to get the predicted class
                pred_proba = func.sigmoid(val_logits).squeeze()
                pred_labels = (pred_proba >= 0.5).long()

                # Compute and save the accuracy on the validation set
                self.valAccuracy[epoch] = torch.sum((pred_labels == self.data_dict["labels"][val_idx]))

        torch.save(self.model, self.modelFileName)

        return [self.trainLOSS, self.valLoss, self.valAccuracy] 
    
    def NNTest(self):
        ## Get the matrix associated to the test set
        self.test_set = self.get_set_matrix(set_id = self.data_dict["test_idx"])

