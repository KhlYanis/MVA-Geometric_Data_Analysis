import torch
import torch.nn as nn
import numpy as np
from utils.preprocess import preprocess_features
from tqdm import tqdm
import torch.nn.functional as func
from torchmetrics import Accuracy
import os 
import random
from sklearn.model_selection import KFold

class DataPipeline :
    def __init__(self, args,
                 nb_patients = 871):
        super(DataPipeline).__init__()

        self.nb_patients = nb_patients
        self.args = args

    def get_kfolds(self, n_splits=10, shuffle=True, seed=42):
        idx = np.arange(self.nb_patients)
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
        return kf.split(idx)

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

    def build_data_dict(self, df, feature_mat, adjacency_matrix, extra_features_tensor = None):
        # Générer les index pour les ensembles de train, validation, et test
        train_idx, val_idx, test_idx = self.get_set_idx()

        # Ajout des données dans un dictionnaire
        data_dict = {
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "labels": self.get_labels(df),
            "inputs": {
                "raw_input": torch.tensor(feature_mat, dtype=torch.float32),
                "raw_normalized_input": preprocess_features(feature_mat)
            },
            "adjacency_matrix": torch.tensor(adjacency_matrix, dtype=torch.float32),
            "extra_features": extra_features_tensor  # Ajout des extra_features
        }

        return data_dict

class TrainTestPipeline:
    def __init__(self, args, data_dict, model, modelName, ROOT_FOLDER, f_vect_type="raw_input"):
        super(TrainTestPipeline, self).__init__()

        self.device = "cuda" if args.use_cuda else "cpu"
        self.args = args
        self.data_dict = data_dict
        self.f_vect_type = f_vect_type

        # On récupère les dimensions d'entrée du modèle
        self.N, self.nb_features = self.data_dict["inputs"][f_vect_type].size()

        # Vérifier si des extra_features sont disponibles
        if "extra_features" in self.data_dict:
            self.extra_features = self.data_dict["extra_features"].to(self.device)
        else:
            self.extra_features = None

        # Initialisation du modèle
        self.model = model.to(self.device)

        self.modelDirectory = os.path.join(ROOT_FOLDER, 'models')
        if not os.path.exists(self.modelDirectory):
            os.makedirs(self.modelDirectory)

        self.modelFileName = os.path.join(self.modelDirectory, modelName) + '_best_model.pt'

        # Sélection des vecteurs de caractéristiques
        self.f_vect_type = f_vect_type

    def get_set_matrix(self, set_id):
        # Matrice d'input pour l'ensemble spécifié (train/val/test)
        mat = torch.zeros([self.N, self.nb_features], dtype=torch.float32).to(self.device)
        mat[set_id] = self.data_dict["inputs"][self.f_vect_type][set_id]
        return mat

    def get_extra_features_for_set(self, set_id):
        # Sélectionner les extra_features pour l'ensemble spécifié (train/val/test)
        if self.extra_features is not None:
            mat = torch.zeros([self.N, self.extra_features.size(1)], dtype=torch.float32).to(self.device)
            mat[set_id] = self.extra_features[set_id]  # Récupérer les extra_features seulement pour les indices concernés
            return mat
        else:
            return torch.zeros(1).to(self.device)  # Retourner un tensor vide si pas d'extra_features

    def NNTrain(self):
        train_idx = self.data_dict["train_idx"]
        self.train_set = self.get_set_matrix(train_idx)
        self.trainLOSS = torch.zeros([self.args.n_epoch])
        self.trainAccuracy = torch.zeros([self.args.n_epoch])

        val_idx = self.data_dict["val_idx"]
        self.val_set = self.get_set_matrix(val_idx)
        self.valLoss = torch.zeros([self.args.n_epoch])
        self.valAccuracy = torch.zeros([self.args.n_epoch])

        best_accuracy = -1.0
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        self.loss_fn = nn.BCEWithLogitsLoss()

        for epoch in range(self.args.n_epoch):
            # TRAINING
            self.model.train()
            self.optimizer.zero_grad()

            # Extra features uniquement pour les indices d'entraînement
            extra_features_train = self.get_extra_features_for_set(train_idx)

            train_logits = torch.squeeze(self.model(self.train_set, extra_features_train))  # Passage à travers le modèle
            loss = self.loss_fn(train_logits[train_idx], self.data_dict["labels"][train_idx])
            loss.backward()
            self.trainLOSS[epoch] = loss.item()
            self.optimizer.step()

            # Accuracy pour l'entraînement
            self.trainAccuracy[epoch] = self.compute_accuracy(train_logits, train_idx)

            # EVALUATION
            self.model.eval()
            with torch.no_grad():
                # Extra features uniquement pour les indices de validation
                extra_features_val = self.get_extra_features_for_set(val_idx)

                val_logits = torch.squeeze(self.model(self.val_set, extra_features_val))  # Passage à travers le modèle
                val_loss = self.loss_fn(val_logits[val_idx], self.data_dict["labels"][val_idx])
                self.valLoss[epoch] = val_loss
                self.valAccuracy[epoch] = self.compute_accuracy(val_logits, val_idx)

            if self.valAccuracy[epoch] > best_accuracy:
                torch.save(self.model, self.modelFileName)
                best_accuracy = self.valAccuracy[epoch]

            print(f"Epoch {epoch} | Train Loss: {self.trainLOSS[epoch]} | Validation Loss: {self.valLoss[epoch]} | Validation Accuracy: {self.valAccuracy[epoch]}")
        return [self.trainLOSS, self.trainAccuracy, self.valLoss, self.valAccuracy]
    
    def NNTrainMiniBatch(self, batch_size):
        ## Get the matrix associated to training and validation set
        ## + Setup the metrics tensors  :

        # Train set
        train_idx = self.data_dict["train_idx"]
        self.trainLOSS = torch.zeros([self.args.n_epoch])
        self.trainAccuracy = torch.zeros([self.args.n_epoch])

        # Validation set
        val_idx = self.data_dict["val_idx"]
        self.val_set = self.get_set_matrix(set_id=val_idx)
        self.valLoss = torch.zeros([self.args.n_epoch])
        self.valAccuracy = torch.zeros([self.args.n_epoch])

        # Best Accuracy
        best_accuracy = -1.0

        # Use Adam as an optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        # Use BCELoss as loss function
        self.loss_fn = nn.BCEWithLogitsLoss()

        for epoch in range(self.args.n_epoch):

            #######################################
            ######### ---- TRAINING ---- ##########
            #######################################
            self.model.train()
            # Set the gradients to zero
            self.optimizer.zero_grad()

            # Select a set of indexes in the train set
            epoch_indexes = random.sample(sorted(train_idx), batch_size)
            # Get the matrix associated to this set of indexes
            self.train_set = self.get_set_matrix(set_id=epoch_indexes)

            # Get the extra features for this batch (training indices)
            extra_features_train = self.get_extra_features_for_set(epoch_indexes)

            # Forward pass with extra features
            train_logits = torch.squeeze(self.model(self.train_set, extra_features_train))
            
            # Compute the loss then backpropagate
            loss = self.loss_fn(train_logits[epoch_indexes], self.data_dict["labels"][epoch_indexes])
            loss.backward()

            # Save the training loss
            self.trainLOSS[epoch] = loss.item()

            # Update the network parameters
            self.optimizer.step()

            # Compute and save the training accuracy
            self.trainAccuracy[epoch] = self.compute_accuracy(train_logits, epoch_indexes)

            #######################################
            ######## ---- EVALUATION ---- #########
            #######################################
            self.model.eval()

            with torch.no_grad():
                # Forward the validation data through the neural network
                extra_features_val = self.get_extra_features_for_set(val_idx)
                val_logits = torch.squeeze(self.model(self.val_set, extra_features_val))

                # Compute and save the loss on the validation set
                val_loss = self.loss_fn(val_logits[val_idx], self.data_dict["labels"][val_idx])
                self.valLoss[epoch] = val_loss

                # Compute and save the accuracy on the validation set
                self.valAccuracy[epoch] = self.compute_accuracy(val_logits, val_idx)

            if self.valAccuracy[epoch] > best_accuracy:
                torch.save(self.model, self.modelFileName)
                best_accuracy = self.valAccuracy[epoch]

            print(f"Epoch {epoch} | Train Loss: {self.trainLOSS[epoch]} | Validation Loss: {self.valLoss[epoch]} | Validation Accuracy: {self.valAccuracy[epoch]}")

        return [self.trainLOSS, self.trainAccuracy, self.valLoss, self.valAccuracy]

    def NNTrainMiniBatchKFold(self, data_pipeline, batch_size, n_splits=10):
        kfold_accuracies = []
        kfolds = data_pipeline.get_kfolds(n_splits=n_splits)

        for fold, (train_idx, val_idx) in enumerate(kfolds):
            print(f"Fold {fold + 1}/{n_splits}")

            # Metrics for this fold
            self.trainLOSS = torch.zeros([self.args.n_epoch])
            self.trainAccuracy = torch.zeros([self.args.n_epoch])
            self.valLoss = torch.zeros([self.args.n_epoch])
            self.valAccuracy = torch.zeros([self.args.n_epoch])

            # Reset model weights for each fold
            self.model.apply(self.reset_weights)

            # Use Adam as an optimizer
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
            # Use BCELoss as loss function
            self.loss_fn = nn.BCEWithLogitsLoss()

            # Training loop for epochs
            for epoch in range(self.args.n_epoch):
                #######################################
                ######### ---- TRAINING ---- ##########
                #######################################
                self.model.train()

                # Shuffle training indices for mini-batch sampling
                train_idx_shuffled = random.sample(sorted(train_idx), len(train_idx))
                for i in range(0, len(train_idx_shuffled), batch_size):
                    batch_idx = train_idx_shuffled[i:i + batch_size]

                    # Get the matrix for the current batch
                    self.train_set = self.get_set_matrix(set_id=batch_idx)

                    # Get the extra features for this batch (training indices)
                    extra_features_train = self.get_extra_features_for_set(batch_idx)
                    # Set the gradients to zero
                    self.optimizer.zero_grad()

                    # Forward pass
                    train_logits = torch.squeeze(self.model(self.train_set, extra_features_train))

                    # Compute loss and backpropagate
                    loss = self.loss_fn(train_logits[batch_idx], self.data_dict["labels"][batch_idx])
                    loss.backward()

                    # Update parameters
                    self.optimizer.step()

                    # Accumulate loss for the epoch
                    self.trainLOSS[epoch] += loss.item() / len(train_idx_shuffled)

                # Compute training accuracy for the epoch
                self.trainAccuracy[epoch] = self.compute_accuracy(train_logits, train_idx)

                #######################################
                ######## ---- EVALUATION ---- #########
                #######################################
                self.model.eval()

                with torch.no_grad():
                    # Get the matrix for the validation set
                    self.val_set = self.get_set_matrix(set_id=val_idx)

                    # Get the extra features for validation
                    extra_features_val = self.get_extra_features_for_set(val_idx)

                    # Forward pass on validation data
                    val_logits = torch.squeeze(self.model(self.val_set, extra_features_val))

                    # Compute validation loss
                    val_loss = self.loss_fn(val_logits[val_idx], self.data_dict["labels"][val_idx])
                    self.valLoss[epoch] = val_loss

                    # Compute validation accuracy
                    self.valAccuracy[epoch] = self.compute_accuracy(val_logits, val_idx)

                print(f"Epoch {epoch} | Train Loss: {self.trainLOSS[epoch]} | Validation Loss: {self.valLoss[epoch]} | Validation Accuracy: {self.valAccuracy[epoch]}")

            # Save the accuracy of the best validation epoch for this fold
            best_fold_accuracy = max(self.valAccuracy).item()
            kfold_accuracies.append(best_fold_accuracy)
            print(f"Fold {fold + 1} Best Validation Accuracy: {best_fold_accuracy}")

        # Compute the average accuracy across all folds
        mean_accuracy = np.mean(kfold_accuracies)
        print(f"Mean Accuracy over {n_splits} folds: {mean_accuracy}")
        return kfold_accuracies

    def reset_weights(self, m):
        """
        Reset model weights to ensure independence between folds.
        """
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()

    def compute_accuracy(self, logits, set_idx):
        # Pass through a sigmoid to get a probability
        predicted_proba = func.sigmoid(logits[set_idx])

        # Get the predicted label
        pred_labels = (predicted_proba >= 0.5).long()

        # Compute the accuracy
        accuracy = torch.sum((pred_labels == self.data_dict["labels"][set_idx]))/len(set_idx)

        return accuracy
    
    def NNTest(self):
        ## Get the matrix associated to the test set
        self.test_set = self.get_set_matrix(set_id = self.data_dict["test_idx"])

        model = torch.load(self.modelFileName)
        ## Evaluate the model on the test set and compute the accuracy
        model.eval()
        test_logits = torch.squeeze(self.model(self.test_set))
        
        test_accuracy = self.compute_accuracy(test_logits, self.data_dict["test_idx"])

        return test_accuracy