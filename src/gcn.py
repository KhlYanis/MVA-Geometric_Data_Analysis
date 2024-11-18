import torch
import torch.nn as nn
import torch.nn.functional as func



class DenseLayerGCN (nn.Module) :

    def __init__(self, input_dim, output_dim, adjacency_matrix,
                 use_activation, bias = False, activation = None):
        """
        Dense Graph Convolutional Layer.

        Args:
            input_dim (int): Number of input features per node.
            output_dim (int): Number of output features per node.
            adjacency_matrix (torch.Tensor): Normalized adjacency matrix (e.g., D^(-1/2) A D^(-1/2)).
            use_activation (bool): Whether to apply an activation function.
            bias (bool): Whether to use a bias term in the fully connected layer.
            activation (callable): Custom activation function (default: ReLU).
        """
        super(DenseLayerGCN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim 
        self.use_bias = bias

        ## NOTE : Normalized adjacency matrix
        self.adjacency_matrix = adjacency_matrix

        self.use_activation = use_activation
        self.activation = activation if activation is not None else nn.ReLU()

        # Initialize the neural network block
        self.InitNNBlock()

    def InitNNBlock(self):
        # Setup the Fully connected layer
        self.FC = nn.Linear(self.input_dim, self.output_dim, bias = self.use_bias)

    def forward(self, x):
        # Pass through the FC layer
        out_FC = self.FC(x)

        # Product between $normalized adjacency$ and the output of the FC layer
        out = torch.matmul(self.adjacency_matrix, out_FC)

        if self.use_activation == False :
            return out
        
        return self.activation(out)




class GCN (torch.nn.Module) :

    def __init__(self, input_dim, adjacency_matrix : torch.tensor):
        super().__init__()

        # Set the device to 'cuda' if GPU is available
        self.device = torch.device('cpu')

        self.input_dim = input_dim

        ## ADJACENCY MATRIX :
        self.adjacency_matrix = adjacency_matrix.to(self.device)
        # Get the size of the adjacency matrix
        self.N = self.adjacency_matrix.shape[0]

        # Compute the normalized adjacency
        self.normalized_adj_matrix = self.compute_normalized_adj_matrix()


    def compute_normalized_adj_matrix(self):
        # Adjacency matrix with added self-connections
        A_tilde = torch.add(self.adjacency_matrix, torch.eye(self.N))

        # Compute D^{-1/2}
        deg_v = torch.sum(self.adjacency_matrix, dim = 1) 
        inv_sqrt_D = torch.diag(1./torch.sqrt(deg_v))

        # Compute the normalized adjacency
        normalized_adjacency = torch.matmul(torch.matmul(inv_sqrt_D, A_tilde), inv_sqrt_D).to(self.device)

        return normalized_adjacency
    
    def forward(self):
        return


