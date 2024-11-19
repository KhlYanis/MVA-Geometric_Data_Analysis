import torch
import torch.nn as nn
import torch.nn.functional as func


def compute_normalized_adj_matrix(adjacency_matrix, N, device):
        # Adjacency matrix with added self-connections
        A_tilde = torch.add(adjacency_matrix, torch.eye(N))

        # Compute D^{-1/2}
        deg_v = torch.sum(adjacency_matrix, dim = 1) 
        inv_sqrt_D = torch.diag(1./torch.sqrt(deg_v))

        # Compute the normalized adjacency
        normalized_adjacency = torch.matmul(torch.matmul(inv_sqrt_D, A_tilde), inv_sqrt_D).to(device)

        return normalized_adjacency

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

    def __init__(self, args, input_dim, hidden_dim, adjacency_matrix : torch.tensor):
        super().__init__()

        # Set the device to 'cuda' if GPU is available
        self.device = "cuda" if args.use_cuda else "cpu"

        self.dropout_rate = args.dropout_rate

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = 1     # Since it's a binary classification

        ## ADJACENCY MATRIX :
        self.adjacency_matrix = adjacency_matrix.to(self.device)
        # Get the size of the adjacency matrix
        self.N = self.adjacency_matrix.shape[0]

        # Compute the normalized adjacency
        self.normalized_adj_matrix = compute_normalized_adj_matrix(self.adjacency_matrix, self.N, self.device)

        self.InitGCNBlock()

    
    def InitGCNBlock(self):
        self.gcn_layer1 = nn.Sequential(
            DenseLayerGCN(self.input_dim, self.hidden_dim),
            nn.Dropout(p = self.dropout_rate)
            ).to(self.device)
        
        self.FC = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_dim)
        ).to(self.device)

    def forward(self, x):
        # Pass through the GCN layer
        out_gcn = self.gcn_layer1(x)

        # Pass through the output layer
        out_FC = self.FC(out_gcn)

        return  out_FC


class ChebConvLayer(torch.nn.Module):
    def __init__(self, args, adjacency_matrix, K, in_features, out_features = 1):
        super(ChebConvLayer, self).__init__()

        self.device = "cuda" if args.use_cuda else "cpu"

        # Input and output dimension
        self.input_dim = in_features
        self.output_dim = out_features

        ## Adjacency matrix 
        self.adjacency_matrix = adjacency_matrix
        self.N = self.adjacency_matrix.shape[0]
        self.normalized_adjacency = compute_normalized_adj_matrix(self.adjacency_matrix, self.N, self.device)
        
        # Compute the L_tilde matrix
        self.L_tilde = self.get_laplacian()

        # Chebyshev approximation order
        ## For ABIDE dataset, K is set to 3 (May be changed for another dataset)
        self.K = K
        self.initialize_Cheb_polynomials()

        # Initialize the layer associated to each term of the polynomial expansion
        self.layers = nn.ModuleList([
            nn.Linear(self.input_dim, self.output_dim).to(self.device) for _ in range(self.K)
        ])
        
        self.bias = torch.nn.Parameter(torch.zeros(self.output_dim)).to(self.device)

        ## Check if the number of layers and polynomials correctly match
        if len(self.ChebPolynomials) != len(self.layers):
            raise ValueError("Number of Chebyshev polynomials and layers must match!")

    def get_laplacian(self):
        ## Compute the Laplacian of the normalized adjacency matrix
        lap_matrix = torch.eye(self.N, device = self.device) - self.normalized_adjacency

        # Get the highest eigenvalue of the Laplacian matrix
        eigvals_lap, _ = torch.linalg.eig(lap_matrix)
        max_eigval = eigvals_lap.real.max()

        # Compute L_tilde = 2/lambda_max * (Lap) - I_{N}
        L_tilde = (2.0/max_eigval)*lap_matrix - torch.eye(self.N, device = self.device)

        return L_tilde
    
    def initialize_Cheb_polynomials(self):
        # T_{0}(x) = 1 
        T0 = torch.eye(self.N, device = self.device)
        # T_{1}(x) = x
        T1 = self.L_tilde

        # List with the terms of the Tchebyshev polynomial expansion
        self.ChebPolynomials = [T0, T1]
        for k in range(2, self.K):
            # Compute T_{k}(x) = 2xT_{k-1}(x) + T_{k_2}(x)
            T_k = 2. * torch.matmul(self.L_tilde, self.ChebPolynomials[-1]) - self.ChebPolynomials[-2]
            self.ChebPolynomials.append(T_k)


    def forward(self, x):
        output = 0

        ## Enumerate through the different terms of the polynomial expansion
        for id, T_k in enumerate(self.ChebPolynomials):
            # Compute the output of the linear layer
            layer_output = self.layers[id](torch.matmul(T_k,x))
            # Add it to the previous outputs
            output += layer_output

        output += self.bias

        return output
    

class ChebGCN (torch.nn.Module):

    def __init__(self, args, in_features, out_features, 
                 adjacency_matrix):
        super(ChebGCN, self).__init__()

        self.device = "cuda" if args.use_cuda else "cpu"

        self.input_dim = in_features
        self.output_dim = out_features

        self.K = args.K
        self.dropout_rate = args.dropout_rate

        self.adjacency_matrix = adjacency_matrix.to(self.device)

        ## Building the Chebyshev GCN layers
        self.layers = nn.ModuleList()

        # First layer (input to hidden)
        self.layers.append(
            ChebConvLayer(args, in_features, args.hidden_dim, self.adjacency_matrix, self.K)
        )

        # Hidden layers
        for _ in range(args.num_layers - 2):
            self.layers.append(
                ChebConvLayer(args, args.hidden_dim, args.hidden_dim, self.adjacency_matrix, self.K)
            )

        # Output layer (hidden to output)
        self.layers.append(
            ChebConvLayer(args, args.hidden_dim, out_features, self.adjacency_matrix, self.K)
        )

    def forward(self, x):
        
        x = x.to(self.device)

        for i, layer in enumerate(self.layers):
            x = layer(x)  # Apply ChebConvLayer
            if i < len(self.layers) - 1:  # Apply dropout and activation for all but the last layer
                x = torch.nn.functional.relu(x)
                x = torch.nn.functional.dropout(x, self.dropout_rate, training=self.training)

        return x

