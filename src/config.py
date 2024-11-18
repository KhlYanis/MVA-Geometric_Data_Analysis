import argparse

def general_settings():
    ## Hyperparameter setting

    parser = argparse.ArgumentParser(prog = 'GCN', \
                                     description = 'Hyperparameters for training GCN')

    parser.add_argument('--use_cuda', type = bool, default = False, metavar = 'CUDA',
                        help = 'if True, use GPU')
    
    parser.add_argument('--lr', type = float, default = 5e-3, metavar = 'LR',
                        help = 'learning rate (default : 5e-3)')
    
    parser.add_argument('--wd', type = float, default = 5e-4 , metavar = 'L2 Regularization',
                        help = 'regularization paramrter (default : 5e-4) ')
    
    parser.add_argument('--n_epoch', type = int, default = 150, metavar = 'N_Epoch',
                        help = 'Number of training epochs') 
    
    parser.add_argument('dropout_rate', type = float, default = 0.3, metavar = 'DropRate',
                        help = 'Dropout rate')
    
    args = parser.parse_args("")

    return args