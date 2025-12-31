import time
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# local utility methods
src_path = '../../'
sys.path.append(src_path + 'data/')
from dataMethods import DataHandler  # pyright: ignore[reportMissingImports]

sys.path.append(src_path + 'nn/mlp/')
from torch_mlp import MLP # pyright: ignore[reportMissingImports]

class DeepONet(nn.Module):
    
    def __init__(self, num_layers, num_neurons, act, \
                 num_br_outputs, num_tr_outputs, \
                 num_inp_fn_points, \
                 out_coordinate_dimension, \
                 num_Y_components, save_file=None, \
                 dropout=False, dropout_rate=0.5, heteroscedastic=False):

        super(DeepONet, self).__init__()
        self.name = 'DeepONet'

        # Set device (use CUDA if available)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        self.num_inp_fn_points = num_inp_fn_points
        self.num_br_outputs = num_br_outputs
        self.num_tr_outputs = num_tr_outputs
        self.out_coordinate_dimension = out_coordinate_dimension
        self.num_Y_components = num_Y_components
        self.heteroscedastic = heteroscedastic
        self.save_file = save_file
        if save_file is None:
            self.save_file = './DeepONet_model/model.pkl'
        self.effective_br_outputs = num_br_outputs * 2 if self.heteroscedastic else num_br_outputs
        self.effective_Y_components = num_Y_components * 2 if self.heteroscedastic else num_Y_components
        # branch network
        self.branch_net = MLP(input_size=num_inp_fn_points, \
                              hidden_size=num_neurons, \
                              num_classes=self.effective_br_outputs, \
                              depth=num_layers, \
                              act=act, dropout=dropout, \
                              dropout_rate=dropout_rate)
        self.branch_net.float().to(self.device)

        # trunk network
        self.trunk_net = MLP(input_size=out_coordinate_dimension, \
                             hidden_size=num_neurons, \
                             num_classes=num_tr_outputs, \
                             depth=num_layers, \
                             act=act, dropout=dropout, \
                             dropout_rate=dropout_rate)
        self.trunk_net.float().to(self.device)
        
        # bias added to the product of branch and trunk networks
        self.bias = nn.ParameterList([nn.Parameter(torch.zeros((1,), device=self.device), requires_grad=True) for i in range(self.effective_Y_components)])

        # dimension d_o of the pointwise value of the target function (u(x) \in R^d_o)
        self.num_Y_components = num_Y_components

        # record losses
        self.train_loss_log = []
        self.test_loss_log = []

        # metadata (used in applications, e.g., bayesian inversion)
        self.metadata = None
    
    def convert_np_to_tensor(self, array):
        if isinstance(array, np.ndarray):
            tensor = torch.from_numpy(array)
            return tensor.to(torch.float32).to(self.device)
        else:
            return array.to(self.device)
    
    def forward(self, X, X_trunk):

        X = self.convert_np_to_tensor(X)
        X_trunk = self.convert_np_to_tensor(X_trunk)
        
        branch_out = self.branch_net.forward(X)
        trunk_out = self.trunk_net.forward(X_trunk,final_act=True)

        if self.heteroscedastic:
            # when modeling heteroscedastic noise, split the outputs accordingly
            branch_out_mean = branch_out[:, :self.num_br_outputs]
            branch_out_logvar = branch_out[:, self.num_br_outputs:]
            trunk_out_mean = trunk_out
            trunk_out_logvar = trunk_out

            if self.num_Y_components == 1:
                mean_output = branch_out_mean @ trunk_out_mean.t() + self.bias[0]
                logvar_output = branch_out_logvar @ trunk_out_logvar.t() + self.bias[1]
            else:
                # when d_o > 1, split the branch output and compute the product
                mean_outputs = []
                logvar_outputs = []
                for i in range(self.num_Y_components):
                    mean_outputs.append(branch_out_mean[:,i*self.num_tr_outputs:(i+1)*self.num_tr_outputs] @ trunk_out_mean.t() + self.bias[2*i])
                    logvar_outputs.append(branch_out_logvar[:,i*self.num_tr_outputs:(i+1)*self.num_tr_outputs] @ trunk_out_logvar.t() + self.bias[2*i+1])
                
                # stack and reshape 
                mean_output = torch.stack(mean_outputs, dim=-1)
                mean_output = mean_output.reshape(-1, X_trunk.shape[0] * self.num_Y_components)
                logvar_output = torch.stack(logvar_outputs, dim=-1)
                logvar_output = logvar_output.reshape(-1, X_trunk.shape[0] * self.num_Y_components)
            return mean_output, logvar_output
        else:
            if self.num_Y_components == 1:
                output = branch_out @ trunk_out.t() + self.bias[0]
            else:
                # when d_o > 1, split the branch output and compute the product
                output = []
                for i in range(self.num_Y_components):
                    output.append(branch_out[:,i*self.num_tr_outputs:(i+1)*self.num_tr_outputs] @ trunk_out.t() + self.bias[i])
                # stack and reshape 
                output = torch.stack(output, dim=-1)
                output = output.reshape(-1, X_trunk.shape[0] * self.num_Y_components)
            return output
    
    def train(self, train_data, test_data, \
              batch_size=32, epochs = 1000, \
              lr=0.001, log=True, \
              loss_print_freq = 100, \
              save_model = False, save_file = None, save_epoch = 100):

        self.epochs = epochs
        self.batch_size = batch_size
        self.save_epoch = save_epoch
        
        if save_file is not None:
            self.save_file = save_file
        

        # train and test dataloaders to sample batches of data
        dataset = DataHandler(train_data['X_train'], \
                              train_data['X_trunk'], train_data['Y_train'])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 

        test_dataset = DataHandler(test_data['X_train'], \
                                   test_data['X_trunk'], test_data['Y_train'])
        test_dataloader = DataLoader(test_dataset, \
                                     batch_size=batch_size, shuffle=True)

        # store coordinates for trunk network
        X_trunk = dataset.X_trunk
        
        if self.heteroscedastic:
            # Negative Log Likelihood for Gaussian
            criterion = nn.GaussianNLLLoss()
            print("Training with Heteroscedastic Aleatoric Uncertainty (GaussianNLLLoss).")
        else:
            criterion = nn.MSELoss()
            print("Training with Homoscedastic assumption (MSELoss).")
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

        self.train_loss_log = np.zeros((epochs, 1))
        self.test_loss_log = np.zeros((epochs, 1))

        # training and testing loop
        start_time = time.perf_counter()

        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print('-'*50)
        print('Starting training with {} trainable parameters...'.format(self.trainable_params))
        print('-'*50)
        
        for epoch in range(1, epochs+1):

            train_losses = []
            test_losses = []

            t1 = time.perf_counter()

            # training loop
            for X_train, _, Y_train in dataloader:

                # move data to device
                X_train = X_train.to(self.device)
                Y_train = Y_train.to(self.device)

                # clear gradients
                optimizer.zero_grad()

                # forward pass through model
                if self.heteroscedastic:
                    Y_train_pred_mean, Y_train_pred_logvar = self.forward(X_train, X_trunk)
                    loss = criterion(Y_train_pred_mean, Y_train, torch.exp(Y_train_pred_logvar))
                else:
                    Y_train_pred = self.forward(X_train, X_trunk)
                    loss = criterion(Y_train_pred, Y_train)
                
                train_losses.append(loss.item())

                # backward pass
                loss.backward()

                # update parameters
                optimizer.step()

            # update learning rate
            scheduler.step()

            # testing loop
            with torch.no_grad():
                for X_test, _, Y_test in test_dataloader:
                    
                    # move data to device
                    X_test = X_test.to(self.device)
                    Y_test = Y_test.to(self.device)
                    if self.heteroscedastic:
                        Y_train_test_mean, Y_train_test_logvar = self.forward(X_test, X_trunk)
                        test_loss = criterion(Y_train_test_mean, Y_test, torch.exp(Y_train_test_logvar))
                    else:
                        Y_test_pred = self.forward(X_test, X_trunk)
                        test_loss = criterion(Y_test_pred, Y_test)
                    test_losses.append(test_loss.item())

            # log losses
            self.train_loss_log[epoch-1, 0] = np.mean(train_losses)
            self.test_loss_log[epoch-1, 0] = np.mean(test_losses)

            # print loss and time
            t2 = time.perf_counter()
            epoch_time = t2 - t1

            if log == True and (epoch % loss_print_freq == 0 or epoch == epochs or epoch == 1):
                loss_label = "NLL" if self.heteroscedastic else "L2 squared"
                print('-'*50)
                print('Epoch: {:5d}, Train Loss ({}}): {:.3e}, Test Loss ({}}): {:.3e}, Time (sec): {:.3f}'.format(epoch, \
                                    loss_label, np.mean(self.train_loss_log[epoch-1, 0]), \
                                    loss_label, np.mean(self.test_loss_log[epoch-1, 0]), \
                                    epoch_time))
                print('-'*50)

            # check if we need to save model parameters
            if save_model == True and (epoch % save_epoch == 0 or epoch == epochs):
                torch.save(self, self.save_file)
                print('-'*50)
                print('Model parameters saved at epoch {}'.format(epoch))
                print('-'*50)

        # print final message
        end_time = time.perf_counter()
        print('-'*50)
        print('Train time: {:.3f}, Epochs: {:5d}, Batch Size: {:5d}, Final Train Loss ({}}): {:.3e}, Final Test Loss ({}}): {:.3e}'.format(end_time - start_time, \
                    epochs, batch_size, \
                    loss_label, self.train_loss_log[-1, 0], \
                    loss_label, self.test_loss_log[-1, 0]))
        print('-'*50)
    
    def predict(self, X, X_trunk):
        with torch.no_grad():
            return self.forward(X, X_trunk)