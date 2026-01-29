import torch.nn as nn
import torch


class ANN(nn.Module):
    def __init__(self, config, in_dim, out_dim):
        super(ANN, self).__init__()
        self.use_bias = config.net_config.is_bias  # boolean
        self.use_batchnorm = config.net_config.is_batchnorm
        self.ini_method = [config.net_config.ini_weight, config.net_config.ini_bias]    # list of strings
        self.num_hiddens = config.net_config.num_hiddens  # list
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Map the string to the corresponding activation function
        self.act_fn = self._get_act_fn(config.net_config.act_fn)
        self.out_act_fn = None

        if len(self.out_dim) == 1:
            self.out_shape = self.out_dim[0]
        else:
            self.out_shape = self.out_dim[0] * self.out_dim[1]
        self.use_batchnorm = True

        # create linear layers and batch normalization layers
        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(len(self.num_hiddens) + 1):
            if i == 0:  # from input to first hidden layer
                in_features = self.in_dim
                out_features = self.num_hiddens[i]
            elif i == len(self.num_hiddens):  # output layer
                in_features = self.num_hiddens[-1]
                out_features = self.out_shape
            else:  # between hidden layers
                in_features = self.num_hiddens[i - 1]
                out_features = self.num_hiddens[i]

            layer = nn.Linear(in_features, out_features, bias=self.use_bias)
            self._initialize_layer(layer)   # initilize weights and bias
            self.layers.append(layer)     # len(layers) = len(num_hiddens)+1

            # Add BatchNorm for non-output layers
            if self.use_batchnorm==True and i < len(self.num_hiddens):
                self.bn_layers.append(nn.BatchNorm1d(out_features))   # len(bn_layers) = 2


    def _get_act_fn(self, activation):
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "softplus": nn.Softplus(),
            "leakyrelu": nn.LeakyReLU(negative_slope=0.01, inplace=False),
            "none": lambda x: x       # Identity function = no activation
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation function: {activation}")
        return activations[activation]


    def _initialize_layer(self, layer):
        # initilize the weights
        if self.ini_method[0] == "default":  # Default initialization
            pass
        elif self.ini_method[0] == "zero":
            nn.init.zeros_(layer.weight)
        elif self.ini_method[0] == "xavier_uniform":
            nn.init.xavier_uniform_(layer.weight)
        elif self.ini_method[0] == "xavier_normal":
            nn.init.xavier_normal_(layer.weight)
        elif self.ini_method[0] == "he_uniform":
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
        elif self.ini_method[0] == "he_normal":
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        elif self.ini_method[0] == "normal":
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)  # this is an example custom initialization
        else:
            raise ValueError(f"Unsupported initialization for weights: {self.ini_method[0]}")

        # Initialize biases
        if self.use_bias and layer.bias is not None:
            if self.ini_method[1] == "default":  # Default initialization
                pass
            elif self.ini_method[1] == "constant":
                nn.init.constant_(layer.bias, 0.0)
            elif self.ini_method[1] == "normal":
                nn.init.normal_(layer.bias, mean=0.0, std=0.01)
            else:
                raise ValueError(f"Unsupported initialization for bias: {self.ini_method[1]}")


    def forward(self, x):   # x_in.shape = [B, dim_x, 1] , out.shape = [B, *, *] depends on y or z
        x = torch.squeeze(x, dim = -1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:    # for non-output layers
                if self.use_batchnorm == True:
                    x = self.bn_layers[i](x)
                x = self.act_fn(x)
            if i == len(self.layers) - 1 and self.out_act_fn is not None:
                x = self.out_act_fn(x)    # optional: output activation
        # reshape the output to fit the need
        if len(self.out_dim) == 1:
            x = torch.unsqueeze(x, dim=-1)
        else:
            x = torch.reshape(x, shape=(x.shape[0], self.out_dim[0], self.out_dim[1]))
        return x




