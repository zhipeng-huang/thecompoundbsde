import importlib
import torch
import numpy as np
from lib.nn_templates import ANN


## general fold compound option method
class DeepCBSDE(torch.nn.Module):
    def __init__(self, config):
        super(DeepCBSDE, self).__init__()
        self.eqn_name = config.eqn_config.eqn_name
        self.batch_s = config.eqn_config.batch_s
        self.dim_x = config.eqn_config.dim_x
        self.dim_y = config.eqn_config.dim_y
        self.dim_w = config.eqn_config.dim_w
        self.type_list = config.eqn_config.type_list
        self.K_list = config.eqn_config.K_list
        self.N_list = config.eqn_config.N_list
        self.T_list = config.eqn_config.T_list
        self.N = sum(self.N_list)
        self.T = sum(self.T_list)
        self.h = float(self.T / self.N)

        self.accu_N_list = [0]
        for n_values in self.N_list:
            self.accu_N_list.append(self.accu_N_list[-1] + n_values)

        # create list for the networks
        self.Z_nets = torch.nn.ModuleList()
        for i in range(0, self.N):              # not include all the terminal Zs
            self.Z_nets.append(ANN(config, in_dim=self.dim_x, out_dim=[self.dim_y, self.dim_w]))
        self.Y_nets = torch.nn.ModuleList()     # for all initial Y0
        for j in range(0, len(self.N_list)):
            self.Y_nets.append(ANN(config, in_dim=self.dim_x, out_dim=[self.dim_y, 1]))

        # load modules and creat instances
        self.fbsde = getattr(importlib.import_module("lib.fbsde"), self.eqn_name)(config)


    def __call__(self, DW):
        X = self.forward(DW)
        Y_pro, Z_pro = self.backward(X, DW)
        Y_approx, Y_target = [], []
        for j in range(0, len(self.N_list)):
            Y_approx.append(Y_pro[j][self.N_list[j]])
            if j < len(self.N_list) - 1:
                Y_target.append(self.fbsde.y_mid(X[self.accu_N_list[j+1]], Y_pro[j+1][0], Z_pro[j+1][0],
                                                 strike=self.K_list[j],opt_type=self.type_list[j]))
            else:
                Y_target.append(self.fbsde.y_termin(X[-1], strike=self.K_list[-1], opt_type=self.type_list[-1]))
        return Y_approx, Y_target


    def forward(self, DW, sample_size = None):
        sample_size = self.batch_s if sample_size is None else sample_size
        X = []
        X.append(self.fbsde.get_x0(sample_size))    # get x0 with specified sample size
        for i in range(0, self.N):
            X.append(self.fbsde.X_forward(i*self.h, X[i], None, None, DW[i]))
        return X


    def backward(self, X, DW):      # get the backward processes Y and Z
        Z_pro = []              # = a single list of M lists, to include all Z processes
        for j in range(0, len(self.N_list)):
            Z_pro.append([self.Z_nets[i + self.accu_N_list[j]](X[i + self.accu_N_list[j]]) for i in range(0, self.N_list[j])])

        Y_pro = [[] for _ in range(len(self.N_list))]     # = a single list of M lists, to include all Y processes
        for j in range(0, len(self.N_list)):
            Y_pro[j].append( self.Y_nets[j]( X[self.accu_N_list[j]]) )       # for each initial Y0
            for i in range(0, self.N_list[j]):      # iterate for each Y process starting from Y0
                Y_pro[j].append(self.fbsde.Y_forward((i+self.accu_N_list[j])*self.h, X[i+self.accu_N_list[j]],
                                                     Y_pro[j][i], Z_pro[j][i], DW[i+self.accu_N_list[j]], self.fbsde.driver))

        ### add all terminal conditions for the Z processes (placeholder, not involved in the training)
        for j in range(0, len(self.N_list)):
            if j < len(self.N_list) - 1:
                Z_pro[j].append(self.fbsde.z_mid(X[self.accu_N_list[j+1]], Y_pro[j+1][0], Z_pro[j+1][0], strike=self.K_list[j], opt_type=self.type_list[j]))
            else:
                Z_pro[j].append(self.fbsde.z_termin(X[self.N], strike=self.K_list[j], opt_type=self.type_list[j]))
        return Y_pro, Z_pro


    def getoutput(self, DW, sample_size, get_np=True):  # only for getting the output
        X = self.forward(DW, sample_size)   # a list of N tensors
        Y, Z = self.backward(X, DW)         # a list of M lists of tensors
        if not get_np:
            return X, Y, Z

        # Helper: tensor -> numpy
        def to_numpy(t):
            return t.detach().cpu().numpy()

        # Convert X, Y, Z
        X_list_np = [to_numpy(tensor) for tensor in X]
        Y_list_np, Z_list_np = [], []
        for j in range(len(self.N_list)):
            Y_list_np.append([to_numpy(tensor) for tensor in Y[j]])
            Z_list_np.append([to_numpy(tensor) for tensor in Z[j]])

        # Stack and transpose to shape [sample_size, time_steps, dim, 1]
        X_list_np = np.transpose(np.array(X_list_np), axes=[1, 0, 2, 3])

        for j in range(len(self.N_list)):
            Y_list_np[j] = np.transpose(np.array(Y_list_np[j]), axes=[1, 0, 2, 3])
            Z_list_np[j] = np.transpose(np.array(Z_list_np[j]), axes=[1, 0, 2, 3])
        return X_list_np, Y_list_np, Z_list_np


    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        p = next(self.parameters(), None)   # infer target device/dtype from any parameter
        if p is not None:
            self.fbsde.to(p.device, p.dtype)
            print(f"Move the fbsde instance to {p.device}")
        return self