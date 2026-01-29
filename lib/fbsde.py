import torch
import numpy as np


############################
class CompoundBSDE(torch.nn.Module):
    def __init__(self, config, convert_np = False):
        super().__init__()
        self.convert_np = convert_np
        self.dim_x = config.eqn_config.dim_x
        self.dim_y = config.eqn_config.dim_y
        self.dim_w = config.eqn_config.dim_w
        self.type_list = config.eqn_config.type_list
        self.T_list = config.eqn_config.T_list
        self.N_list = config.eqn_config.N_list
        self.K_list = config.eqn_config.K_list
        self.T = sum(self.T_list)
        self.N = sum(self.N_list)
        self.h = float(self.T/self.N)

        self.cons_r = config.eqn_config.cons_r  # python float, no need to move device
        if isinstance(config.eqn_config.x0, list):
            self.x0_ini = torch.tensor(config.eqn_config.x0)
        elif isinstance(config.eqn_config.x0, (float, int)):
            self.x0_ini = torch.ones(self.dim_x, 1) * config.eqn_config.x0
        else:
            raise TypeError(f"Unsupported input type for config.eqn_config.x0")
        self.cons_mu = config.eqn_config.cons_r - torch.tensor(config.eqn_config.cons_q)
        self.cons_sigma = torch.tensor(config.eqn_config.cons_sigma)


    def to(self, device, dtype):
        print('move tensors of self.fbsde within model to device and dtype:', device, dtype)
        self.x0_ini = self.x0_ini.to(torch.device(device), dtype=dtype)
        self.cons_mu = self.cons_mu.to(torch.device(device), dtype=dtype)
        self.cons_sigma = self.cons_sigma.to(torch.device(device), dtype=dtype)
        return self

    def _convert_to_tensor(self, *args):
        # input can be single item, list and tuple. Always return a list
        if not self.convert_np:
            return list(args)
        tensors_list = []
        for arg in args:
            if arg is None:
                tensors_list.append(None)
            elif isinstance(arg, torch.Tensor):
                tensors_list.append(arg)
            elif isinstance(arg, np.ndarray):
                tensors_list.append(torch.from_numpy(arg))
            elif isinstance(arg, (float, int)):
                tensors_list.append(torch.tensor(arg))
            else:
                raise TypeError(f"Unsupported input type {type(arg)} in _convert_to_tensor.")
        return tensors_list


    def get_x0(self, sample_size):
        x0 = self.x0_ini.unsqueeze(0).repeat(sample_size, 1, 1)
        return x0

    def drift(self, t, x, y, z):    # general GBM drift
        x, y, z = self._convert_to_tensor(x, y, z)
        drift =  x * self.cons_mu
        return drift        # output shape=[B, dim_x,1]

    def diffusion(self, t, x, y, z):    # general GBM diffusion
        x, y, z = self._convert_to_tensor(x, y, z)
        diag_x = torch.diag_embed(torch.squeeze(x, dim = -1))
        # diffusion = diag_x * self.cons_sigma         # if sigma non-diagonal = correlated dW
        diffusion = diag_x @ self.cons_sigma        # general sigma matrix with indepedent dW
        return diffusion    # outshape = [B,dim_x,dim_w]

    def diffusion_inv(self, t, x, y, z):    # invserse of  diffusion
        x, y, z = self._convert_to_tensor(x, y, z)
        diff = self.diffusion(t, x, y, z)
        diff_inv = torch.linalg.inv(diff)
        return diff_inv    # outshape = [B,dim_x,dim_w]

    def X_forward(self, t, x, y, z, dw):
        x, y, z, dw = self._convert_to_tensor(x, y, z, dw)
        x_np1 = x + self.drift(t,x,y,z) * self.h + self.diffusion(t,x,y,z) @ dw
        return x_np1

    def driver(self, t, x, y, z):
        x, y, z = self._convert_to_tensor(x, y, z)
        driver = - (self.cons_r * y)
        return driver

    def Y_backward(self, t, x, y, z, dw, your_driver):  # your_driver argument allows you to choose different driver
        x, y, z, dw = self._convert_to_tensor(x, y, z, dw)
        y_nm1 = y + your_driver(t, x, y, z) * self.h - z @ dw
        return y_nm1

    def Y_forward(self, t, x, y, z, dw, your_driver):
        x, y, z, dw = self._convert_to_tensor(x, y, z, dw)
        y_np1 = y - your_driver(t, x, y, z) * self.h + z @ dw
        return y_np1

    def y_mid(self, x, y, z, strike, opt_type):   # compound payoff, compare common option price and strik price
        x, y, z = self._convert_to_tensor(x, y, z)
        if opt_type == "call":
            yT = torch.maximum(y - strike, torch.zeros((), device=y.device, dtype=y.dtype))    # payoff type 1 = max(y-k, 0)
            # g = torch.maximum(y, torch.tensor(self.K_out) )     # payoff type 2 = max(y, k),
        elif opt_type == "put":
            yT = torch.maximum(strike - y, torch.zeros((), device=y.device, dtype=y.dtype))
        else:
            raise ValueError("option type must be 'call' or 'put' for y_mid")
        return yT

    def y_termin(self, x, strike, opt_type):    # normal option payoff
        x, = self._convert_to_tensor(x)
        if opt_type == "call":
            yT = torch.maximum(x - strike, torch.zeros((), device=x.device, dtype=x.dtype))
        elif opt_type == "put":
            yT = torch.maximum(strike - x, torch.zeros((), device=x.device, dtype=x.dtype))
        else:
            raise ValueError("opt_type must be 'call' or 'put' for y_termin")
        return yT

    def z_mid(self, x, y, z, strike, opt_type):
        x, y, z = self._convert_to_tensor(x, y, z)
        if opt_type == "call":
            delta = (x > strike).type_as(x)
            zT = torch.permute(delta, dims=(0,2,1)) @ self.diffusion(None, x, None, None)
        elif opt_type == "put":
            delta = -(x < strike).type_as(x)
            zT = torch.permute(delta, dims=(0, 2, 1)) @ self.diffusion(None, x, None, None)
        else:
            raise ValueError("option type must be 'call' or 'put' for z_mid")
        return zT

    def z_termin(self, x, strike, opt_type):
        x, = self._convert_to_tensor(x)
        if opt_type == "call":
            delta = (x > strike).type_as(x)
            zT = torch.permute(delta, dims=(0,2,1)) @ self.diffusion(None, x, None, None)
        elif opt_type == "put":
            delta = -(x < strike).type_as(x)
            zT = torch.permute(delta, dims=(0,2,1)) @ self.diffusion(None, x, None, None)
        else:
            raise ValueError("opt_type must be 'call' or 'put' for z_termin")
        return zT




class PlainCompound(CompoundBSDE):
    pass

class NFoldCall(CompoundBSDE):
    pass


class BasketBermudan(CompoundBSDE):
    def y_mid(self, x, y, z, strike, opt_type):
        x, y, z = self._convert_to_tensor(x, y, z)
        gav_x = torch.exp(torch.mean(torch.log(x), dim=-2, keepdim=True))   # for overflow issue
        if opt_type == "call":
            payoff = torch.maximum(gav_x - strike, torch.zeros((), device=y.device, dtype=y.dtype))
            yT = torch.maximum(y, payoff)
        elif opt_type == "put":
            payoff = torch.maximum(strike - gav_x, torch.zeros((), device=y.device, dtype=y.dtype))
            yT = torch.maximum(y, payoff)
        else:
            raise ValueError("option type must be 'call' or 'put' for y_mid")
        return yT

    def y_termin(self, x, strike, opt_type):
        x, = self._convert_to_tensor(x)
        gav_x = torch.exp(torch.mean(torch.log(x), dim=-2, keepdim=True))
        if opt_type == "call":
            yT = torch.maximum(gav_x - strike, torch.zeros((), device=gav_x.device, dtype=gav_x.dtype))
        elif opt_type == "put":
            yT = torch.maximum(strike - gav_x, torch.zeros((), device=gav_x.device, dtype=gav_x.dtype))
        else:
            raise ValueError("opt_type must be 'call' or 'put' for y_termin")
        return yT

    def z_mid(self, x, y, z, strike, opt_type):     # placeholder only, no impact on approxiamtion
        x, y, z = self._convert_to_tensor(x,y, z)
        # batch_size = y.shape[0]
        if opt_type == "call":
            payoff = torch.maximum(x - strike, torch.zeros((), device=x.device, dtype=x.dtype))
            delta = (y>payoff).type_as(y) + torch.zeros(x.shape, device=x.device, dtype=x.dtype)
            zT = torch.permute(delta, dims=(0, 2, 1)) @ self.diffusion(None, x, None, None)
            # zT = torch.zeros(batch_size, self.dim_y, self.dim_w) + 0.1
        elif opt_type == "put":
            payoff = torch.maximum(strike - x, torch.zeros((), device=x.device, dtype=x.dtype))
            delta = - (y>payoff).type_as(y) + torch.zeros(x.shape, device=x.device, dtype=x.dtype)
            zT = torch.permute(delta, dims=(0, 2, 1)) @ self.diffusion(None, x, None, None)
            # zT = torch.zeros(batch_size, self.dim_y, self.dim_w) + 0.1
        else:
            raise ValueError("option type must be 'call' or 'put' for z_mid")
        return zT

    def z_termin(self, x, strike, opt_type):
        x, = self._convert_to_tensor(x)
        # batch_size = x.shape[0]
        if opt_type == "call":
            y = self.y_termin(x, strike, opt_type)
            delta = (y > 0.0).type_as(y) + torch.zeros(x.shape, device=x.device, dtype=x.dtype)
            zT = torch.permute(delta, dims=(0, 2, 1)) @ self.diffusion(None, x, None, None)
            # zT = torch.zeros(batch_size, self.dim_y, self.dim_w) + 0.1
        elif opt_type == "put":
            y = self.y_termin(x, strike, opt_type)
            delta = -(y > 0.0).type_as(y) + torch.zeros(x.shape, device=x.device, dtype=x.dtype)
            zT = torch.permute(delta, dims=(0, 2, 1)) @ self.diffusion(None, x, None, None)
            # zT = torch.zeros(batch_size, self.dim_y, self.dim_w) + 0.1
        else:
            raise ValueError("opt_type must be 'call' or 'put' for z_termin")
        return zT


class PlainBermudan(CompoundBSDE):        # usual call or put Bermudan option
    def y_mid(self, x, y, z, strike, opt_type):
        x, y, z = self._convert_to_tensor(x, y, z)
        if opt_type == "call":
            payoff = torch.maximum(x - strike, torch.zeros((), device=x.device, dtype=x.dtype))
            yT = torch.maximum(y, payoff)
        elif opt_type == "put":
            payoff = torch.maximum(strike - x, torch.zeros((), device=x.device, dtype=x.dtype))
            yT = torch.maximum(y, payoff)
        else:
            raise ValueError("option type must be 'call' or 'put' for y_mid")
        return yT

    def y_termin(self, x, strike, opt_type):
        x, = self._convert_to_tensor(x)
        if opt_type == "call":
            yT = torch.maximum(x - strike, torch.zeros((), device=x.device, dtype=x.dtype))
        elif opt_type == "put":
            yT = torch.maximum(strike - x, torch.zeros((), device=x.device, dtype=x.dtype))
        else:
            raise ValueError("opt_type must be 'call' or 'put' for y_termin")
        return yT

    def z_mid(self, x, y, z, strike, opt_type):     # placeholder, no impact on training and approxiamtion
        x, y, z = self._convert_to_tensor(x,y, z)
        batch_size = y.shape[0]
        if opt_type == "call":
            payoff = torch.maximum(x - strike, torch.zeros((), device=x.device, dtype=x.dtype))
            delta = (y>payoff).type_as(y)
        elif opt_type == "put":
            zT = torch.zeros(batch_size, self.dim_y, self.dim_w) + 0.1
        else:
            raise ValueError("option type must be 'call' or 'put' for z_mid")
        return zT

    def z_termin(self, x, strike, opt_type):    #  placeholder, no impact on training and approxiamtion
        x, = self._convert_to_tensor(x)
        # batch_size = x.shape[0]
        if opt_type == "call":
            y = self.y_termin(x, strike, opt_type)
            delta = (y > 0.0).type_as(x)
            zT = torch.permute(delta, dims=(0, 2, 1)) @ self.diffusion(None, x, None, None)
            # zT = torch.zeros(batch_size, self.dim_y, self.dim_w) + 0.1
        elif opt_type == "put":
            y = self.y_termin(x, strike, opt_type)
            delta = -(y > 0.0).type_as(x)
            zT = torch.permute(delta, dims=(0, 2, 1)) @ self.diffusion(None, x, None, None)
            # zT = torch.zeros(batch_size, self.dim_y, self.dim_w) + 0.1
        else:
            raise ValueError("opt_type must be 'call' or 'put' for z_termin")
        return zT








