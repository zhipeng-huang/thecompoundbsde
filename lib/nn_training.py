import torch
import torch.nn as nn
import time
from torch.optim.lr_scheduler import ExponentialLR
from lib.early_stop import EarlyStop
from lib.nn_optimizers import build_optimizer



class TrainFun:
    def __init__(self, config, loaded_model):
        self.loss_fn = CompoundLoss()
        self.dim_w = config.eqn_config.dim_w
        self.N = sum(config.eqn_config.N_list)
        self.h = float( sum(config.eqn_config.T_list) / self.N )

        self.config = config
        self.loaded_model = loaded_model
        self.batch_s = config.eqn_config.batch_s
        self.num_train_iter = config.net_config.num_train_iter
        self.log_freq = config.net_config.log_freq
        self.print_freq = config.net_config.print_freq

        # ---- learning rate settings----
        self.ini_lr = config.net_config.ini_lr
        self.exp_decay = config.net_config.is_exp_decay_lr
        self.decay_rate = config.net_config.decay_rate
        self.lr_freq = config.net_config.lr_freq
        self.lr_bound = config.net_config.lr_bound      # lower bound for lr
        if self.exp_decay == True:
            ## updated lr = lr_last_iteration * decay rate, after each scheduler.step()
            ## lr for the k-th iteration = initial lr * (decay rate^k), k = 0,1,...
            self.decay_rate = self.decay_rate
        else:
            self.decay_rate = 1.0

        # ---- Early stopping settings----
        self.early_stop_enabled = bool(config.net_config.early_stopping.status)
        self.early_stop_method = config.net_config.early_stopping.method
        self.early_stop_start_iter = int(config.net_config.early_stopping.start_iter)
        self.early_stopper = EarlyStop(stopmode = self.early_stop_method)

        # ---- optimizer and scheduler created here ----
        self.optim_name = config.net_config.optim_name.lower()
        self.optimizer = build_optimizer(self.loaded_model, self.optim_name, self.ini_lr)
        if self.optim_name == "rprop":
            self.scheduler = None
        else:
            self.scheduler = ExponentialLR(self.optimizer, gamma=self.decay_rate)


    def __call__(self, data_input, target_output):
        index_train_iter, lr_hist, loss_hist, time_hist = [], [], [], []
        start_time = time.perf_counter()
        self.loaded_model.train()       # set model in train mode

        for k in range(1, self.num_train_iter+1):
            # get output from model (need this for every update for gradient tracking)
            # torch.manual_seed(42)
            data_input = [torch.normal(0., self.h**0.5, size=[self.batch_s, self.dim_w, 1], device = self.config.net_config.device) for _ in range(self.N)]
            model_output = self.loaded_model(data_input)
            loss_value = self.loss_fn(model_out = model_output, target_out = 0.)

            if self.optim_name == "rprop":
                current_lr = float("nan")  # or label it differently
            else:
                current_lr = self.optimizer.param_groups[0]['lr']

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss_value.backward()
            self.optimizer.step()

            # update learning rate for every self.lr_freq iterations
            if self.scheduler is not None and k % self.lr_freq == 0 and current_lr >= self.lr_bound:
                self.scheduler.step()

            # log the results, including k=0 and the last iterations
            if k == 1 or k % self.log_freq == 0 or k == self.num_train_iter:
                index_train_iter.append(k)
                lr_hist.append(current_lr)
                loss_hist.append(loss_value.item())
                time_hist.append(time.perf_counter() - start_time)

            # print results
            if k == 1 or k % self.print_freq == 0 or k == self.num_train_iter:
                print(f'Current iteration k = {k}, elapsed time = {time.perf_counter()-start_time:.1f}, Loss = {loss_value.item():.10f}, LR = {current_lr:.6f}')

            # ---- Early stopping check ----
            if self.early_stop_enabled and k >= self.early_stop_start_iter:
                if k == self.early_stop_start_iter:
                    print(f'Applying early stopping from iteration {k}')
                if self.early_stopper.update(loss_value.item(), self.loaded_model, k):
                    if self.early_stop_method == "bestcheck":
                        print(f"Early Stop at iter {k}, Best iter={self.early_stopper.best_iter}, best loss={self.early_stopper.best_loss:.6g}")
                    else:
                        print(f"Early Stop at iter {k}.")
                    break

        # for checking restore
        state_before = {k: v.detach().clone() for k, v in self.loaded_model.state_dict().items()}

        # Restore best weights (only for best-check strategy)
        if self.early_stop_enabled and self.early_stop_method == "bestcheck":
            self.early_stopper.restore_best(self.loaded_model)

        # for checking restore (expected non-zero)
        diff = max((state_before[k] - self.loaded_model.state_dict()[k]).abs().max().item() for k in state_before)
        print(f"RestoreCheck, max parameters difference = {diff:.5f}")

        train_hist = [index_train_iter, lr_hist, loss_hist, time_hist]
        return train_hist


    def save_model(
            self,
            filepath,
            save_model_state=True,
            save_optimizer_state=True,
            save_scheduler_state=True,
    ):
        """
        Save checkpoint to filepath with fine-grained control.

        Args:
            filepath: path to save checkpoint.
            save_model_state: whether to save model parameters.
            save_optimizer_state: whether to save optimizer state.
            save_scheduler_state: whether to save scheduler state.
        """
        ckpt = {}
        if save_model_state:
            ckpt["model_state"] = self.loaded_model.state_dict()
        if save_optimizer_state and self.optimizer is not None:
            ckpt["optimizer_state"] = self.optimizer.state_dict()
        if save_scheduler_state and self.scheduler is not None:
            ckpt["scheduler_state"] = self.scheduler.state_dict()
        torch.save(ckpt, filepath)


    def load_model(
            self,
            filepath,
            strict=True,
            load_model_state=True,
            load_optimizer_state=True,
            load_scheduler_state=True,
    ):
        """
        Load checkpoint from filepath with fine-grained control.
        Args:
            filepath: path to checkpoint file.
            strict: passed to load_state_dict for model.
            load_model_state: whether to load model weights.
            load_optimizer_state: whether to load optimizer state.
            load_scheduler_state: whether to load scheduler state.
        """
        ckpt = torch.load(filepath, map_location="cpu")
        if load_model_state:
            state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
            self.loaded_model.load_state_dict(state, strict=strict)
        if load_optimizer_state and self.optimizer is not None and "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if load_scheduler_state and self.scheduler is not None and "scheduler_state" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        return ckpt


    def move_optimizer_state_to_device(self, device):
        """
        Call this AFTER you move the model outside the trainer, if you resume training on MPS/CUDA.
        Example:
            model.to("mps")
            Trainer.move_optimizer_state_to_device(optimizer, "mps")
        """
        optimizer = self.optimizer
        device = torch.device(device)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)




class CompoundLoss(nn.Module):
    def __call__(self, model_out, target_out):
        Y_approx, Y_target = model_out
        loss_list = []
        for i in range(len(Y_approx)):
            loss_list.append(torch.mean(torch.linalg.norm(Y_approx[i]-Y_target[i], ord='fro', dim=(-2,-1))**2))
        total_loss = sum(loss_list)
        return total_loss


class CompoundLoss_freeze(nn.Module):       # loss functional with freezing tensors
    def __call__(self, model_out, target_out):
        Y_approx, Y_target = model_out
        loss_list = []
        for i in range(len(Y_approx)):
            copy_y_target = Y_target[i].detach().clone()
            loss_list.append(torch.mean(torch.linalg.norm(Y_approx[i] - copy_y_target, ord = 'fro', dim=(-2, -1))**2 ))
        total_loss = sum(loss_list)
        return total_loss






