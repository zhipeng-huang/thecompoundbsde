from collections import deque
import copy


class EarlyStop:
    def __init__(self, stopmode):
        self.stopmode = stopmode.lower()
        if self.stopmode == "bestcheck":
            self.patience = 100       # max number of non-improved iterations
            self.abs_tol = 0.0
            self.rel_tol = 0.05     # too small:easy to update the best loss, too big: never update the best loss
            self.tol_type = "rel"
            if self.tol_type not in {"abs", "rel", "both"}:
                raise ValueError("tol_type must be one of 'abs', 'rel', 'both'")
            self.best_loss = float("inf")  # any float will be smaller than this, for initilization
            self.best_iter = 0  # record the number of best iteration
            self.best_state = None  # record the best parameters
            self.bad_iters = 0  # counter

        elif self.stopmode == "window":
            self.window = 100  # Number of recent loss values to keep
            self.min_rel_improve = 1e-4  # Minimum required relative improvement
            self.patience = 3  # Number of consecutive "bad" windows allowed
            self.eps = 1e-12  # Small value to avoid division by zero
            self.bad_windows = 0  # Counter
            self.losses = deque(maxlen=self.window)
        else:
            raise ValueError("stopmode must be one of 'bestcheck', 'window'")


    def update_bestcheck(self, loss, model, k):
        loss = float(loss)
        # Determine improvement threshold
        if self.best_state is None:
            improved = True  # always treat first observation as improvement
        else:
            if self.tol_type == "abs":
                thresh = self.abs_tol
            elif self.tol_type == "rel":
                thresh = self.rel_tol * abs(self.best_loss)
            else:  # "both"
                thresh = max(self.abs_tol, self.rel_tol * abs(self.best_loss))
            improved = loss < self.best_loss - thresh

        # Update state
        if improved:
            self.best_loss = loss
            self.best_iter = k
            self.bad_iters = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.bad_iters += 1
        # Decide whether to stop
        return self.bad_iters >= self.patience

    def restore_best(self, model):
        """Only meaningful for mode='bestcheck'."""
        if self.stopmode != "bestcheck":
            return
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


    def update_window(self, loss):
        """
        Add the current loss value and decide whether training should stop.
        Returns:
            True  -> stop training
            False -> continue training
        """
        # Store the new loss in the sliding window
        self.losses.append(float(loss))

        # Wait until the window is full before evaluating improvement
        if len(self.losses) < self.window:
            return False

        old = self.losses[0]
        new = self.losses[-1]
        rel_improve = (old - new) / max(abs(old), self.eps)
        if rel_improve < self.min_rel_improve:
            self.bad_windows += 1
        else:
            self.bad_windows = 0    # reset, so it only countes consecutive bad windows
        # Stop if too many consecutive bad windows occurred
        return self.bad_windows >= self.patience


    def update(self, *args, **kwargs):
        method_name = f"update_{self.stopmode}"

        if not hasattr(self, method_name):
            raise ValueError(f"Early stop mode '{self.stopmode}' is not supported")

        method = getattr(self, method_name)

        if not callable(method):
            raise ValueError(f"Attribute '{method_name}' exists but is not callable")

        return method(*args, **kwargs)



#
# class EarlyStopBestCheck:
#     def __init__(self, patience=100, abs_tol=0.0, rel_tol=0.0, tol_type="rel"):
#         """
#         Args:
#             patience : int, number of non-improving iterations consecutive allowed
#             abs_tol  : float, absolute tolerance
#             rel_tol  : float, relative tolerance, percentage amount less than the best loss
#             tol_type : "abs", "rel", or "both"
#         """
#         self.patience = patience
#         self.abs_tol = abs_tol
#         self.rel_tol = rel_tol
#         self.tol_type = tol_type.lower()
#         if self.tol_type not in {"abs", "rel", "both"}:
#             raise ValueError("tol_type must be one of 'abs', 'rel', 'both'")
#
#         self.best_loss = float("inf")   # any float will be smaller than this, for initilization
#         self.best_iter = 0      # record the number of best iteration
#         self.best_state = None  # record the best parameters
#         self.bad_iters = 0      # counter
#
#     def _is_improvement(self, loss):
#         if self.best_state is None:
#             return True     # for the 1st check
#         if self.tol_type == "abs":
#             thresh = self.abs_tol
#         elif self.tol_type == "rel":
#             thresh = self.rel_tol * abs(self.best_loss)
#         else:  # "both"
#             thresh = max(self.abs_tol, self.rel_tol * abs(self.best_loss))
#         return loss < self.best_loss - thresh
#
#     def update(self, loss, model, k):
#         loss = float(loss)
#         if self._is_improvement(loss):
#             self.best_loss = loss
#             self.best_iter = k
#             self.bad_iters = 0
#             self.best_state = copy.deepcopy(model.state_dict())
#         else:
#             self.bad_iters += 1
#         return self.bad_iters >= self.patience
#
#     def restore_best(self, model):
#         if self.best_state is not None:
#             model.load_state_dict(self.best_state)
#
#
# class WindowedEarlyStop:
#     def __init__(self, window=200, min_rel_improve=1e-4, patience=3, eps=1e-12):
#         self.window = window        # Number of recent loss values to keep
#         self.min_rel_improve = min_rel_improve  # Minimum required relative improvement
#         self.patience = patience    # Number of consecutive "bad" windows allowed
#         self.eps = eps              # Small value to avoid division by zero
#         self.losses = deque(maxlen=window)
#         self.bad_windows = 0    # Counter
#
#     def update(self, loss):
#         """
#         Add the current loss value and decide whether training should stop.
#         Returns:
#             True  -> stop training
#             False -> continue training
#         """
#         # Store the new loss in the sliding window
#         self.losses.append(float(loss))
#
#         # Wait until the window is full before evaluating improvement
#         if len(self.losses) < self.window:
#             return False
#
#         old = self.losses[0]
#         new = self.losses[-1]
#         rel_improve = (old - new) / max(abs(old), self.eps)
#         if rel_improve < self.min_rel_improve:
#             self.bad_windows += 1
#         else:
#             self.bad_windows = 0    # reset, so it only countes consecutive bad windows
#
#         # Stop if too many consecutive bad windows occurred
#         return self.bad_windows >= self.patience
