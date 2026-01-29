import numpy as np
import matplotlib.pyplot as plt
import os, csv


class ErrorsCollect:
    def get_errors(self, metrics, *args):
        """
        Compute specified error metrics for multiple (x, x_true) pairs.

        Args:
            metrics (list of str): Must be a list containing "mse", "rootmse", or both.
                                   Examples: ["mse"], ["rootmse"], ["mse", "rootmse"]
            *args: Should be passed as pairs: (x_approx, x_true, x_approx2, x_true2, ...)

        Returns:
            dict: Dictionary with metric names as keys and lists of results as values.
        """
        if len(args) % 2 != 0:
            raise ValueError("Arguments must be in pairs: (x_approx, x_true)")

        # Require metrics to be a list of strings
        if not isinstance(metrics, list):
            raise TypeError("`metrics` must be a list of strings, e.g., ['mse'] or ['mse', 'rootmse']")

        metrics_list = [m.lower() for m in metrics]
        # Mapping of valid metrics to functions
        metric_funcs = {
            "mse": self.MSE,
            "rootmse": self.rootMSE,
            "remse": self.reMSE,
            "remse_re":self.reMSE_revised
        }

        # Validate metrics
        for metric in metrics_list:
            if metric not in metric_funcs:
                raise ValueError(f"Invalid metric '{metric}'. Valid options are: {list(metric_funcs.keys())}")

        # Initialize results dictionary
        results = {metric: [] for metric in metrics_list}

        # Compute errors for each pair and each metric
        for i in range(0, len(args), 2):
            x_approx, x_true = args[i], args[i + 1]
            for metric in metrics_list:
                func = metric_funcs[metric]     # get the map for the metric
                results[metric].append(func(x_approx, x_true))
        return results      # a dictionary, use it by: results.get("mse")

    def MSE(self, x_approx, x_true):   # x.shape = [paths, time steps, dim_x, 1]
        out = np.mean(np.linalg.norm(x_approx - x_true, ord='fro', axis=(-2,-1), keepdims=False)**2, axis=0)
        return out      # output shape = [time steps, ]      didn't divide the dim here

    def rootMSE(self, x_approx, x_true):
        out = np.sqrt(self.MSE(x_approx, x_true))
        return out

    def reMSE(self, x_approx, x_true):     # dived by the true sol
        out = self.MSE(x_approx, x_true) / np.mean(np.linalg.norm(x_true, ord='fro', axis=(-2,-1), keepdims=False)**2, axis=0)
        return out

    def reMSE_revised(self, x_approx, x_true): # divided by the approximation
        out = self.MSE(x_approx, x_true) / np.mean(np.linalg.norm(x_approx, ord='fro', axis=(-2,-1), keepdims=False)**2, axis=0)
        return out


    def plot_errors(self, x_axis, errors_list, errors_names, ylabel, fig_title,
                    save_fig = False, fig_name = None, fig_path = None, logscale = True, figsize = (8, 5)):
        plt.figure(figsize=figsize)
        for i in range(len(errors_list)):
            plt.plot(x_axis, errors_list[i], label=f'{errors_names[i]}')

        if logscale == True:
            plt.yscale("log")
        plt.xlabel('Time')
        plt.ylabel(f'{ylabel}')
        plt.title(f'{fig_title}')
        plt.legend()
        plt.grid(True)

        if save_fig is True and fig_path is not None:
            plt.savefig(os.path.join(fig_path, f'{fig_name}'))
            plt.close()
        if save_fig is False or fig_path is None:
            plt.show()

    def save_errors(self, errors_list, names_list, file_name, file_path):
        # file_name must have .csv as the end
        num_timestep = errors_list[0].shape[0]
        with open(f'{file_path}/{file_name}', mode='w', newline='') as file:
            writer = csv.writer(file)

            # Build the header dynamically based on the names_list
            header = ["TimeStep"]
            for i in range(len(names_list)):
                header.append(names_list[i])
            writer.writerow(header)

            # Write each row: time step + all means flattened row-wise
            for t in range(num_timestep):
                row = [t]
                for item in errors_list:
                    row.append(item[t, ])
                writer.writerow(row)




################################################
class MeansCollect:    # get the mean of all componenet of a matrix along the sample size axis
    def mean(self, x):   # x input shape = [num paths, num time steps, dim_row, dim_col]
        _, num_timestep, dim_row, dim_col = x.shape
        mean_all_comp = np.zeros(shape=(num_timestep, dim_row, dim_col))
        for i in range(dim_row):
            for j in range(dim_col):
                mean_all_comp[:, i, j] = np.mean(x[:, :, i, j], axis=0)
        return mean_all_comp      # output shape = [num time steps,  dim_row, dim_col]

    # new method: compute mean for each array in a list
    def get_means(self, xs):   # xs is a list of arrays
        return [self.mean(x) for x in xs]

    def plot_mean(self, x_axis, mean_all_comp, process_name = "E( )", save_fig = False, fig_path = None, logscale = False, figsize = (8, 5)):
        _, dim_row, dim_col = mean_all_comp.shape
        plt.figure(figsize=figsize)
        for i in range(dim_row):
            for j in range(dim_col):
                plt.plot(x_axis, mean_all_comp[:, i, j], label=f'${process_name}$ ({i+1},{j+1})')
        if logscale == True:
            plt.yscale("log")
        plt.xlabel('t')
        plt.ylabel('Mean values')
        plt.title(f'Components of ${process_name}$')
        plt.legend()
        plt.grid(True)

        if save_fig is True and fig_path is not None:
            plt.savefig(os.path.join(fig_path, f'means_{process_name}.png'))
            plt.close()
        if save_fig is False or fig_path is None:
            plt.show()

    def save_means(self, file_path, file_name, means_list, names_list):
        """
        Save means of any number of processes into one CSV.
        means_list: a list of NumPy arrays, each shape [time steps, dim_row, dim_col].
        dim_row and dim_col can differ between arrays.
        """
        if not means_list:
            raise ValueError("The means_list is empty.")

        # Check that all arrays have the same number of time steps
        time_steps = [m.shape[0] for m in means_list]
        if len(set(time_steps)) != 1:
            raise ValueError(f"All means must have the same number of time steps, got {time_steps}")

        num_timestep = time_steps[0]
        with open(f'{file_path}/{file_name}', mode='w', newline='') as file:
            writer = csv.writer(file)

            # Build the header dynamically: E1(i,j), E2(i,j), ...
            header = ["TimeStep"]
            for idx, m in enumerate(means_list, start=0):
                if len(m.shape) == 1:
                    header.append(f"{names_list[idx]}")
                if len(m.shape) == 3:
                    dim_row, dim_col = m.shape[1], m.shape[2]
                    for i in range(dim_row):
                        for j in range(dim_col):
                            header.append(f"{names_list[idx]}({i+1},{j+1})")
            writer.writerow(header)

            # Write each row: time step + all means flattened row-wise
            for t in range(num_timestep):
                row = [t]
                for m in means_list:
                    if len(m.shape) == 1:
                        row.append(m[t])
                    if len(m.shape) == 3:
                        dim_row, dim_col = m.shape[1], m.shape[2]
                        row.extend(m[t, i, j] for i in range(dim_row) for j in range(dim_col))
                writer.writerow(row)
