import munch, os, json, csv, importlib
import torch
import numpy as np
from absl import app
from absl import flags
from lib.nn_solver import DeepCBSDE
from lib.nn_training import TrainFun


# Define flags and make flags required
FLAGS = flags.FLAGS
flags.DEFINE_string("config_file", None, "Path to the JSON configuration file.")
flags.DEFINE_string("exp_name", None, "Experiment name for this run.")
flags.mark_flag_as_required("config_file")
flags.mark_flag_as_required("exp_name")


def main(argv):
    try:    # load json file and apply munchify on it to enable dot notation
        with open(FLAGS.config_file, 'r') as json_file:
            config = json.load(json_file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{FLAGS.config_file}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to parse JSON in '{FLAGS.config_file}'.")
        return
    config = munch.munchify(config)

    # Define paths that used to save results
    save_folder_path = os.path.join("logs", f"{FLAGS.exp_name}")
    os.makedirs(save_folder_path, exist_ok=True)  # Create the folder if it doesn't exist


    ##################### main part #####################
    print(f"Running the Compound BSDE Method on {config.net_config.device}: {FLAGS.exp_name}")
    dtype_map = {
        "torch.float32": torch.float32,
        "torch.float64": torch.float64,
    }
    torch.set_default_dtype(dtype_map[config.net_config.torchdtype])



    # =====================  network training ===========================
    chosen_model = DeepCBSDE(config)
    chosen_model.to(config.net_config.device)
    trainer = TrainFun(config, chosen_model)
    train_hist = trainer(data_input=None, target_output=None)
    torch.set_default_device("cpu")     # set back to cpu for the rest steps
    chosen_model.to("cpu")
    print("training finished")



    # =====================  get the approxiamted solution from network ===========================
    validate_s = config.eqn_config.validate_s
    dim_x = config.eqn_config.dim_x
    dim_w = config.eqn_config.dim_w
    N = sum(config.eqn_config.N_list)
    T = sum(config.eqn_config.T_list)
    N_list = config.eqn_config.N_list
    M = len(N_list)

    fbsde = getattr(importlib.import_module("lib.fbsde"), config.eqn_config.eqn_name)(config, convert_np=True)

    DW = [torch.normal(0., (T/N)**0.5, size=[validate_s, dim_w, 1]) for _ in range(N)]    # if not use ref solution
    X, Y, Z = chosen_model.getoutput(DW, sample_size = validate_s)    # X output shape = [sample size, time steps, dim_x, 1]

    delta = [ ]
    for i in range(M):
        delta.append(np.zeros((validate_s, N_list[i]+1, dim_x, 1)))
        for j in range(N_list[i]+1):
            index = sum(N_list[:i]) + j
            delta[i][:,j,:,:] = np.transpose((Z[i][:,j,:,:] @ fbsde.diffusion_inv(None,X[:,index,:,:],None,None).numpy()), axes=[0,2,1])

    print("Approximated Y at t0 is:", Y[0][0, 0, :, :])
    print("Approximated Z at t0 is:", Z[0][0, 0, :, :])
    print("Approximated delta at t0 is:", delta[0][0, 0, :, :])



    # # =====================  error computation ===========================
    # Y_pairs = [item for pair in zip(Y, ref_Y) for item in pair]
    # Z_pairs = [item for pair in zip(Z, ref_Z) for item in pair]
    # delta_pairs = [item for pair in zip(delta, ref_delta) for item in pair]
    #
    # errors_dict = errorscollect.get_errors(["mse"], X, ref_X, *Y_pairs, *Z_pairs, *delta_pairs)
    # mse_names = ["MSE_X"] + [f"MSE_Y{i + 1}" for i in range(M)] + [f"MSE_Z{i + 1}" for i in range(M)] + [f"MSE_delta{i + 1}" for i in range(M)]
    # mse_list = errors_dict.get("mse")
    #
    # for i in range(M):  # for mse of all Y
    #     before_zeros = np.zeros( sum(N_list[:i]) )
    #     after_zeros = np.zeros( sum(N_list[i + 1:]) )
    #     mse_list[1 + i] = np.r_[before_zeros, mse_list[1 + i], after_zeros]
    #
    # for i in range(M):  # for mse of all Z
    #     before_zeros = np.zeros( sum(N_list[:i]) )
    #     after_zeros = np.zeros( sum(N_list[i + 1:]) )
    #     mse_list[1 + M + i] = np.r_[before_zeros, mse_list[1 + M + i], after_zeros]
    #
    # for i in range(M):  # for mse of all delta
    #     before_zeros = np.zeros( sum(N_list[:i]) )
    #     after_zeros = np.zeros( sum(N_list[i + 1:]) )
    #     mse_list[1 + M + M + i] = np.r_[before_zeros, mse_list[1 + M + M + i], after_zeros]
    #
    # errorscollect.plot_errors(t_axis, mse_list, mse_names, ylabel = "MSE",
    #                          fig_title=f"MSE, {N} steps", save_fig = True, fig_name= "mse_fig.png", fig_path = save_folder_path)
    # errorscollect.save_errors(mse_list, mse_names, file_name = "mse_table.csv", file_path = save_folder_path)
    #
    #
    # ## for relative MSE
    # errors_dict = errorscollect.get_errors(["remse"], X, ref_X, *Y_pairs, *Z_pairs, *delta_pairs)
    # remse_names = (["reMSE_X"] + [f"reMSE_Y{i + 1}" for i in range(M)] + [f"reMSE_Z{i + 1}" for i in range(M)]
    #                + [f"reMSE_delta{i + 1}" for i in range(M)])
    # remse_list = errors_dict.get("remse")
    # for i in range(M):
    #     before_zeros = np.zeros( sum(N_list[:i]) )
    #     after_zeros = np.zeros( sum(N_list[i + 1:]) )
    #     remse_list[1 + i] = np.r_[before_zeros, remse_list[1 + i], after_zeros]
    # for i in range(M):
    #     before_zeros = np.zeros( sum(N_list[:i]) )
    #     after_zeros = np.zeros( sum(N_list[i + 1:]) )
    #     remse_list[1 + M + i] = np.r_[before_zeros, remse_list[1 + M + i], after_zeros]
    # for i in range(M):
    #     before_zeros = np.zeros( sum(N_list[:i]) )
    #     after_zeros = np.zeros( sum(N_list[i + 1:]) )
    #     remse_list[1 + M + M + i] = np.r_[before_zeros, remse_list[1 + M + M + i], after_zeros]
    # errorscollect.plot_errors(t_axis, remse_list, remse_names, ylabel="reMSE",
    #                          fig_title=f"reMSE, {N} steps", save_fig=True, fig_name= "remse_fig.png", fig_path=save_folder_path)
    # errorscollect.save_errors(remse_list, remse_names, file_name = "remse_table.csv", file_path = save_folder_path)
    #
    #
    #
    # # =====================  get all means of X,Y,Z ===========================
    # means_list = meanscollect.get_means([X, ref_X, *Y_pairs, *Z_pairs, *delta_pairs])
    #
    # # the following are for one-dimensional X,Y,Z.
    # # #To consider higher dimension, change slicing [:,0,0] to [:,i,j] and add more loops
    # means_list[0] = means_list[0][:, 0, 0]      # for X
    # means_list[1] = means_list[1][:, 0, 0]      # for ref_X
    # for i in range(M):      # for Y and ref_Y
    #     before_zeros = np.zeros(sum(N_list[:i]))
    #     after_zeros = np.zeros(sum(N_list[i + 1:]))
    #     means_list[2 + 2*i] = np.r_[before_zeros, means_list[2 + 2*i][:,0,0], after_zeros]
    #     means_list[2 + 2*i+1] = np.r_[before_zeros, means_list[2 + 2*i+1][:,0,0], after_zeros]
    # for i in range(M):      # # for Z and ref_Z
    #     before_zeros = np.zeros(sum(N_list[:i]))
    #     after_zeros = np.zeros(sum(N_list[i + 1:]))
    #     means_list[2+2*M + 2*i] = np.r_[before_zeros, means_list[2+2*M + 2*i][:,0,0], after_zeros]
    #     means_list[2+2*M + 2*i+1] = np.r_[before_zeros, means_list[2+2*M + 2*i+1][:,0,0], after_zeros]
    #
    # for i in range(M):      # # for delta
    #     before_zeros = np.zeros(sum(N_list[:i]))
    #     after_zeros = np.zeros(sum(N_list[i + 1:]))
    #     means_list[2+2*M + 2*M + 2*i] = np.r_[before_zeros, means_list[2+2*M + 2*M + 2*i][:,0,0], after_zeros]
    #     means_list[2+2*M + 2*M + 2*i+1] = np.r_[before_zeros, means_list[2+2*M + 2*M + 2*i+1][:,0,0], after_zeros]
    #
    #
    # means_names = (
    #         ["EX", "ref_EX"]
    #         + [name for i in range(M) for name in (f"EY{i + 1}", f"ref_EY{i + 1}")]
    #         + [name for i in range(M) for name in (f"EZ{i + 1}", f"ref_EZ{i + 1}")]
    #         + [name for i in range(M) for name in (f"Edelta{i + 1}", f"ref_Edelta{i + 1}")]
    # )
    #
    # meanscollect.save_means(file_path = save_folder_path, means_list= means_list,
    #                         names_list= means_names)
    #
    #
    #
    # # get csv files for error estimate (all terms of loss function, and T/N)
    # estimate = []
    # for i in range(M):
    #     if i < M-1:
    #         loss = fbsde.y_mid(None, Y[i+1][:,0,:,:], None, K_list[i], opt_type=type_list[i]).numpy() - Y[i][:, -1, :, :]
    #     else:
    #         loss = fbsde.y_termin(X[:,-1,:,:], K_list[i], opt_type=type_list[i]).numpy() - Y[i][:,-1,:,:]
    #     loss = np.mean(np.linalg.norm(loss, ord="fro", axis=(-2, -1))**2, axis=0)   # squared l2 nomr of the difference
    #     estimate.append(loss)
    #
    # estimate.append(T/N)
    #
    # with open(f'{save_folder_path}/err_estimate.csv', mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     headers = [f"Loss {i + 1}" for i in range(len(estimate) - 1)] + ["Step size"]
    #     writer.writerow(headers)
    #     writer.writerow([float(x) for x in estimate])
    #
    #


    # =====================  save the models and training result ===========================
    torch.save(chosen_model, f'{save_folder_path}/full_model.pth')     # save the model
    trainer.save_model(f'{save_folder_path}/model_dict.pt')

    with open(f'{save_folder_path}/exp_config.json', 'w') as json_file:
        json.dump(config, json_file, indent=2)

    with open(f'{save_folder_path}/train_hist.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Iter_Num", "LR", "Loss", "Elapsed Time"])
        for a, b, c, d in zip(train_hist[0], train_hist[1], train_hist[2], train_hist[3]):
            writer.writerow([a, b, c, d])

    print(f"Experiment {FLAGS.exp_name} completed successfully.")
    ##################### end of main part #####################



if __name__ == "__main__":
    app.run(main)

