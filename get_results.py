import importlib
import numpy as np
import torch, csv
from lib.helpers import load_json
from lib.get_errors_means import ErrorsCollect, MeansCollect


########################  load the model for results
folder_path = f"logs/ex2/"      # change this to the folder you want to load
save_folder_path = folder_path
config = load_json(folder_path, "exp_config.json")      # reload the config
chosen_model = torch.load(f'{folder_path}/full_model.pth')       # reload the model
errorscollect = ErrorsCollect()
meanscollect = MeansCollect()
fbsde = getattr(importlib.import_module("lib.fbsde"), config.eqn_config.eqn_name)(config, convert_np = True)


########################  load the configurations from json file
validate_s = config.eqn_config.validate_s
type_list = config.eqn_config.type_list
K_list = config.eqn_config.K_list
N_list = config.eqn_config.N_list
T_list = config.eqn_config.T_list
M = len(N_list)
N = sum(config.eqn_config.N_list)
T = sum(config.eqn_config.T_list)
dim_x = config.eqn_config.dim_x
dim_y = config.eqn_config.dim_y
dim_w = config.eqn_config.dim_w
t_axis = np.linspace(0, T, N + 1, endpoint=True)
h = T / N
opt_type = config.eqn_config.type_list[-1]
accu_N_list = [0]
for n_values in N_list:
    accu_N_list.append(accu_N_list[-1] + n_values)


# #########################  get reference solution
get_refsol = getattr(importlib.import_module("lib.ref_sol"), config.eqn_config.eqn_name)(config)
ref_X, ref_Y, ref_Z, ref_DW, ref_delta = get_refsol(sample_size=config.eqn_config.validate_s)

# output shape [sample size, time step, dim, dim]
print("ref_X at t0 is", ref_X[0, 0, :, :])
print("ref_Y at t0 is", ref_Y[0][0, 0, :, :])
print("ref_Z at t0 is", ref_Z[0][0, 0, :, :])

if config.eqn_config.eqn_name == "BasketBermudan":
    print("ref_delta at t0 is", ref_delta[0][0, 0, :, :] / dim_x)
else:
    print("ref_delta at t0 is", ref_delta[0][0, 0, :, :])



# #########################  get approx solution
DW = [torch.from_numpy(arr).to(torch.float32) for arr in ref_DW]  # if compare with ref sol
# DW = [torch.normal(0., (T/N)**0.5, size=[validate_s, dim_w, 1]) for _ in range(N)]    # if not compare with ref sol
X, Y, Z = chosen_model.getoutput(DW, sample_size=validate_s)  # X output shape = [sample size, time steps, dim_x, 1]
delta = []
for i in range(M):
    delta.append(np.zeros((validate_s, N_list[i] + 1, dim_x, 1)))
    for j in range(N_list[i] + 1):
        index = sum(N_list[:i]) + j
        delta[i][:, j, :, :] = np.transpose(
            (Z[i][:, j, :, :] @ fbsde.diffusion_inv(None, X[:, index, :, :], None, None).numpy()), axes=[0, 2, 1])

print("Approximated Y at t0 is:", Y[0][0, 0, :, :])
print("Approximated Z at t0 is:", Z[0][0, 0, :, :])
print("Approximated delta at t0 is:", delta[0][0, 0, :, :])



# #########################  get csv files for error estimate (all terms of loss function, and T/N)
estimate = []
for i in range(M):
    if i < M - 1:
        loss = fbsde.y_mid(X[:,accu_N_list[i+1],:,:], Y[i+1][:,0,:,:], None, K_list[i], opt_type=type_list[i]).numpy() - Y[i][:,-1,:,:]
    else:
        loss = fbsde.y_termin(X[:,accu_N_list[i+1],:,:],K_list[i],opt_type=type_list[i]).numpy() - Y[i][:,-1,:,:]
    loss = np.mean(np.linalg.norm(loss, ord="fro", axis=(-2, -1)) ** 2, axis=0)
    estimate.append(loss)
estimate.append(h)

with open(f'{save_folder_path}/err_estimate.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    headers = [f"Loss {i + 1}" for i in range(len(estimate) - 1)] + ["Step size"]
    writer.writerow(headers)
    writer.writerow([float(x) for x in estimate])



######## to the get means of X, Y and Z
Y_pairs = [item for pair in zip(Y, ref_Y) for item in pair]
Z_pairs = [item for pair in zip(Z, ref_Z) for item in pair]


############################### get all the means
raw_means_list = meanscollect.get_means([X, ref_X, *Y_pairs, *Z_pairs])
means_list = [None] * int(2*(dim_x + dim_y*M + dim_y*dim_w*M))

for index in range(0, dim_x):   # for X and ref_X
    means_list[2*index] = raw_means_list[0][:, index, 0]
    means_list[2*index + 1] = raw_means_list[1][:, index, 0]

for i in range(M):  # for Y and ref_Y
    before_zeros = np.zeros(sum(N_list[:i]))
    after_zeros = np.zeros(sum(N_list[i + 1:]))
    for index in range(dim_y):
        means_list[2*dim_x + (i*2*dim_y) + 2*index] = np.r_[before_zeros, raw_means_list[2+ 2*i][:, index, 0], after_zeros]
        means_list[2*dim_x + (i*2*dim_y) + 2*index + 1] = np.r_[before_zeros, raw_means_list[2+ 2*i+1][:, index, 0], after_zeros]

for i in range(M):  # # for Z and ref_Z
    before_zeros = np.zeros(sum(N_list[:i]))
    after_zeros = np.zeros(sum(N_list[i + 1:]))
    for row_i in range(dim_y):
        for col_j in range(dim_w):
            index = row_i * dim_w + col_j
            means_list[2 * (dim_x + dim_y * M) + (i * 2 * dim_y * dim_w) + 2 * index] = \
                (np.r_)[before_zeros, raw_means_list[2 + 2 * M + 2 * i][:, row_i, col_j], after_zeros]
            means_list[2 * (dim_x + dim_y * M) + (i * 2 * dim_y * dim_w) + 2 * index + 1] \
                = np.r_[before_zeros, raw_means_list[2 + 2 * M + 2 * i + 1][:, row_i, col_j], after_zeros]

means_names = ([name for i in range(dim_x) for name in (f"EX_C{i+1}_j1", f"ref_EX_C{i+1}_j1")]
               + [name for j in range(M) for i in range(dim_y) for name in (f"EY_C{i+1}_j{j+1}", f"ref_EY_C{i+1}_j{j+1}")]
               + [name for j in range(M) for i in range(dim_w) for name in (f"EZ_C{i+1}_j{j+1}", f"ref_EZ_C{i+1}_j{j+1}")] )

meanscollect.save_means(file_path=save_folder_path, file_name="mean_table.csv", means_list=means_list,
                        names_list=means_names)



################# Erros：MSE
errors_dict = errorscollect.get_errors(["mse"], X, ref_X, *Y_pairs, *Z_pairs)
mse_names = ["MSE_X_j1"] + [f"MSE_Y_j{i + 1}" for i in range(M)] + [f"MSE_Z_j{i + 1}" for i in range(M)]
mse_list = errors_dict.get("mse")
# fill the non-defined region with zeros
for i in range(M):  # for mse of all Y
    before_zeros = np.zeros(sum(N_list[:i]))
    after_zeros = np.zeros(sum(N_list[i + 1:]))
    mse_list[1 + i] = np.r_[before_zeros, mse_list[1 + i], after_zeros]

for i in range(M):  # for mse of all Z
    before_zeros = np.zeros(sum(N_list[:i]))
    after_zeros = np.zeros(sum(N_list[i + 1:]))
    mse_list[1 + M + i] = np.r_[before_zeros, mse_list[1 + M + i], after_zeros]

errorscollect.plot_errors(t_axis, mse_list, mse_names, ylabel="MSE",
                          fig_title=f"MSE, {N} steps", save_fig=True, fig_name="mse_fig.png", fig_path=save_folder_path)
errorscollect.save_errors(mse_list, mse_names, file_name="mse_table.csv", file_path=save_folder_path)



################# Erros：relative MSE
errors_dict = errorscollect.get_errors(["remse"], X, ref_X, *Y_pairs, *Z_pairs)
remse_names = ["reMSE_X_j1"] + [f"reMSE_Y_j{i + 1}" for i in range(M)] + [f"reMSE_Z_j{i + 1}" for i in range(M)]
remse_list = errors_dict.get("remse")
for i in range(M):
    before_zeros = np.zeros(sum(N_list[:i]))
    after_zeros = np.zeros(sum(N_list[i + 1:]))
    remse_list[1 + i] = np.r_[before_zeros, remse_list[1 + i], after_zeros]
for i in range(M):
    before_zeros = np.zeros(sum(N_list[:i]))
    after_zeros = np.zeros(sum(N_list[i + 1:]))
    remse_list[1 + M + i] = np.r_[before_zeros, remse_list[1 + M + i], after_zeros]

errorscollect.plot_errors(t_axis, remse_list, remse_names, ylabel="reMSE",
                          fig_title=f"reMSE, {N} steps", save_fig=True, fig_name="remse_fig.png",
                          fig_path=save_folder_path)
errorscollect.save_errors(remse_list, remse_names, file_name="remse_table.csv", file_path=save_folder_path)


