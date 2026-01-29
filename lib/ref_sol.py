import importlib
import numpy as np
import torch
from scipy.stats import norm, multivariate_normal
from scipy.optimize import brentq


# input and output should be numpy array, and the sol is written using np
# when use fbsde class the input numpy array will be converted to tensor automatically and return tensors
# so you need to convert the output back to np


# all 4 type of comppund options wiht one intermediate time
class PlainCompound:
    def __init__(self, config):
        self.config = config
        self.eqn_name = config.eqn_config.eqn_name
        self.batch_s = config.eqn_config.batch_s
        self.N_list = config.eqn_config.N_list
        self.dim_y = config.eqn_config.dim_y
        self.dim_x = config.eqn_config.dim_x
        self.dim_w = config.eqn_config.dim_w

        self.compound_type = config.eqn_config.type_list[0]
        self.vanilla_type = config.eqn_config.type_list[1]
        self.N1 = config.eqn_config.N_list[0]
        self.N2 = config.eqn_config.N_list[1]
        self.T1 = config.eqn_config.T_list[0]
        self.T2 = config.eqn_config.T_list[1]
        self.K1 = config.eqn_config.K_list[0]
        self.K2 = config.eqn_config.K_list[1]
        self.T = self.T1 + self.T2
        self.N = self.N1 + self.N2
        self.ref_N = config.eqn_config.ref_N
        self.ref_h = self.T / self.ref_N

        #
        self.cons_r = config.eqn_config.cons_r
        self.cons_q = np.array(config.eqn_config.cons_q).item()
        self.cons_sigma = np.array(config.eqn_config.cons_sigma).item()

        self.fbsde = getattr(importlib.import_module("lib.fbsde"), self.eqn_name)(self.config, convert_np = True)
        self.fbsde.to(device="cpu", dtype=torch.float64)

    def bs_formula(self, S, remain_t, K, r, q, sigma, opt_type):
        S = np.asarray(S, dtype=float)

        if remain_t <= 0:  # remain time = 0 then get immediate payoff
            if opt_type == 'call':
                price = np.maximum(S - K, 0.0)
                delta = np.where(S > K, 1.0, 0.0)  # derivative of payoff
            elif opt_type == 'put':
                price = np.maximum(K - S, 0.0)
                delta = np.where(S < K, -1.0, 0.0)
            else:
                raise ValueError("option type must be 'call' or 'put'")
            return price, delta

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * remain_t) / (sigma * np.sqrt(remain_t))
        d2 = d1 - sigma * np.sqrt(remain_t)
        if opt_type == 'call':
            price = S * np.exp(-q * remain_t) * norm.cdf(d1) - K * np.exp(-r * remain_t) * norm.cdf(d2)
            delta = np.exp(-q * remain_t) * norm.cdf(d1)
        elif opt_type == 'put':
            price = K * np.exp(-r * remain_t) * norm.cdf(-d2) - S * np.exp(-q * remain_t) * norm.cdf(-d1)
            delta = -np.exp(-q * remain_t) * norm.cdf(-d1)
        else:
            raise ValueError("option type must be 'call' or 'put'")
        return price, delta

    def compound_option(self, S, tau, T2, K1, K2, r, q, sigma, opt_type1, opt_type2):
        S = np.asarray(S, dtype=float)  # allow vectorized S
        tau_T = tau + T2

        #### if tau = 0, then only cosider the second period
        if tau <= 0.0:
            opt_price, opt_delta = self.bs_formula(S, remain_t = T2, K=K2, r=r, q=q, sigma=sigma, opt_type=opt_type2)
            if opt_type1 == 'call':
                price = np.maximum(opt_price - K1, 0.0)
                delta = (np.where(opt_price - K1 > 0.0, 1.0, 0.0)) * opt_delta
            elif opt_type1 == 'put':
                price = np.maximum(K1 - opt_price, 0.0)
                delta = (np.where(K1 - opt_price > 0.0, -1.0, 0.0)) * opt_delta
            else:
                raise ValueError("option type must be 'call' or 'put'")
            return price, delta

        # if tau > 0.0, then plain compound with total life time tau_T = tau + T2
        # Step 1: compute the threshold S*
        if opt_type1 == "call":
            func = lambda S_star: self.bs_formula(S_star, remain_t=T2, K=K2, r=r, q=q, sigma=sigma, opt_type=opt_type2)[0] - K1
        elif opt_type1 == "put":
            func = lambda S_star: K1 - self.bs_formula(S_star, remain_t=T2, K=K2, r=r, q=q, sigma=sigma, opt_type=opt_type2)[0]
        else:
            raise ValueError("option type must be 'call' or 'put' for defining the func")
        S_star = brentq(func, 1e-8, 1e8)

        # Step 2: Parameters (all are vectorized for array S)
        a1 = (np.log(S / S_star) + (r - q + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
        a2 = a1 - sigma * np.sqrt(tau)
        b1 = (np.log(S / K2) + (r - q + 0.5 * sigma ** 2) * tau_T) / (sigma * np.sqrt(tau_T))
        b2 = b1 - sigma * np.sqrt(tau_T)
        mean = [0, 0]

        # Step 3: Price
        if opt_type1 == "call" and opt_type2 == "call":
            rho = np.sqrt(tau / tau_T)
            cov = [[1, rho], [rho, 1]]
            N2_a1b1 = np.array([multivariate_normal.cdf([x, y], mean=mean, cov=cov) for x, y in zip(a1, b1)])
            N2_a2b2 = np.array([multivariate_normal.cdf([x, y], mean=mean, cov=cov) for x, y in zip(a2, b2)])
            price = (S * np.exp(-q * tau_T) * N2_a1b1
                     - K2 * np.exp(-r * tau_T) * N2_a2b2
                     - K1 * np.exp(-r * tau) * norm.cdf(a2))

        elif opt_type1 == "call" and opt_type2 == "put":
            rho = np.sqrt(tau / tau_T)
            cov = [[1, rho], [rho, 1]]
            N2_na2nb2 = np.array([multivariate_normal.cdf([x, y], mean=mean, cov=cov) for x, y in zip(-a2, -b2)])
            N2_na1nb1 = np.array([multivariate_normal.cdf([x, y], mean=mean, cov=cov) for x, y in zip(-a1, -b1)])
            price = (K2 * np.exp(-r * tau_T) * N2_na2nb2
                     - S * np.exp(-q * tau_T) * N2_na1nb1
                     - K1 * np.exp(-r * tau) * norm.cdf(-a2))

        elif opt_type1 == "put" and opt_type2 == "call":
            rho = - np.sqrt(tau / tau_T)
            cov = [[1, rho], [rho, 1]]
            N2_na2b2_n = np.array([multivariate_normal.cdf([x, y], mean=mean, cov=cov) for x, y in zip(-a2, b2)])
            N2_na1b1_n = np.array([multivariate_normal.cdf([x, y], mean=mean, cov=cov) for x, y in zip(-a1, b1)])
            price = (K2 * np.exp(-r * tau_T) * N2_na2b2_n
                     - S * np.exp(-q * tau_T) * N2_na1b1_n
                     + K1 * np.exp(-r * tau) * norm.cdf(-a2))

        elif opt_type1 == "put" and opt_type2 == "put":
            rho = - np.sqrt(tau / tau_T)
            cov = [[1, rho], [rho, 1]]
            N2_a2nb2_n = np.array([multivariate_normal.cdf([x, y], mean=mean, cov=cov) for x, y in zip(a2, -b2)])
            N2_a1nb1_n = np.array([multivariate_normal.cdf([x, y], mean=mean, cov=cov) for x, y in zip(a1, -b1)])
            price = (S * np.exp(-q * tau_T) * N2_a1nb1_n
                     - K2 * np.exp(-r * tau_T) * N2_a2nb2_n
                     + K1 * np.exp(-r * tau) * norm.cdf(a2))
        else:
            raise ValueError("option type must be 'call' or 'put' in computing the price")

        # Step 4: Delta (vectorized)
        da1_dS = 1 / (S * sigma * np.sqrt(tau))
        da2_dS = da1_dS
        db1_dS = 1 / (S * sigma * np.sqrt(tau_T))
        db2_dS = db1_dS

        if opt_type1 == "call" and opt_type2 == "call":
            dN2_a1b1_dS = norm.pdf(a1) * norm.cdf((b1 - rho * a1) / np.sqrt(1 - rho ** 2)) * da1_dS \
                          + norm.pdf(b1) * norm.cdf((a1 - rho * b1) / np.sqrt(1 - rho ** 2)) * db1_dS
            dN2_a2b2_dS = norm.pdf(a2) * norm.cdf((b2 - rho * a2) / np.sqrt(1 - rho ** 2)) * da2_dS \
                          + norm.pdf(b2) * norm.cdf((a2 - rho * b2) / np.sqrt(1 - rho ** 2)) * db2_dS
            delta = (np.exp(-q * tau_T) * (N2_a1b1 + S * dN2_a1b1_dS)
                     - K2 * np.exp(-r * tau_T) * dN2_a2b2_dS
                     - K1 * np.exp(-r * tau) * norm.pdf(a2) * da2_dS)

        elif opt_type1 == "call" and opt_type2 == "put":
            dN2_na2nb2_dS = norm.pdf(-a2) * norm.cdf((-b2 + rho * a2) / np.sqrt(1 - rho ** 2)) * (-da2_dS) \
                            + norm.pdf(-b2) * norm.cdf((-a2 + rho * b2) / np.sqrt(1 - rho ** 2)) * (-db2_dS)
            dN2_na1nb1_dS = norm.pdf(-a1) * norm.cdf((-b1 + rho * a1) / np.sqrt(1 - rho ** 2)) * (-da1_dS) \
                            + norm.pdf(-b1) * norm.cdf((-a1 + rho * b1) / np.sqrt(1 - rho ** 2)) * (-db1_dS)
            delta = (K2 * np.exp(-r * tau_T) * dN2_na2nb2_dS
                     - np.exp(-q * tau_T) * (N2_na1nb1 + S * dN2_na1nb1_dS)
                     - K1 * np.exp(-r * tau) * norm.pdf(-a2) * (-da2_dS))

        elif opt_type1 == "put" and opt_type2 == "call":
            dN2_na2b2_n_dS = norm.pdf(-a2) * norm.cdf((b2 + rho * a2) / np.sqrt(1 - rho ** 2)) * (-da2_dS) \
                             + norm.pdf(b2) * norm.cdf((-a2 - rho * b2) / np.sqrt(1 - rho ** 2)) * db2_dS
            dN2_na1b1_n_dS = norm.pdf(-a1) * norm.cdf((b1 + rho * a1) / np.sqrt(1 - rho ** 2)) * (-da1_dS) \
                             + norm.pdf(b1) * norm.cdf((-a1 - rho * b1) / np.sqrt(1 - rho ** 2)) * db1_dS
            delta = (K2 * np.exp(-r * tau_T) * dN2_na2b2_n_dS
                     - np.exp(-q * tau_T) * (N2_na1b1_n + S * dN2_na1b1_n_dS)
                     + K1 * np.exp(-r * tau) * norm.pdf(-a2) * (-da2_dS))

        elif opt_type1 == "put" and opt_type2 == "put":
            dN2_a1nb1_n_dS = norm.pdf(a1) * norm.cdf((-b1 - rho * a1) / np.sqrt(1 - rho ** 2)) * da1_dS \
                             + norm.pdf(-b1) * norm.cdf((a1 + rho * b1) / np.sqrt(1 - rho ** 2)) * (-db1_dS)
            dN2_a2nb2_n_dS = norm.pdf(a2) * norm.cdf((-b2 - rho * a2) / np.sqrt(1 - rho ** 2)) * da2_dS \
                             + norm.pdf(-b2) * norm.cdf((a2 + rho * b2) / np.sqrt(1 - rho ** 2)) * (-db2_dS)
            delta = (np.exp(-q * tau_T) * (N2_a1nb1_n + S * dN2_a1nb1_n_dS)
                     - K2 * np.exp(-r * tau_T) * dN2_a2nb2_n_dS
                     + K1 * np.exp(-r * tau) * norm.pdf(a2) * da2_dS)
        else:
            raise ValueError("option type must be 'call' or 'put' in computing the delta")
        return price, delta


    def __call__(self, sample_size, dw=None):
        ref_X, ref_Y, ref_Z, ref_delta = [], [], [], []

        for j in range(len(self.N_list)):  # do not contain placeholder zeros
            ref_Y.append([np.zeros([sample_size, self.dim_y, 1]) for _ in range(self.N_list[j] + 1)])
            ref_Z.append([np.zeros([sample_size, self.dim_y, self.dim_w]) for _ in range(self.N_list[j] + 1)])
            ref_delta.append([np.zeros([sample_size, self.dim_x, 1]) for _ in range(self.N_list[j] + 1)])

        ref_X.append(self.fbsde.get_x0(sample_size).numpy())

        if dw is None:
            ref_DW = [np.random.normal(loc=0., scale=self.ref_h**0.5, size=[sample_size, self.dim_w, 1]) for _ in
                      range(self.ref_N)]
        else:
            ref_DW = dw     # dw = list with len = N, and each element has shape [sample_size, self.dim_w, 1]

        ### use these 3 lines if want to anlyatical solution for GBM X
        W = np.array(ref_DW)
        W = np.cumsum(W, axis=0)
        W = np.insert(W, 0, 0.0, axis=0)

        factor = int(self.ref_N / (self.N1 + self.N2))

        for i in range(0, self.ref_N + 1):
            if 0 <= i <= (self.N1 * factor):  # from index 0 to N1, included. so len = N1+1
                price, delta = self.compound_option(ref_X[i][:, 0, 0], tau=self.T1 - i * self.ref_h, T2=self.T2,
                                                    K1=self.K1, K2=self.K2, r=self.cons_r, q=self.cons_q,
                                                    sigma=self.cons_sigma,
                                                    opt_type1=self.compound_type, opt_type2=self.vanilla_type)
                temp_Y = np.reshape(price, (sample_size, self.dim_y, 1))
                temp_Z = np.reshape(delta, (sample_size, self.dim_y, self.dim_w)) @ (
                    self.fbsde.diffusion(i * self.ref_h, ref_X[i], None, None).numpy())
                temp_delta = np.reshape(delta, (sample_size, self.dim_x, 1))
                ref_Y[0][i - 0] = temp_Y
                ref_Z[0][i - 0] = temp_Z
                ref_delta[0][i - 0] = temp_delta

            if (self.N1 * factor) <= i <= (
                    self.N1 + self.N2) * factor:  # from index N1 to N1+N2, included. so len = N2+1
                price, delta = self.bs_formula(ref_X[i][:, 0, 0], (self.T1 + self.T2) - i * self.ref_h, K=self.K2,
                                               r=self.cons_r, q=self.cons_q, sigma=self.cons_sigma,
                                               opt_type=self.vanilla_type)
                temp_Y = np.reshape(price, (sample_size, self.dim_y, 1))
                temp_Z = np.reshape(delta, (sample_size, self.dim_y, self.dim_w)) @ (
                    self.fbsde.diffusion(i * self.ref_h, ref_X[i], None, None).numpy())
                temp_delta = np.reshape(delta, (sample_size, self.dim_x, 1))
                ref_Y[1][i - self.N1 * factor] = temp_Y
                ref_Z[1][i - self.N1 * factor] = temp_Z
                ref_delta[1][i - self.N1 * factor] = temp_delta

            if i <= self.ref_N - 1:
                ### use euler discretization for X
                # ref_X.append(self.fbsde.X_forward(i * self.ref_h, ref_X[i], 0., 0., ref_DW[i]).numpy())
                ### use analytical x:
                ref_X.append(ref_X[0] * np.exp((self.cons_r - self.cons_q - 0.5 * self.cons_sigma ** 2) * (
                        (i + 1) * self.ref_h) + self.cons_sigma * W[i + 1]))

        # draw the points from the finer grid
        ref_DW = [sum(ref_DW[i:i + factor]) for i in range(0, len(ref_DW), factor)]
        ref_X = ref_X[::factor]

        # convert list to np array and then to the shape [sample size, time step, dim, dim]
        ref_X = np.transpose(np.array(ref_X), axes=[1, 0, 2, 3])
        for j in range(len(self.N_list)):
            ref_Y[j] = np.transpose(np.array(ref_Y[j]), axes=[1, 0, 2, 3])
            ref_Z[j] = np.transpose(np.array(ref_Z[j]), axes=[1, 0, 2, 3])
            ref_delta[j] = np.transpose(np.array(ref_delta[j]), axes=[1, 0, 2, 3])
        return ref_X, ref_Y, ref_Z, ref_DW, ref_delta


class NFoldCall:
    def __init__(self, config):
        self.config = config
        self.eqn_name = config.eqn_config.eqn_name
        self.batch_s = config.eqn_config.batch_s
        self.dim_x = config.eqn_config.dim_x
        self.dim_y = config.eqn_config.dim_y
        self.dim_w = config.eqn_config.dim_w
        self.type_list = config.eqn_config.type_list  # it's all call in this case
        self.N_list = config.eqn_config.N_list
        self.T_list = config.eqn_config.T_list
        self.K_list = config.eqn_config.K_list
        self.T = sum(self.T_list)
        self.N = sum(self.N_list)
        self.ref_N = config.eqn_config.ref_N
        self.ref_h = self.T / self.ref_N

        self.accu_N_list = [0]
        for n_values in self.N_list:
            self.accu_N_list.append(self.accu_N_list[-1] + n_values)

        self.accu_T_list = []
        self.accu_T_list.append(self.T_list[0])
        for t_values in self.T_list[1:]:
            self.accu_T_list.append(self.accu_T_list[-1] + t_values)

        #
        self.cons_r = config.eqn_config.cons_r
        self.cons_q = np.array(config.eqn_config.cons_q).item()
        self.cons_sigma = np.array(config.eqn_config.cons_sigma).item()
        self.fbsde = getattr(importlib.import_module("lib.fbsde"), self.eqn_name)(self.config, convert_np = True)
        self.fbsde.to(device="cpu", dtype=torch.float64)

    def _brownian_time_corr(self, t_ref: float, times: np.ndarray) -> np.ndarray:
        tau = times - t_ref
        L = len(times)
        R = np.eye(L)
        for i in range(L):
            for j in range(i + 1, L):
                rho = np.sqrt(tau[i] / tau[j])
                R[i, j] = rho
                R[j, i] = rho
        return R  # 2d array, including L=1 case.


    def _mvnorm_cdf(self, batch_x: np.ndarray, mu: np.ndarray, corr: np.ndarray, batch_mu=False) -> np.ndarray:
        B, k = batch_x.shape  # support a batch of vectors
        # mu = np.zeros(k)
        out = np.empty(B)
        if batch_mu == True:
            for i in range(B):
                out[i] = multivariate_normal.cdf(x=batch_x[i, :], mean=mu[i, :], cov=corr)
        else:
            for i in range(B):
                out[i] = multivariate_normal.cdf(x=batch_x[i, :], mean=mu, cov=corr)
        return out  # return array (B, )


    def _derivative_mvnorm_cdf(self, batch_x: np.ndarray,
                               mu: np.ndarray, corr: np.ndarray, dx_dv: np.ndarray) -> np.ndarray:
        B, k = batch_x.shape
        # mu = np.zeros(k)
        output = np.zeros(B)
        x = batch_x  # 2d array, = a batch of (a1, a2, ..., ak)
        if k > 1:
            for i in range(k):
                x_ni = np.delete(x, i, axis=1)  # (B, k-1)
                x_i = x[:, i:(i + 1)]  # (B, 1)
                mu_i = mu[i,]

                mu_ni = np.delete(mu, i)  # (k-1, )
                corr_ii = corr[i, i]
                corr_nini = np.delete(np.delete(corr, i, axis=0), i, axis=1)  # (k-1, k-1)
                corr_ini = np.delete(corr, i, axis=1)[i:(i + 1), :]  # (1, k-1)
                corr_nii = np.delete(corr, i, axis=0)[:, i:(i + 1)]  # (k-1, 1)

                mu_ni_con_i = mu_ni + corr_nii[:, 0] * (1.0 / corr_ii) * (x_i - mu_i)  # (B, k-1)
                corr_ni_con_i = corr_nini - corr_nii * (1.0 / corr_ii) @ corr_ini  # (k-1, k-1)

                cdf_vals = self._mvnorm_cdf(x_ni, mu_ni_con_i, corr_ni_con_i, batch_mu=True)  # (B, )
                pdf_vals = norm.pdf(x_i[:, 0] - mu_i, loc=0.0, scale=corr_ii ** 0.5)  # (B, )
                output += pdf_vals * cdf_vals * dx_dv[:, i]
        elif k == 1:
            output = norm.pdf(x[:, 0] - mu[0], loc=0.0, scale=corr[0, 0] ** 0.5) * dx_dv[:, 0]
        else:
            raise ValueError("k must be positive")
        return output  # (B, )


    def _ab_vectors(self, V: np.ndarray, Vstars: np.ndarray, t_ref: float, t_slice: np.ndarray,
                    r: float, sigma: float):
        V = V.reshape(-1, 1)  # from (B, ) to (B, 1)
        Vstars = np.asarray(Vstars, dtype=float).reshape(1, -1)  # (1, k)
        tau = (t_slice - t_ref).reshape(1, -1)  # (1, k)
        sqrt_tau = np.sqrt(tau)  # (1, k)

        # Broadcast: (B,1) / (1,k) → (B,k)  for V/Vstars
        b = (np.log(V / Vstars) + (r - 0.5 * sigma ** 2) * tau) / (sigma * sqrt_tau)
        a = b + sigma * sqrt_tau
        db_dv = 1 / V * (sigma * sqrt_tau) / (sigma * sqrt_tau) ** 2
        da_dv = db_dv
        return a, b, da_dv, db_dv  # each is (B,k) arrays

    def _option_price(self, V, Vstars: np.ndarray,
                      t_ref: float, times: np.ndarray, strikes: np.ndarray,
                      r: float, sigma: float) -> float:

        if t_ref >= times[0]:
            raise ValueError("t_ref must be less than times[0]")

        # _option_price must accept scalar V input, output could be scalar or (1,) or (1,1) arrays
        V = np.asarray(V)
        if V.ndim == 0:  # if it is scalar → make (1,)
            V = V.reshape(1, )

        n = len(times)
        # Build correlation F^1_m for m = 1..n (each is the top-left m×m block of the full R)
        R_full = self._brownian_time_corr(t_ref, times)
        # Build a_l, b_l from V^* and times relative to t_ref
        a, b, _, _ = self._ab_vectors(V, Vstars, t_ref, times, r, sigma)  # (B, n)

        # First term: V * N_n(a1..an; F^1_n)
        term1 = V * self._mvnorm_cdf(batch_x=a, mu=np.zeros(n), corr=R_full)  # (B, )*(B, )

        # Sum term: sum_{m=1..n} K_m * e^{-r(t_m - t_ref)} * N_m(b1..bm; F^1_m)
        sum_term = np.zeros(V.shape[0])
        disc = np.exp(-r * (times - t_ref))  # (n, )
        for m in range(1, n + 1):
            Rm = R_full[:m, :m]
            bm = b[:, :m]
            Nm = self._mvnorm_cdf(batch_x=bm, mu=np.zeros(Rm.shape[0]), corr=Rm)
            sum_term += strikes[m - 1] * disc[m - 1] * Nm
        price = term1 - sum_term
        return price  # shape = (B, )

    def _solve_thresholds(self, all_times: np.ndarray, all_strikes: np.ndarray, r: float, sigma: float) -> np.ndarray:
        """
        t0 doesn't change the V_stars if the number of fold remain the same
        Solve V^n..V^1 thresholds backwards:
        V^n = K_n;
        for l=n-1..1: find V such that Price(subchain starting at l+1, with t_ref=t_l) = K_l
        """
        n = len(all_times)  # all_times = (t1, t2, ..., tn)
        Vstar = np.zeros(n)
        Vstar[-1] = all_strikes[-1]  # V^n = K_n, all_strikes = (k1, k2, ..., kn)

        # this give you the corresponding compound option price at time t_i, that used for solving for V*
        def price_downstream(scalarV, i):  # will be used for index i = n-1, .., 1
            t_ref = all_times[i - 1].item()  # from t_{n-1} = all_times[n-2], to t_1
            times_slice = all_times[i:]  # from t_n, and add until t_2
            strikes_slice = all_strikes[i:]  # from k_{n} and add until t_2
            Vstars_slice = Vstar[i:]  # V^{l+1}..V^n (already known when solving for index l)
            price = self._option_price(scalarV, Vstars_slice, t_ref, times_slice, strikes_slice, r, sigma)
            return price

        # iterate from idx = n-2 to index = 0
        for idx in range(n - 2, -1, -1):
            K_idx = all_strikes[idx]  # from strike prices k_{n-1} until k_1

            # bracket root: prices are increasing in V; pick a wide bracket
            # lower bound: something small; upper bound: say 100× max(K_l, V^{idx+1})
            scalarV_low = 1e-12
            scalarV_high = 100.0 * max(K_idx.item(), float(np.max(all_strikes[idx + 1:])))
            f = lambda scalarV: price_downstream(scalarV, idx + 1) - K_idx

            # expand upper bound if needed
            while f(scalarV_high) < 0:
                scalarV_high *= 2.0
                if scalarV_high > 1e12:
                    raise RuntimeError("Failed to bracket the threshold root")
            Vstar[idx] = brentq(f, scalarV_low, scalarV_high, maxiter=200, xtol=1e-10, rtol=1e-10)
        return Vstar

    def _option_delta(self, V: np.ndarray, Vstars: np.ndarray, t_ref: float, t_slice: np.ndarray, strikes,
                      r: float, sigma: float):
        a, b, da_dv, db_dv = self._ab_vectors(V, Vstars, t_ref, t_slice, r, sigma)
        n = len(t_slice)  # full time list exclude t0 but include T
        R_full = self._brownian_time_corr(t_ref, t_slice)

        term1 = self._mvnorm_cdf(batch_x=a, mu=np.zeros(n), corr=R_full)
        term1 += V * self._derivative_mvnorm_cdf(batch_x=a, mu=np.zeros(n), corr=R_full, dx_dv=da_dv)

        sum_term = np.zeros(V.shape[0])  # (B, )
        disc = np.exp(-r * (t_slice - t_ref))  # (n, )

        for m in range(1, n + 1):
            Rm = R_full[:m, :m]
            bm = b[:, :m]
            dNm_dV = self._derivative_mvnorm_cdf(batch_x=bm, mu=np.zeros(Rm.shape[0]), corr=Rm, dx_dv=db_dv[:, :m])
            sum_term += strikes[m - 1] * disc[m - 1] * dNm_dV

        delta = term1 - sum_term
        return delta  # (B, )


    def nfold_compound_call(self, curr_V, curr_t, times_slice, strikes_slice, r, sigma):
        # _check_inputs(params)
        # V0, r, sigma, t0 = params.V0, params.r, params.sigma, params.t0
        times_slice = np.array(times_slice, dtype=float)
        strikes_slice = np.array(strikes_slice, dtype=float)
        curr_t = float(f"{curr_t:.8f}")

        # times = times_slice, same for strikes
        # V0 is associated with t0, with 0\leq t0 \leq T
        if curr_t < times_slice[0]:  # if current accumulated t_0 < times[0], then consider the current n-fold
            Vstars = self._solve_thresholds(times_slice, strikes_slice, r, sigma)
            price = self._option_price(curr_V, Vstars, curr_t, times_slice, strikes_slice, r, sigma)
            delta = self._option_delta(curr_V, Vstars, curr_t, times_slice, strikes_slice, r, sigma)
        elif curr_t == times_slice[0]:  # consider (n-1)-fold at the compound time
            Vstars = self._solve_thresholds(times_slice, strikes_slice, r, sigma)
            if len(times_slice) >= 2:  # when len(times) = 2, below return usual bs option result
                price = self._option_price(curr_V, Vstars[1:], curr_t, times_slice[1:], strikes_slice[1:], r, sigma)
                delta = self._option_delta(curr_V, Vstars[1:], curr_t, times_slice[1:], strikes_slice[1:], r, sigma)
            else:  # when len(times) = 1, return terminal payoff at final time
                price = np.maximum(curr_V - strikes_slice[0:], 0.0)
                delta = np.where(curr_V - strikes_slice[0:] > 0.0, 1.0, 0.0)
        else:
            raise ValueError("curr_t and times_slice are not correct, current values are:", curr_t, times_slice)
        return price, delta


    def __call__(self, sample_size, dw=None, get_t0_sol = True):
        ref_X, ref_Y, ref_Z, ref_delta = [], [], [], []
        for j in range(len(self.N_list)):
            ref_Y.append([np.zeros([sample_size, self.dim_y, 1]) for _ in range(self.N_list[j] + 1)])
            ref_Z.append([np.zeros([sample_size, self.dim_y, self.dim_w]) for _ in range(self.N_list[j] + 1)])
            ref_delta.append([np.zeros([sample_size, self.dim_x, 1]) for _ in range(self.N_list[j] + 1)])

        ref_DW = dw if dw is not None else [
            np.random.normal(0.0, self.ref_h ** 0.5, (sample_size, self.dim_w, 1))
            for _ in range(self.ref_N)
        ]

        ### if use anlyatical solution of X
        W = np.array(ref_DW)
        W = np.cumsum(W, axis=0)
        W = np.insert(W, 0, 0.0, axis=0)

        ref_X.append(self.fbsde.get_x0(sample_size).numpy())
        factor = int(self.ref_N / self.N)

        for i in range(0, self.ref_N):
            ### use euler discretization
            # ref_X.append(self.fbsde.X_forward(i * self.ref_h, ref_X[i], 0., 0., ref_DW[i]).numpy())
            ### use analytical x:
            ref_X.append(ref_X[0] * np.exp(
                (self.cons_r - self.cons_q - 0.5 * self.cons_sigma ** 2) * ((i + 1) * self.ref_h) + self.cons_sigma * W[
                    i + 1]))

        if get_t0_sol == True:
            for m in range(0, len(self.K_list)):
                for i in range(self.accu_N_list[m] * factor,
                               self.accu_N_list[m + 1] * factor + 1):  # from index 0 to N1, included. so len = N1+1
                    if i == 0:  # only compute the sol at t=0
                        price, delta = self.nfold_compound_call(curr_V=ref_X[i][:, 0, 0], curr_t=i * self.ref_h,
                                                                times_slice=self.accu_T_list[m:], strikes_slice=self.K_list[m:],
                                                                r=self.cons_r, sigma=self.cons_sigma)
                    else:
                        price = np.zeros((sample_size))
                        delta = np.zeros((sample_size))
                    if i < self.accu_N_list[m + 1] * factor:
                        temp_Y = np.reshape(price, (sample_size, 1, 1))
                        temp_Z = np.reshape(delta, (sample_size, 1, 1)) @ (
                            self.fbsde.diffusion(i * self.ref_h, ref_X[i], None, None).numpy())
                        temp_delta = np.reshape(delta, (sample_size, 1, 1))
                    elif i == self.accu_N_list[m + 1] * factor:
                        temp_Y = np.reshape(price, (sample_size, 1, 1))
                        temp_Z = np.reshape(delta, (sample_size, 1, 1)) @ (
                            self.fbsde.diffusion(i * self.ref_h, ref_X[i], None, None).numpy())
                        temp_delta = np.reshape(delta, (sample_size, 1, 1))
                    else:
                        raise ValueError("index i error in computing ref solution")
                    ref_Y[m][i - self.accu_N_list[m] * factor] = temp_Y
                    ref_Z[m][i - self.accu_N_list[m] * factor] = temp_Z
                    ref_delta[m][i - self.accu_N_list[m] * factor] = temp_delta
        else:
            for m in range(0, len(self.K_list)):
                for i in range(self.accu_N_list[m] * factor,
                               self.accu_N_list[m + 1] * factor + 1):  # from index 0 to N1, included. so len = N1+1
                    price, delta = self.nfold_compound_call(curr_V=ref_X[i][:, 0, 0], curr_t=i * self.ref_h,
                                                            times_slice=self.accu_T_list[m:], strikes_slice=self.K_list[m:],
                                                            r=self.cons_r, sigma=self.cons_sigma)
                    if i < self.accu_N_list[m + 1] * factor:
                        temp_Y = np.reshape(price, (sample_size, 1, 1))
                        temp_Z = np.reshape(delta, (sample_size, 1, 1)) @ (
                            self.fbsde.diffusion(i * self.ref_h, ref_X[i], None, None).numpy())
                        temp_delta = np.reshape(delta, (sample_size, 1, 1))
                    elif i == self.accu_N_list[m + 1] * factor:
                        temp_Y = np.reshape(price, (sample_size, 1, 1))
                        temp_Z = np.reshape(delta, (sample_size, 1, 1)) @ (
                            self.fbsde.diffusion(i * self.ref_h, ref_X[i], None, None).numpy())
                        temp_delta = np.reshape(delta, (sample_size, 1, 1))
                    else:
                        raise ValueError("index i error in computing ref solution")
                    ref_Y[m][i - self.accu_N_list[m] * factor] = temp_Y
                    ref_Z[m][i - self.accu_N_list[m] * factor] = temp_Z
                    ref_delta[m][i - self.accu_N_list[m] * factor] = temp_delta


        # draw the points from the finer grid
        ref_DW = [sum(ref_DW[i:i + factor]) for i in range(0, len(ref_DW), factor)]
        ref_X = ref_X[::factor]

        # convert list to np array and then to the shape [sample size, time step, dim, dim]
        ref_X = np.transpose(np.array(ref_X), axes=[1, 0, 2, 3])
        for j in range(len(self.N_list)):
            ref_Y[j] = np.transpose(np.array(ref_Y[j]), axes=[1, 0, 2, 3])
            ref_Z[j] = np.transpose(np.array(ref_Z[j]), axes=[1, 0, 2, 3])
            ref_delta[j] = np.transpose(np.array(ref_delta[j]), axes=[1, 0, 2, 3])
        return ref_X, ref_Y, ref_Z, ref_DW, ref_delta



class BasketBermudan:    # binomial tree method and therefore for 1d example
    def __init__(self, config):
        self.config = config
        self.eqn_name = config.eqn_config.eqn_name
        self.batch_s = config.eqn_config.batch_s
        self.type_list = config.eqn_config.type_list
        self.N_list = config.eqn_config.N_list
        self.T_list = config.eqn_config.T_list
        self.K_list = config.eqn_config.K_list
        self.N = int(sum(self.N_list))
        self.T = sum(self.T_list)
        self.ref_N = int(config.eqn_config.ref_N)
        self.ref_h = float(self.T/self.ref_N)
        self.N_tree = self.N * 100
        self.factor = int(self.ref_N/self.N)

        self.accu_N_list = [0]
        for n_values in self.N_list:
            self.accu_N_list.append(self.accu_N_list[-1] + n_values)

        self.accu_T_list = []
        self.accu_T_list.append(self.T_list[0])
        for t_values in self.T_list[1:]:
            self.accu_T_list.append(self.accu_T_list[-1] + t_values)

        # self.fbsde = getattr(importlib.import_module("lib.fbsde"), self.eqn_name)(self.config, convert_np = True)

        ############ convert parameters to 1d setting
        self.dim_x = config.eqn_config.dim_x
        self.dim_y = config.eqn_config.dim_y
        self.dim_w = config.eqn_config.dim_w

        if isinstance(config.eqn_config.x0, list):
            self.x0_ini = np.array(config.eqn_config.x0)
        elif isinstance(config.eqn_config.x0, (float, int)):
            self.x0_ini = np.ones((self.dim_x, 1)) * config.eqn_config.x0
        else:
            raise TypeError(f"Unsupported input type for config.eqn_config.x0")
        self.cons_r = config.eqn_config.cons_r  #float
        self.cons_q = np.array(config.eqn_config.cons_q)    #2d
        self.cons_sigma = np.array(config.eqn_config.cons_sigma)    #2d


        # Initial basket level: G0 = (∏ S0_i)^(1/d)
        self.x0_ini = np.prod(self.x0_ini)**(1.0/self.dim_x)

        # Effective variance: sigma_bar^2 = (1/d^2) * sigma^T rho sigma
        sigma_1d = (np.diag( self.cons_sigma @ np.transpose(self.cons_sigma) ))**0.5
        sigma_diag = np.diag(sigma_1d)
        rho = np.linalg.inv(sigma_diag) @ (self.cons_sigma @ np.transpose(self.cons_sigma)) @ np.linalg.inv(sigma_diag)
        sigma_bar_sq = (sigma_1d  @ rho @ sigma_1d )/(self.dim_x**2)

        # Effective dividend yield q_bar so that: dG/G = (r - q_bar) dt + sigma_bar dB
        # Exact identity: q_bar = mean(q) + 0.5*mean(sigma^2) - 0.5*sigma_bar^2
        self.cons_sigma = np.sqrt(sigma_bar_sq)
        self.cons_q = np.mean(self.cons_q) + 0.5 * np.mean(sigma_1d**2) - 0.5 * sigma_bar_sq

        self.dim_x, self.dim_y, self.dim_w = 1, 1, 1



    def bermudan_option(self, S0, K, T, r, q, sigma, N_tree, exercise_times, option_type):
        N = N_tree

        # input = accu time without t=0, convert it to exercise time in terms of the steps number
        # for time_values in exercise_times:
        #     val = N * (time_values / T)
        #     if not float(val).is_integer():
        #         raise ValueError(f"Should set N_tree correctly. N_tree = {N_tree}, T= {T}, accu_time = {exercise_times}")
        exercise_times = [int(N * (time_values / T)) for time_values in exercise_times]

        option_type = option_type.lower()
        if option_type not in {"put", "call"}:
            raise ValueError("option_type must be 'put' or 'call'")

        # Define payoff function depending on option_type
        if option_type == "put":
            payoff = lambda S: np.maximum(K - S, 0.0)
        else:  # "call"
            payoff = lambda S: np.maximum(S - K, 0.0)

        dt = T / N
        disc = np.exp(-r * dt)

        # CRR up/down factors and risk-neutral probability
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        p = (np.exp((r - q) * dt) - d) / (u - d)

        # If None: exercise allowed at all time steps (American-style)
        if exercise_times is None:
            exercise_times = set(range(1, N + 1))
        else:
            exercise_times = set(exercise_times)

        # Build stock price tree
        S_tree = []
        for i in range(N + 1):
            j = np.arange(i + 1)
            S_i = S0 * (u ** j) * (d ** (i - j))
            S_tree.append(S_i)

        V_tree = [None] * (N + 1)
        exercise_mask = [None] * (N + 1)

        # Terminal payoff at maturity (must be exercised or expire)
        S_T = S_tree[-1]
        V_T = payoff(S_T)
        V_tree[-1] = V_T

        # True: exercise / final payoff taken at maturity
        # exercise_mask[-1] = np.ones_like(V_T, dtype=bool)
        exercise_mask[-1] = V_T > 0.0

        # Backward induction
        for i in reversed(range(N)):  # i = N-1,...,0
            S_i = S_tree[i]
            V_next = V_tree[i + 1]

            # Continuation value
            V_cont = disc * (p * V_next[1:] + (1.0 - p) * V_next[:-1])

            # Immediate exercise value (intrinsic)
            intrinsic = payoff(S_i)

            tol = 0.0
            if i in exercise_times:
                # Bermudan: can exercise here
                V_i = np.maximum(intrinsic, V_cont)
                # mark exercise only if strictly better than continuation
                ex_i = intrinsic > (V_cont + tol)
            else:
                # Cannot exercise here
                V_i = V_cont
                ex_i = np.zeros_like(V_i, dtype=bool)

            V_tree[i] = V_i
            exercise_mask[i] = ex_i
        price0 = V_tree[0][0]

        # Delta at t=0 from first-step nodes
        S1 = S_tree[1]
        V1 = V_tree[1]
        delta0 = (V1[1] - V1[0]) / (S1[1] - S1[0])
        return price0, delta0, S_tree, exercise_mask


    def bermudan_exercise_region(self, S_tree, exercise_mask, T, exercise_times, include_maturity=True):
        N = len(S_tree) - 1
        exercise_times = [int(N * (time_values / T)) for time_values in exercise_times]  # in terms of the time steps

        times = np.linspace(0.0, T, N + 1)
        # exercise_times always includes N, but plotting should obey include_maturity
        plot_times = set(exercise_times)
        if not include_maturity:
            plot_times.discard(N)  # make 100% sure maturity is excluded

        t_ex, S_ex = [], []
        t_cont, S_cont = [], []

        for i, (S_i, ex_i) in enumerate(zip(S_tree, exercise_mask)):
            # Only plot times in plot_times
            if i not in plot_times:
                continue
            t_i = np.full_like(S_i, times[i], dtype=float)
            t_ex.append(t_i[ex_i])
            S_ex.append(S_i[ex_i])
            t_cont.append(t_i[~ex_i])
            S_cont.append(S_i[~ex_i])
        return t_cont, S_cont, t_ex, S_ex


    def __call__(self, sample_size, dw=None):
        ref_X, ref_Y, ref_Z, ref_delta = [], [], [], []

        if dw is None:
            ref_DW = [np.random.normal(loc=0., scale=self.ref_h**0.5, size=[sample_size, self.dim_w, 1]) for _ in
                      range(self.ref_N)]
        else:
            ref_DW = dw  # dw = list with len = N, and each element has shape [sample_size, self.dim_w, 1]
            if sample_size != dw.shape[0]:
                raise ValueError("sample_size must be equal to dw.shape[0]")

        ### if use anlyatical solution of X
        W = np.array(ref_DW)
        W = np.cumsum(W, axis=0)
        W = np.insert(W, 0, 0.0, axis=0)

        for j in range(len(self.N_list)):
            ref_Y.append([np.zeros([sample_size, self.dim_y, 1]) for _ in range(self.N_list[j] + 1)])
            ref_Z.append([np.zeros([sample_size, self.dim_y, self.dim_w]) for _ in range(self.N_list[j] + 1)])
            ref_delta.append(([np.zeros([sample_size, self.dim_x, 1]) for _ in range(self.N_list[j] + 1)]))

        ref_X.append(self.x0_ini + np.zeros([sample_size, self.dim_x, 1]))

        for i in range(0, self.ref_N):
            ### use euler discretization
            # ref_X.append(self.fbsde.X_forward(i * self.ref_h, ref_X[i], 0., 0., ref_DW[i]).numpy())
            ### use analytical x:
            ref_X.append(ref_X[0] * np.exp(
                (self.cons_r - self.cons_q - 0.5 * self.cons_sigma ** 2) * ((i + 1) * self.ref_h)
                + self.cons_sigma * W[i + 1]))

        S0 = ref_X[0][0, 0, 0]
        for m in range(0, len(self.K_list)):
            for i in range(self.accu_N_list[m] * self.factor, self.accu_N_list[m + 1] * self.factor + 1):  # from index 0 to N1, included.
                if i == 0:
                    price, delta, S_tree, exercise_mask = self.bermudan_option(S0=S0, K=self.K_list[-1], T=self.T,
                                                                  r=self.cons_r, q=self.cons_q, sigma=self.cons_sigma,
                                                                  N_tree=self.N_tree, exercise_times=self.accu_T_list,
                                                                  option_type=self.type_list[-1])
                    price = price + np.zeros((sample_size, 1, 1))   # increase the dimension only
                    delta = delta + np.zeros((sample_size, 1, 1))
                    t_cont, S_cont, t_ex, S_ex = self.bermudan_exercise_region(S_tree, exercise_mask, T=self.T, exercise_times=self.accu_T_list)
                    exercise_region = [t_cont, S_cont, t_ex, S_ex]
                else:   # placeholders
                    price = np.zeros((sample_size, 1, 1)) + 1.0
                    delta = np.zeros((sample_size, 1, 1)) + 1.0

                temp_Y = np.reshape(price, (sample_size, 1, 1))
                temp_delta = np.reshape(delta, (sample_size, 1, 1))
                temp_Z = np.transpose(temp_delta, axes=(0,2,1)) @ (self.cons_sigma * ref_X[i])

                ref_Y[m][i - self.accu_N_list[m] * self.factor] = temp_Y
                ref_delta[m][i - self.accu_N_list[m] * self.factor] = temp_delta
                ref_Z[m][i - self.accu_N_list[m] * self.factor] = temp_Z


        # convert to finder grid and change shape
        ref_DW = [sum(ref_DW[i:i + self.factor]) for i in range(0, len(ref_DW), self.factor)]
        ref_X = ref_X[::self.factor]
        # convert list to np array and then to the shape [sample size, time step, dim, dim]
        ref_X = np.transpose(np.array(ref_X), axes=[1, 0, 2, 3])
        for j in range(len(self.N_list)):
            ref_Y[j] = np.transpose(np.array(ref_Y[j]), axes=[1, 0, 2, 3])
            ref_Z[j] = np.transpose(np.array(ref_Z[j]), axes=[1, 0, 2, 3])
            ref_delta[j] = np.transpose(np.array(ref_delta[j]), axes=[1, 0, 2, 3])
        return ref_X, ref_Y, ref_Z, ref_DW, ref_delta




