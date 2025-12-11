import numpy as np
from scipy.stats import beta, gamma
import matplotlib.pyplot as plt
import scipy.stats as ss
import matplotlib.dates as mdates
import pandas as pd

class BayesianFunctions:
    """
    Class for Bayesian computation functions including priors, likelihood,
    MCMC sampling, and visualization of priors/posteriors.
    """
    #def __init__(self:, data_processor)
    def __init__(self):
        """
        Initialize with a HospitalDataProcessor instance.

        Args:
            data_processor: HospitalDataProcessor object containing data and parameters
        """
        #self.dp = data_processor
        #self.params = data_processor.get_all_parameters()
        self.cv = 0.5  # coefficient of variation

    def gamma_prior(self, x, mean, sd):
        var = sd ** 2
        shape = mean ** 2 / var
        scale = var / mean
        return ss.gamma.logpdf(x, a=shape, scale=scale)

    def beta_logpdf_from_mean_sd(self, x, mean, sd):
        if x <= 0 or x >= 1 or mean <= 0 or mean >= 1 or sd <= 0:
            return -np.inf
        var = sd ** 2
        # method-of-moments for Beta: mean = a/(a+b), var = ab/((a+b)^2 (a+b+1))
        # Solve for a and b:
        temp = mean * (1 - mean) / var - 1
        if temp <= 0:
            return -np.inf
        a = mean * temp
        b = (1 - mean) * temp
        return ss.beta.logpdf(x, a=a, b=b)

    def gamma_params_from_mean_cv(self, mean, cv):
        """Return shape, scale for a Gamma with given mean and coefficient of variation."""
        sd = cv * mean
        shape = (mean / sd) ** 2
        scale = sd ** 2 / mean
        return shape, scale

    def beta_params_from_mean_sd(self, mean, sd):
        """Return alpha, beta for a Beta with given mean and sd."""
        var = sd ** 2
        alpha = ((1 - mean) / var - 1 / mean) * mean ** 2
        beta = alpha * (1 / mean - 1)
        return alpha, beta

    def gamma_rvs_from_mean_sd(self, mean, sd):
        var = sd ** 2
        shape = mean ** 2 / var
        scale = var / mean
        return ss.gamma.rvs(a=shape, scale=scale)

    # Beta prior for pM
    def beta_rvs_from_mean_sd(self, mean, sd):
        var = sd ** 2
        temp = mean * (1 - mean) / var - 1
        a = mean * temp
        b = (1 - mean) * temp
        return ss.beta.rvs(a=a, b=b)

    def plot_priors(self, param_priors, init_pars, ncols=3):
        n = len(param_priors)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = axes.flatten()

        for i, (param, spec) in enumerate(param_priors.items()):
            ax = axes[i]
            prior_value= init_pars[param]
            a, scale = spec["args"]
            if spec["dist"] == "gamma":
                x = np.linspace(0, gamma.ppf(0.995, a, scale=scale), 500)
                ax.plot(x, gamma.pdf(x, a, scale=scale), 'b-', lw=2)
                ax.set_title(f"{param} ~ Gamma(shape={a:.2f}, scale={scale:.2f})")
            else: #spec["dist"] == "beta":
                alpha, beta = spec["args"]
                x = np.linspace(0, 1, 500)
                ax.plot(x, ss.beta.pdf(x, alpha, beta), 'b-', lw=2, label="Prior")
            ax.axvline(prior_value, color='k', linestyle='--', lw=1.5)
            ax.grid(True)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    def plot_priors_posteriors(self, param_priors, init_pars, samples, ncols=3):
        n = len(param_priors)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = axes.flatten()

        for i, (param, spec) in enumerate(param_priors.items()):
            ax = axes[i]
            prior_value = init_pars[param]
            a, scale = spec["args"]

            if spec["dist"] == "gamma":
                x = np.linspace(0, gamma.ppf(0.995, a, scale=scale), 500)
                ax.plot(x, gamma.pdf(x, a, scale=scale), 'b-', lw=2)
                ax.set_title(f"{param} ~ Gamma(shape={a:.2f}, scale={scale:.2f})")

            elif spec["dist"] == "normal":
                mu, sigma = spec["args"]
                x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 500)
                ax.plot(x, ss.norm.pdf(x, mu, sigma), 'b-', lw=2)

            else:  # spec["dist"] == "beta":
                alpha, beta = spec["args"]
                x = np.linspace(0, 1, 500)
                ax.plot(x, ss.beta.pdf(x, alpha, beta), 'b-', lw=2, label="Prior")
            ax.axvline(prior_value, color='k', linestyle='--', lw=1.5, label='Init value')

            # Overlay posterior histogram/density
            if samples.shape[1] > i:
                post = samples[:, i]
                ax.hist(post, bins=40, density=True, color='orange', alpha=0.6, label='Posterior')
                ax.axvline(np.mean(post), color='red', linestyle='-', lw=1.5, label='Posterior mean')

            ax.set_title(f"{param} ~ Gamma(shape={a:.2f}, scale={scale:.2f})")
            ax.grid(True)
            ax.legend()
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    def plot_surge_event(self, df, solns_plain, n_train, solns_plain_ss=None, surge='baseline'):
        fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        axs = axs.ravel()
        colors = ['blue', 'green', 'orange']
        colors1 = ['red', 'red', 'red']
        col_name = ['OCC_BEDS_IP_SURGE', 'OCC_BEDS_MED_SURG_TELE', 'OCC_BEDS_ICU']
        col_name2 = ['IP SURGE', 'MED SURG ELE', 'ICU']
        dates = df['Date']

        for i in range(3):
            median = np.median(solns_plain[:, i, :], axis=0)
            lower = np.percentile(solns_plain[:, i, :], 2.5, axis=0)
            upper = np.percentile(solns_plain[:, i, :], 97.5, axis=0)
            axs[i].fill_between(dates[:n_train], lower[:n_train], upper[:n_train], color=colors[i], alpha=0.3)
            axs[i].plot(dates[:n_train], median[:n_train], label=col_name2[i], color=colors[i], lw=2)
            axs[i].fill_between(dates[n_train:], lower[n_train:], upper[n_train:], color=colors1[i], alpha=0.3)
            axs[i].plot(dates[n_train:], median[n_train:], color=colors1[i], lw=2)
            if surge == 'surge':
                median_s = np.median(solns_plain_ss[:, i, :], axis=0)
                lower_s = np.percentile(solns_plain_ss[:, i, :], 2.5, axis=0)
                upper_s = np.percentile(solns_plain_ss[:, i, :], 97.5, axis=0)
                axs[i].fill_between(dates[n_train:], lower_s, upper_s, color='grey', alpha=0.3)
                axs[i].plot(dates[n_train:], median_s, color='grey', lw=2)
                if i == 1:
                    axs[i].set_ylim(350, 600)

            axs[i].plot(dates, df[col_name[i]], color='k', alpha=0.8, lw=2)
            axs[i].grid('on')
            axs[i].legend(loc='upper left', fontsize=18)
            axs[i].tick_params(axis='both', labelsize=16)  # increase tick font size

        axs[1].set_ylabel("Beds Occupied", fontsize=20)
        axs[-1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.SU, interval=1))
        axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %y'))
        fig.autofmt_xdate(rotation=30, ha='center')
        plt.tight_layout()
        # plt.savefig("beds_occupancy_plot.png", dpi=300, bbox_inches="tight")  # save figure
        plt.show()

    def plot_priors_posteriors(self, param_priors, init_pars, samples, ncols=3):
        n = len(param_priors)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = axes.flatten()

        for i, (param, spec) in enumerate(param_priors.items()):
            ax = axes[i]
            prior_value = init_pars[param]
            a, scale = spec["args"]
            x = np.linspace(0, gamma.ppf(0.995, a, scale=scale), 500)
            ax.plot(x, gamma.pdf(x, a, scale=scale), 'b-', lw=2, label='Prior')
            ax.axvline(prior_value, color='k', linestyle='--', lw=1.5, label='Init value')

            # Overlay posterior histogram/density
            if samples.shape[1] > i:
                post = samples[:, i]
                ax.hist(post, bins=40, density=True, color='orange', alpha=0.6, label='Posterior')
                ax.axvline(np.mean(post), color='red', linestyle='-', lw=1.5, label='Posterior mean')
            ax.set_title(f"{param} ~ Gamma(shape={a:.2f}, scale={scale:.2f})")
            ax.grid(True)
            ax.legend()

        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    def estimate_phi_moments(self, y, min_phi=1e-6, max_phi=1e8, method='global'):
        """
        Estimate NB dispersion phi by method-of-moments.
        y: 1D array-like of counts (numpy array or pandas Series)
        method: 'global' (use entire series) or 'rolling' (compute rolling and take median)
        Returns: phi (float)
        """
        y = np.asarray(y, dtype=float)
        if method == 'global':
            mu = np.mean(y)
            var = np.var(y, ddof=1)  # sample variance
            denom = var - mu
            if denom <= 0:
                return max_phi  # treat as Poisson-like
            phi = (mu ** 2) / denom
            phi = np.clip(phi, min_phi, max_phi)
            return phi

        elif method == 'rolling':
            # choose a reasonable window: e.g., 14 or 28 days depending on data length
            window = min(28, max(7, len(y) // 10))
            rolling_mu = pd.Series(y).rolling(window, min_periods=5).mean()
            rolling_var = pd.Series(y).rolling(window, min_periods=5).var(ddof=1)
            phis = []
            for mu, var in zip(rolling_mu.dropna(), rolling_var.dropna()):
                denom = var - mu
                if denom <= 0:
                    continue
                phis.append((mu ** 2) / denom)
            if len(phis) == 0:
                return max_phi
            phi_med = float(np.median(phis))
            phi_med = np.clip(phi_med, min_phi, max_phi)
            return phi_med
        else:
            raise ValueError("method must be 'global' or 'rolling'")

# def get_erlang_entry_weights(m, decay_rate=0.8):
#     weights = np.array([decay_rate ** k for k in range(m)])
#     return weights / weights.sum()


#data = np.load("mcmc_output.npz", allow_pickle=True)



####
# fig, axs = plt.subplots(4, 1, figsize=(14, 10),sharex=True)
# axs = axs.ravel()
# colors = ['blue', 'green', 'orange', 'red', 'magenta', 'cyan']
# colors1 = ['red', 'red', 'red', 'red']
# col_name= ['OCC_BEDS_IP_SURGE','OCC_BEDS_MED_SURG_TELE', 'OCC_BEDS_ICU','Discharges_all']
# col_name2= ['IP SURGE',' MED SURG ELE', 'ICU','Discharge']

# params= (varphi_I_est, varphi_D_est, pM_est, eps_Hm_est, eps_Hs_est, nu_est, shape_est, scale_est, shape_y0_est, scale_y0_est)
#
#
# fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
#
# sol_temp = simulate(params, t_fit, obs_Hs, obs_Hm, obs_I,obs_D, mode='admission true')
# Hs_pred = sol_temp[:, 0]
# Hm_pred = np.sum(sol_temp[:, 1:1 + m], axis=1)
# I_pred = sol_temp[:, 1 + m]
# D_pred = sol_temp[:, 1 + m]
# solns_plain = [Hs_pred, Hm_pred, I_pred, D_pred]
#
# colors = ['blue', 'green', 'orange', 'red', 'magenta', 'cyan']
# colors1 = ['red', 'red', 'red', 'red']
# col_name = ['OCC_BEDS_IP_SURGE', 'OCC_BEDS_MED_SURG_TELE', 'OCC_BEDS_ICU', 'Discharges_all']
# col_name2 = ['IP SURGE', ' MED SURG ELE', 'ICU', 'Discharge']
# for i in range(4):  # 3
#     median = solns_plain[i]
#     axs[i].plot(t_fit[:n_train], median[:n_train], label=col_name2[i], color=colors[i], lw=2)
#     axs[i].plot(t_fit[n_train:], median[n_train:], color=colors1[i], lw=2)
#     if col_name[i] == 'Discharges_all':
#         axs[i].plot(t_fit, np.diff(np.insert(df[col_name[i]], 0, 0)), color='k', alpha=0.8, lw=2)
#     else:
#         axs[i].plot(t_fit, df[col_name[i]], color='k', alpha=0.8, lw=2)
#     plt.legend()
# plt.show()
