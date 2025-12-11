from data_params import *
from BayesianFunc import BayesianFunctions
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from scipy.linalg import expm
from copy import deepcopy
from scipy.interpolate import interp1d

plt.rcParams.update({'font.size': 18, 'axes.labelsize': 22,   'axes.titlesize': 22,  'xtick.labelsize': 20,
    'ytick.labelsize': 20,   'legend.fontsize': 22})

bayes_func = BayesianFunctions()
df = procces_data(init_day='2024-01-01')


MIN2DAY = 1.0 / 1440.0
fixed_params = compute_params_from_df(df)

#df = df[df['Date'] >= '2025-02-23'] #######
arrivals = df['DAILY_ED_ARRIVALS'].values
T = len(arrivals)
t_data = np.arange(len(arrivals))


Ad_Hs = df['DIRECT_Admt_IP_Surge'].values
Ad_Hm = df['DIRECT_Admt_MED_SURG_TELE'].values
Ad_ICU = df['DIRECT_Admt_ICU'].values

At_Hs = df['TRNSFR_ADMT_IP_Surge'].values
At_Hm = df['TRNSFR_ADMT_MED_SURG_TELE'].values
At_ICU = df['TRNSFR_ADMT_ICU'].values


# W, S, B_Hs, B_Hm, B_I, Hs, Hm, I, D
y0 = [10, 10, 5, 5, 2, df['OCC_BEDS_IP_SURGE'].iloc[0] if len(df) > 0 else 10,df['OCC_BEDS_MED_SURG_TELE'].iloc[0],  df['OCC_BEDS_ICU'].iloc[0], 0]

obs_Hs = df['OCC_BEDS_IP_SURGE'].values
obs_Hm = df['OCC_BEDS_MED_SURG_TELE'].values
obs_ICU = df['OCC_BEDS_ICU'].values

obs_Dis_Hs = df['Discharges_IP_Surge'].values
obs_Dis_Hm = df['Discharges_MED_SURG_TELE'].values
obs_Dis_ICU = df['Discharges_ICU'].values

obs_ED_adm_Hs  = df['ED_Admit_IP_Surge'].values
obs_ED_adm_Hm  = df['ED_Admit_MED_SURG_TELE'].values
obs_ED_adm_ICU = df['ED_Admit_ICU'].values

arrivals_mean = arrivals.mean()
Ad_Hs_mean = Ad_Hs.mean()
Ad_Hm_mean = Ad_Hm.mean()
Ad_ICU_mean = Ad_ICU.mean() + 0.5
At_Hs_mean = At_Hs.mean()
At_Hm_mean = At_Hm.mean()
At_ICU_mean = At_ICU.mean()


def compute_equilibrium_data(fixed_params, arrivals_mean, Ad_Hs_mean, Ad_Hm_mean, Ad_ICU_mean,
                             At_Hs_mean, At_Hm_mean, At_ICU_mean):
    sigma = fixed_params['sigma']
    omega = fixed_params['omega']

    pED_Hs = fixed_params['pED_Hs']
    pED_Hm = fixed_params['pED_Hm']
    pED_ICU = fixed_params['pED_ICU']

    xi_I = fixed_params['xi_I']
    xi_Hs = fixed_params['xi_Hs']
    xi_Hm = fixed_params['xi_Hm']

    varphi_I = fixed_params['varphi_I']
    varphi_D = fixed_params['varphi_D'] #0.25
    varphi_Hm = fixed_params['varphi_Hm']
    eps_Hs = fixed_params['eps_Hs']

    ###############################
    gamma = fixed_params['gamma']
    eps_Hm = fixed_params['eps_Hm']
    psi_D = fixed_params['psi_D']
    psi_I = fixed_params['psi_I']
    eps_D = fixed_params['eps_D']

    # Step 2a: upstream compartments
    W_star = arrivals_mean / (sigma + omega)
    S_star = sigma * W_star / gamma

    B_Hs_star = pED_Hs * gamma * S_star / xi_Hs
    B_Hm_star = pED_Hm * gamma * S_star / xi_Hm
    B_I_star = pED_ICU * gamma * S_star / xi_I

    # Step 2b: total admissions including external and internal
    A_Hs = xi_Hs * B_Hs_star + Ad_Hs_mean + At_Hs_mean
    A_Hm = xi_Hm * B_Hm_star + Ad_Hm_mean + At_Hm_mean
    A_I = xi_I * B_I_star + Ad_ICU_mean + At_ICU_mean

    A_mat = np.array([[varphi_I + varphi_D + varphi_Hm, 0, -eps_Hs],
        [-varphi_Hm, psi_I + psi_D, -eps_Hm],
        [-varphi_I, -psi_I, eps_Hs + eps_Hm + eps_D]])
    b_vec = np.array([A_Hs, A_Hm, A_I])
    Hs_star, Hm_star, I_star = np.linalg.solve(A_mat, b_vec)
    D_star = varphi_D * Hs_star + psi_D * Hm_star + eps_D * I_star

    return { "W": W_star, "S": S_star, "B_Hs": B_Hs_star, "B_Hm": B_Hm_star,  "B_I": B_I_star, "Hs": Hs_star,
             "Hm": Hm_star,  "I": I_star, "D": D_star }

def jacobian_at_equilibrium(fixed_params):
    varphi_I = fixed_params['varphi_I']
    varphi_D = fixed_params['varphi_D']
    varphi_Hm = fixed_params['varphi_Hm']
    psi_I = fixed_params['psi_I']
    psi_D = fixed_params['psi_D']
    eps_Hs = fixed_params['eps_Hs']
    eps_Hm = fixed_params['eps_Hm']
    eps_D = fixed_params['eps_D']
    # Jacobian rows correspond to [dHs/dt, dHm/dt, dI/dt, dD/dt] derivatives wrt [Hs,Hm,I,D]
    J = np.array([[-(varphi_I + varphi_D + varphi_Hm), 0, eps_Hs, 0],
        [varphi_Hm, -(psi_I + psi_D), eps_Hm, 0],
        [varphi_I, psi_I, -(eps_Hs + eps_Hm + eps_D), 0],
        [varphi_D, psi_D, eps_D, 0] ])
    # eigenvalues
    eigvals = np.linalg.eigvals(J)
    return J, eigvals



def transient_response_for_surge(ad_hs, ad_hm, ad_icu, T_surge=14, dt=0.5):
    equilibrium = compute_equilibrium_data(fixed_params, arrivals_mean, Ad_Hs_mean, Ad_Hm_mean, Ad_ICU_mean,
                                           At_Hs_mean, At_Hm_mean, At_ICU_mean)

    # baseline downstream equilibrium vector x0 (Hs,Hm,I)
    x0 = np.array([equilibrium['Hs'], equilibrium['Hm'], equilibrium['I']], dtype=float)

    xs = compute_equilibrium_data(fixed_params, arrivals_mean, ad_hs, ad_hm, ad_icu,
                                           At_Hs_mean, At_Hm_mean, At_ICU_mean)
    x_step =  np.array([xs['Hs'], xs['Hm'], xs['I']])
    J_full, eigvals = jacobian_at_equilibrium(fixed_params)
    J = J_full[:3, :3]

    # Precompute matrix exponential function for multiples of dt
    times = np.arange(0, T_surge + 80, dt)  # simulate some relaxation after surge
    x_ts = np.zeros((len(times), 3))
    # initial state is baseline equilibrium
    x_ts[0,:] = x0.copy()
    delta_eq = x_step - x0
    for k, t in enumerate(times):
        if t <= T_surge:
            y_t = (np.eye(3) - expm(J*t)).dot(delta_eq)
            x_ts[k,:] = x0 + y_t
        else:
            # After step-off at T_surge, system returns towards baseline: treat as a new initial condition
            # At t=T_surge state is x(T_surge); for t > T_surge the forcing is removed, so homogeneous solution:
            x_Ts = x0 + (np.eye(3) - expm(J*T_surge)).dot(delta_eq)
            # for time tau = t - T_surge:
            tau = t - T_surge
            x_ts[k,:] = x0 + expm(J*tau).dot(x_Ts - x0)

    # compute cumulative extra bed-days during [0, times[-1]]: integrate (x(t)-x0)
    dt_arr = np.diff(times, prepend=0)
    extra_beddays = np.trapz(np.sum(x_ts - x0, axis=1), times)  # total extra bed-days across all 3 comps
    extra_beddays_per_comp = { 'Hs': np.trapz(x_ts[:,0]-x0[0], times),
                               'Hm': np.trapz(x_ts[:,1]-x0[1], times),
                               'I' : np.trapz(x_ts[:,2]-x0[2], times) }

    # Pack results
    ts_results = {'times': times, 'x_ts': x_ts, 'x0': x0, 'x_step': x_step,
                  'extra_beddays_total': extra_beddays,
                  'extra_beddays_per_comp': extra_beddays_per_comp,
                  'eigvals': eigvals}
    return ts_results


T_surge=14
def ed_ode(y, t, params):
    W, S, B_Hs, B_Hm, B_I, Hs, Hm, I, D = y
    if t <= T_surge:
        ad_hs, ad_hm, ad_icu = params
    else:
        ad_hs, ad_hm, ad_icu = Ad_Hs_mean, Ad_Hm_mean,  Ad_ICU_mean


    sigma = fixed_params['sigma']
    omega = fixed_params['omega']

    pED_Hs = fixed_params['pED_Hs']
    pED_Hm = fixed_params['pED_Hm']
    pED_ICU = fixed_params['pED_ICU']

    xi_I = fixed_params['xi_I']
    xi_Hs = fixed_params['xi_Hs']
    xi_Hm = fixed_params['xi_Hm']

    varphi_I = fixed_params['varphi_I']
    varphi_D = fixed_params['varphi_D'] #0.25
    varphi_Hm = fixed_params['varphi_Hm']
    eps_Hs = fixed_params['eps_Hs']

    ###############################
    gamma = fixed_params['gamma']
    eps_Hm = fixed_params['eps_Hm']
    psi_D = fixed_params['psi_D']
    psi_I = fixed_params['psi_I']
    eps_D = fixed_params['eps_D']
    # Prevent negative values
    lam = arrivals_mean
    dW = lam - (sigma + omega) * W
    dS = sigma * W - gamma * S

    dB_Hs = pED_Hs * gamma * S - xi_Hs * B_Hs
    dB_Hm = pED_Hm * gamma * S - xi_Hm * B_Hm
    dB_I = pED_ICU * gamma* S - xi_I * B_I

    Admissions_Hs = xi_Hs * B_Hs + ad_hs +  At_Hs_mean
    Admissions_Hm = xi_Hm * B_Hm + ad_hm +  At_Hm_mean
    Admissions_ICU = xi_I * B_I +  ad_icu +  At_ICU_mean

    dHs = Admissions_Hs + eps_Hs * I - (varphi_I + varphi_D + varphi_Hm) * Hs
    dHm = Admissions_Hm + eps_Hm * I + varphi_Hm * Hs - (psi_D + psi_I) * Hm
    dI = Admissions_ICU + varphi_I * Hs + psi_I * Hm - (eps_Hs + eps_Hm + eps_D) * I
    dD = varphi_D * Hs + psi_D * Hm + eps_D * I

    return [dW, dS, dB_Hs, dB_Hm, dB_I, dHs, dHm, dI, dD]

t = np.arange(94)
params = (1.1 *Ad_Hs_mean, 1.2 * Ad_Hm_mean,  1.5 *Ad_ICU_mean)

mu = odeint(ed_ode, y0, t, hmax=0.5, mxstep=5000, rtol=1e-3, atol=1e-3, args=(params,) )

mu_Hs= mu[:,5]
mu_Hm= mu[:,6]
mu_I= mu[:,7]

if __name__ == '__main__':
    #---------------------- Transient response for surge
    res = transient_response_for_surge(ad_hs= 1.1 * Ad_Hs_mean, ad_hm= 1.2 * Ad_Hm_mean, ad_icu= 1.5 * Ad_ICU_mean, T_surge=14, dt=0.5)

    times = res['times'];
    x_ts = res['x_ts'];
    x0 = res['x0']

    plt.figure(figsize=(10, 5))
    #plt.plot(times, x_ts[:, 0], label='Hs (total)')
    plt.plot(t, mu_Hm, label='Hm', ls='--')
    plt.plot(times, x_ts[:, 1], label='Hm (total)')
    #plt.plot(times, x_ts[:, 2], label='ICU (I)')
    plt.axvline(21, color='k', linestyle='--', label='Surge off (day 21)')
    plt.hlines(x0, times[0], times[-1], colors=['C0', 'C1', 'C2'], linestyles=':', label='Baseline')
    plt.xlabel('Days since surge start')
    plt.ylabel('Occupancy (beds)')
    plt.title('Transient occupancy during +50% direct Hm admission surge (21 days)')
    plt.legend()
    plt.grid(True)
    plt.show()
    # Print cumulative extra bed-days
    print('Extra bed-days (total, across Hs+Hm+I approx):', res['extra_beddays_total'])
    print('Extra bed-days per compartment:', res['extra_beddays_per_comp'])

    #--------
    #cap_med_surg = int((df['TTL_BEDS_MED_SURG_TELE'] - df['UNAVBL_BEDS_MED_SURG_TELE']).median())



