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


from scipy.linalg import expm
import numpy as np





# def transient_response_for_multi_surge(
#     surge_specs,
#     relax_time=80,
#     dt=0.5
# ):
def transient_response_for_multi_surge(surge_specs, times):
    """
    Generalized transient response with independent surges per compartment.

    surge_specs : dict
        Example:
        {
          'Hs': {'on': 3, 'off': 7,  'amp': ad_hs},
          'Hm': {'on': 0, 'off': 10, 'amp': ad_hm},
          'I' : {'on': 3, 'off': 4,  'amp': ad_icu}
        }
    """

    # ---- Baseline equilibrium ----
    equilibrium = compute_equilibrium_data(
        fixed_params, arrivals_mean,
        Ad_Hs_mean, Ad_Hm_mean, Ad_ICU_mean,
        At_Hs_mean, At_Hm_mean, At_ICU_mean
    )

    x0 = np.array([equilibrium['Hs'],
                   equilibrium['Hm'],
                   equilibrium['I']], dtype=float)

    # ---- Jacobian ----
    J_full, eigvals = jacobian_at_equilibrium(fixed_params)
    J = J_full[:3, :3]

    # ---- Time grid ----
    #t_end = max(spec['off'] for spec in surge_specs.values()) + relax_time
    #times = np.arange(0, t_end, dt)

    x_ts = np.zeros((len(times), 3))
    x_ts[0, :] = x0.copy()

    # ---- Precompute equilibrium shifts for each compartment ----
    delta_eq = {}
    baseline_ads = {'Hs': Ad_Hs_mean, 'Hm': Ad_Hm_mean, 'I': Ad_ICU_mean}

    for comp, spec in surge_specs.items():
        xs = compute_equilibrium_data(
            fixed_params,
            arrivals_mean,
            spec['amp'] if comp == 'Hs' else Ad_Hs_mean,
            spec['amp'] if comp == 'Hm' else Ad_Hm_mean,
            spec['amp'] if comp == 'I'  else Ad_ICU_mean,
            At_Hs_mean, At_Hm_mean, At_ICU_mean
        )

        x_step = np.array([xs['Hs'], xs['Hm'], xs['I']])
        delta_eq[comp] = x_step - x0

    # ---- Time evolution ----
    for k, t in enumerate(times):
        z = np.zeros(3)

        for comp, spec in surge_specs.items():
            t_on, t_off = spec['on'], spec['off']
            d_eq = delta_eq[comp]

            if t < t_on:
                continue

            elif t_on <= t <= t_off:
                tau = t - t_on
                z += (np.eye(3) - expm(J * tau)).dot(d_eq)

            else:
                tau_surge = t_off - t_on
                z_T = (np.eye(3) - expm(J * tau_surge)).dot(d_eq)
                z += expm(J * (t - t_off)).dot(z_T)

        x_ts[k, :] = x0 + z

    # ---- Extra bed-days ----
    extra_beddays = np.trapz(np.sum(x_ts - x0, axis=1), times)
    extra_beddays_per_comp = {
        'Hs': np.trapz(x_ts[:, 0] - x0[0], times),
        'Hm': np.trapz(x_ts[:, 1] - x0[1], times),
        'I' : np.trapz(x_ts[:, 2] - x0[2], times)
    }

    return {
        'times': times,
        'x_ts': x_ts,
        'x0': x0,
        'extra_beddays_total': extra_beddays,
        'extra_beddays_per_comp': extra_beddays_per_comp,
        'eigvals': eigvals
    }





T_surge=5
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



def ed_ode1(y, t, params, surge_windows):
    W, S, B_Hs, B_Hm, B_I, Hs, Hm, I, D = y

    ad_hs_surge, ad_hm_surge, ad_icu_surge = params

    # Default (baseline)
    ad_hs = Ad_Hs_mean
    ad_hm = Ad_Hm_mean
    ad_icu = Ad_ICU_mean

    # Apply independent surges
    if surge_windows['Hs'][0] <= t <= surge_windows['Hs'][1]:
        ad_hs = ad_hs_surge

    if surge_windows['Hm'][0] <= t <= surge_windows['Hm'][1]:
        ad_hm = ad_hm_surge

    if surge_windows['I'][0] <= t <= surge_windows['I'][1]:
        ad_icu = ad_icu_surge

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



############################
surge_windows = {
    'Hs': [(3, 7)],
    'Hm': [(1, 5), (10, 20)],
    'I' : [(3, 4)]
}

def is_in_any_window(t, windows):
    return any(t0 <= t <= t1 for (t0, t1) in windows)

def active_surge_amplitude(t, windows, baseline):
    """
    Returns surge amplitude if t is in any window,
    otherwise returns baseline.
    If multiple windows overlap, amplitudes are summed.
    """
    amp = baseline
    for (t0, t1, a) in windows:
        if t0 <= t <= t1:
            amp += (a - baseline)
    return amp


def ed_ode2(y, t, surge_specs):
    W, S, B_Hs, B_Hm, B_I, Hs, Hm, I, D = y

    # Time-dependent admissions
    ad_hs = active_surge_amplitude(t, surge_specs['Hs'], Ad_Hs_mean)
    ad_hm = active_surge_amplitude(t, surge_specs['Hm'], Ad_Hm_mean)
    ad_icu = active_surge_amplitude(t, surge_specs['I'],  Ad_ICU_mean)

    sigma = fixed_params['sigma']
    omega = fixed_params['omega']
    gamma = fixed_params['gamma']

    pED_Hs = fixed_params['pED_Hs']
    pED_Hm = fixed_params['pED_Hm']
    pED_ICU = fixed_params['pED_ICU']

    xi_Hs = fixed_params['xi_Hs']
    xi_Hm = fixed_params['xi_Hm']
    xi_I  = fixed_params['xi_I']

    varphi_I = fixed_params['varphi_I']
    varphi_D = fixed_params['varphi_D']
    varphi_Hm = fixed_params['varphi_Hm']

    eps_Hs = fixed_params['eps_Hs']
    eps_Hm = fixed_params['eps_Hm']
    eps_D  = fixed_params['eps_D']

    psi_I = fixed_params['psi_I']
    psi_D = fixed_params['psi_D']

    lam = arrivals_mean

    dW = lam - (sigma + omega) * W
    dS = sigma * W - gamma * S

    dB_Hs = pED_Hs * gamma * S - xi_Hs * B_Hs
    dB_Hm = pED_Hm * gamma * S - xi_Hm * B_Hm
    dB_I  = pED_ICU * gamma * S - xi_I * B_I

    Admissions_Hs = xi_Hs * B_Hs + ad_hs + At_Hs_mean
    Admissions_Hm = xi_Hm * B_Hm + ad_hm + At_Hm_mean
    Admissions_I  = xi_I  * B_I  + ad_icu + At_ICU_mean

    dHs = Admissions_Hs + eps_Hs * I - (varphi_I + varphi_D + varphi_Hm) * Hs
    dHm = Admissions_Hm + eps_Hm * I + varphi_Hm * Hs - (psi_I + psi_D) * Hm
    dI  = Admissions_I  + varphi_I * Hs + psi_I * Hm - (eps_Hs + eps_Hm + eps_D) * I
    dD  = varphi_D * Hs + psi_D * Hm + eps_D * I

    return [dW, dS, dB_Hs, dB_Hm, dB_I, dHs, dHm, dI, dD]


def transient_response_for_multi_surge(surge_specs, times):

    # Baseline equilibrium
    eq = compute_equilibrium_data(
        fixed_params, arrivals_mean,
        Ad_Hs_mean, Ad_Hm_mean, Ad_ICU_mean,
        At_Hs_mean, At_Hm_mean, At_ICU_mean
    )

    x0 = np.array([eq['Hs'], eq['Hm'], eq['I']])

    J_full, eigvals = jacobian_at_equilibrium(fixed_params)
    J = J_full[:3, :3]

    x_ts = np.zeros((len(times), 3))
    x_ts[0] = x0.copy()

    # Precompute equilibrium shifts per surge event
    surge_deltas = []

    for comp, windows in surge_specs.items():
        for (t_on, t_off, amp) in windows:
            xs = compute_equilibrium_data(
                fixed_params,
                arrivals_mean,
                amp if comp == 'Hs' else Ad_Hs_mean,
                amp if comp == 'Hm' else Ad_Hm_mean,
                amp if comp == 'I'  else Ad_ICU_mean,
                At_Hs_mean, At_Hm_mean, At_ICU_mean
            )
            x_step = np.array([xs['Hs'], xs['Hm'], xs['I']])
            surge_deltas.append((t_on, t_off, x_step - x0))

    # Superposition of linear responses
    for k, t in enumerate(times):
        z = np.zeros(3)

        for (t_on, t_off, delta_eq) in surge_deltas:
            if t < t_on:
                continue

            elif t_on <= t <= t_off:
                tau = t - t_on
                z += (np.eye(3) - expm(J * tau)).dot(delta_eq)

            else:
                tau_s = t_off - t_on
                z_T = (np.eye(3) - expm(J * tau_s)).dot(delta_eq)
                z += expm(J * (t - t_off)).dot(z_T)

        x_ts[k] = x0 + z

    return {
        'times': times,
        'x_ts': x_ts,
        'x0': x0,
        'eigvals': eigvals
    }


if __name__ == '__main__':

    # --------------------------------------------------
    # 1) Define surge events (MULTIPLE per compartment)
    #    Each tuple = (start_day, end_day, amplitude)
    # --------------------------------------------------
    surge_specs = {
        'Hs': [
            (3, 7,  1.2 * Ad_Hs_mean),
            (8, 10, 1.5 * Ad_Hs_mean)
        ],
        'Hm': [
            (1, 5,  2.0 * Ad_Hm_mean),
            (10, 20, 1.5 * Ad_Hm_mean)
        ],
        'I': [
            (3, 4,  1.3 * Ad_ICU_mean)
        ]
    }

    # --------------------------------------------------
    # 2) Time grid (SHARED by ODE and analytic solution)
    # --------------------------------------------------
    dt = 0.5
    t_end = max(w[1] for comp in surge_specs.values() for w in comp) + 80
    times = np.arange(0, t_end + dt, dt)

    # --------------------------------------------------
    # 3) Initial condition from equilibrium
    # --------------------------------------------------
    y0_dict = compute_equilibrium_data(
        fixed_params, arrivals_mean,
        Ad_Hs_mean, Ad_Hm_mean, Ad_ICU_mean,
        At_Hs_mean, At_Hm_mean, At_ICU_mean
    )
    y0 = np.array(list(y0_dict.values()))

    # --------------------------------------------------
    # 4) Solve FULL NONLINEAR ODE
    # --------------------------------------------------
    mu = odeint(
        ed_ode2,
        y0,
        times,
        args=(surge_specs,),
        rtol=1e-6,
        atol=1e-6,
        mxstep=5000
    )

    mu_Hs = mu[:, 5]
    mu_Hm = mu[:, 6]
    mu_I  = mu[:, 7]

    # --------------------------------------------------
    # 5) Solve LINEARIZED ANALYTIC MODEL
    # --------------------------------------------------
    res = transient_response_for_multi_surge(surge_specs, times)
    x_ts = res['x_ts']

    # --------------------------------------------------
    # 6) Plot comparison
    # --------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    labels = ['Hs', 'Hm', 'ICU']
    analytic = [x_ts[:, 0], x_ts[:, 1], x_ts[:, 2]]
    ode_sol  = [mu_Hs, mu_Hm, mu_I]

    for i, ax in enumerate(axes):
        ax.plot(times, analytic[i], label=f'{labels[i]} (analytic)', lw=2)
        ax.plot(times, ode_sol[i], '--', label=f'{labels[i]} (ODE)', lw=2, alpha=0.8)
        ax.set_ylabel('Beds')
        ax.legend()
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel('Time (days)')
    fig.suptitle('Multi-surge validation: analytic vs ODE', fontsize=16)
    plt.tight_layout()
    plt.show()




###########################


# t = np.arange(94)
#
#
#
# if __name__ == '__main__':
#
#     surge_specs = {
#         'Hm': {'on': 0, 'off': 10, 'amp': 2 * Ad_Hm_mean},
#         'Hs': {'on': 3, 'off': 7,  'amp': 1 * Ad_Hs_mean},
#         'I' : {'on': 3, 'off': 4,  'amp': 1 * Ad_ICU_mean}}
#
#     surge_windows = {
#         'Hs': (surge_specs['Hs']['on'], surge_specs['Hs']['off']),
#         'Hm': (surge_specs['Hm']['on'], surge_specs['Hm']['off']),
#         'I' : (surge_specs['I']['on'],  surge_specs['I']['off'])
#     }
#
#     params = ( surge_specs['Hs']['amp'], surge_specs['Hm']['amp'], surge_specs['I']['amp'])
#
#     dt = 0.5
#     t_end = max(spec['off'] for spec in surge_specs.values()) + 80
#     times = np.arange(0, t_end + dt, dt)
#
#     # Initial condition from equilibrium
#     y0 = compute_equilibrium_data(
#         fixed_params, arrivals_mean,
#         Ad_Hs_mean, Ad_Hm_mean, Ad_ICU_mean,
#         At_Hs_mean, At_Hm_mean, At_ICU_mean
#     )
#     y0 = np.array(list(y0.values()))
#
#     t = times  # same grid as analytic solution
#
#     mu = odeint(
#         ed_ode1,
#         y0,
#         times,
#         args=(params, surge_windows),
#         rtol=1e-6,
#         atol=1e-6,
#         mxstep=5000
#     )
#
#     mu_Hs = mu[:, 5]
#     mu_Hm = mu[:, 6]
#     mu_I  = mu[:, 7]
#     res = transient_response_for_multi_surge(surge_specs, times)
#     #res = transient_response_for_multi_surge(surge_specs, dt=0.5)
#
#     times = res['times']
#     x_ts = res['x_ts']
#
#
#     fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
#     labels = ['Hs', 'Hm', 'ICU']
#     analytic = [x_ts[:, 0], x_ts[:, 1], x_ts[:, 2]]
#     ode_sol  = [mu_Hs, mu_Hm, mu_I]
#     #baseline = x0
#
#     for i, ax in enumerate(axes):
#         ax.plot(times, analytic[i], label=f'{labels[i]} (analytic)', lw=2)
#         ax.plot(times, ode_sol[i], '--', label=f'{labels[i]} (ODE)', lw=2, alpha=0.8)
#         #ax.axhline(baseline[i], color='gray', linestyle=':', label='Baseline')
#         ax.set_ylabel('Beds')
#         ax.legend()
#         ax.grid(alpha=0.3)
#
#     axes[-1].set_xlabel('Time (days)')
#     fig.suptitle('Multi-surge validation: analytic vs ODE', fontsize=14)
#     plt.tight_layout()
#     plt.show()




# if __name__ == '__main__':
#     #---------------------- Transient response for surge
#     params = (1 * Ad_Hs_mean, 2 * Ad_Hm_mean, 1 * Ad_ICU_mean)
#     res = transient_response_for_surge(ad_hs= 1*Ad_Hs_mean, ad_hm= 2 * Ad_Hm_mean, ad_icu= 1 * Ad_ICU_mean, T_surge=T_surge, dt=0.5)
#
#     times = res['times'];
#     x_ts = res['x_ts'];
#     x0 = res['x0']
#
#     #y0 = [10, 10, 5, 5, 2, x0[0], x0[1],x0[2],1]
#
#     y0 = compute_equilibrium_data(fixed_params, arrivals_mean,
#                                            Ad_Hs_mean, Ad_Hm_mean, Ad_ICU_mean,
#                                            At_Hs_mean, At_Hm_mean, At_ICU_mean)
#     y0 = np.array(list(y0.values()))
#
#
#     mu = odeint(ed_ode, y0, t, hmax=0.1, mxstep=5000, rtol=1e-6, atol=1e-6, args=(params,))
#
#     mu_Hs = mu[:, 5]
#     mu_Hm = mu[:, 6]
#     mu_I = mu[:, 7]
#
#     fig = plt.figure(figsize=(12, 10))
#     ax1 = plt.subplot(3, 1, 1)  # 3 rows, 1 column, position 1
#     ax1.plot(times, x_ts[:, 0], label='Hs (analytic)', color='blue', linewidth=2)
#     ax1.plot(t, mu_Hs, label='Hs (ODE)', color='red', linestyle='--', linewidth=2, alpha=0.7)
#     ax1.axhline(x0[0], color='gray', linestyle=':', label='Baseline', alpha=0.7)
#     ax1.set_ylabel('Occupancy (beds)', fontsize=12)
#     ax1.legend(loc='best')
#
#     ax2 = plt.subplot(3, 1, 2)  # 3 rows, 1 column, position 2
#     ax2.plot(times, x_ts[:, 1], label='Hm (analytic)', color='blue', linewidth=2)
#     ax2.plot(t, mu_Hm, label='Hm (ODE)', color='red', linestyle='--', linewidth=2, alpha=0.7)
#     ax2.axhline(x0[1], color='gray', linestyle=':', label='Baseline', alpha=0.7)
#     ax2.legend(loc='best')
#
#     ax3 = plt.subplot(3, 1, 3)  # 3 rows, 1 column, position 3
#     ax3.plot(times, x_ts[:, 2], label='ICU (analytic)', color='blue', linewidth=2)
#     ax3.plot(t, mu_I, label='ICU (ODE)', color='red', linestyle='--', linewidth=2, alpha=0.7)
#     ax3.axhline(x0[2], color='gray', linestyle=':', label='Baseline', alpha=0.7)
#     ax3.legend(loc='best')
#     ax3.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.subplots_adjust(top=0.93)  # Make room for the suptitle
#
#     plt.show()
#
#     # Print cumulative extra bed-days
#     print('Extra bed-days (total, across Hs+Hm+I approx):', res['extra_beddays_total'])
#     print('Extra bed-days per compartment:', res['extra_beddays_per_comp'])

    #--------
    #cap_med_surg = int((df['TTL_BEDS_MED_SURG_TELE'] - df['UNAVBL_BEDS_MED_SURG_TELE']).median())

