import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares



def procces_data(init_day='2024-03-31', end_day='2025-03-31'):
    xls = pd.ExcelFile("data/IMPACTS Data Extract_V2.xlsx")
    data_ED = pd.read_excel('data/Patient Surge Model ED Data.xlsx')

    df_capacity = pd.read_excel(xls, sheet_name="Capacity")
    df_OR = pd.read_excel(xls, sheet_name="OR")

    df_adt = pd.read_excel(xls, sheet_name="ADT Summary")
    df_adt_all = pd.read_excel(xls, sheet_name="Admission Breakdown")
    df_adt_all = df_adt_all[df_adt_all.ADMIT_DATE <= '2025-03-31']
    df_ICU_downUp = pd.read_excel(xls, sheet_name='ICU DownUp')

    df_capacity = df_capacity.rename({'DAY_DT': 'Date'}, axis=1)
    df_OR = df_OR.rename({'DateValue': 'Date'}, axis=1)
    df_adt = df_adt.rename({'DateValue': 'Date'}, axis=1)

    df_adt_all = df_adt_all.rename(columns={"ADMIT_DATE": "Date"})
    data_ED = data_ED.rename(columns={'ED_ARRIVAL_IMPUTED_DT': "Date"})
    df_ICU_downUp = df_ICU_downUp.rename({'Transfer_Date': 'Date'}, axis=1)

    df_ICU_downUp = df_ICU_downUp[df_ICU_downUp['Date'] <= '2025-03-31'].copy()

    df = pd.merge(df_adt_all, data_ED, on='Date')
    df = pd.merge(df, df_capacity, on='Date')
    df = pd.merge(df, df_ICU_downUp, on='Date')
    df = pd.merge(df, df_adt, on='Date')
    df = pd.merge(df, df_OR, on='Date')

    df = df.sort_values(by='Date').reset_index(drop=True)
    #df = df[df['Date'] >= '2024-03-31']
    df = df[df['Date'] >= init_day]
    df = df[df['Date'] <= end_day]

    df["AVERAGE_ED_WAITING_INTERVAL_DAYS"] = df["AVERAGE_ED_WAITING_INTERVAL"] / 1440
    df["AVERAGE_ED_BOARDING_INTERVAL_DAYS"] = df["AVERAGE_ED_BOARDING_INTERVAL"] / 1440
    return df


def compute_params_from_df(df):
    MIN2DAY = 1.0 / 1440.0
    mean_wait_min = float(df['AVERAGE_ED_WAITING_INTERVAL'].dropna().mean())
    mean_board_min = float(df['AVERAGE_ED_BOARDING_INTERVAL'].dropna().mean())
    mean_LOS_days = df['AVERAGE_ED_LOS_INTERVAL'].dropna().mean() * MIN2DAY

    mean_wait_days = max(mean_wait_min * MIN2DAY, 1e-6)
    mean_board_days = max(mean_board_min * MIN2DAY, 1e-6)

    sigma = 1.0 / mean_wait_days  # rate to be attended (per day)
    psi = 1.0 / mean_board_days   # boarding -> inpatient transfer rate

    # LWBT fraction and omega:
    total_arr = df['DAILY_ED_ARRIVALS'].sum()
    total_lwbt = df['DAILY_LWBT_COUNT'].sum() if 'DAILY_LWBT_COUNT' in df.columns else 0.0
    f = float(total_lwbt) / total_arr if total_arr > 0 else 0.0
    f = np.clip(f, 0.0, 0.9999)
    omega = sigma * f / max((1.0 - f), 1e-9)

    # p_admit: if admitted series available compute fraction among seen
    cols = [ 'ED_Admit_ICU', 'ED_Admit_MED_SURG_TELE', 'ED_Admit_NICU', 'ED_Admit_Peds',
        'ED_Admit_PICU', 'ED_Admit_WMN_PAV', 'ED_Admit_Obstetrics', 'ED_Admit_IP_Surge']
    # sum available admit columns (skip missing)
    admit_cols = [c for c in cols if c in df.columns]
    total_adm_from_ED = df[admit_cols].sum(axis=1).fillna(0)
    total_adm = total_adm_from_ED.sum()

    seen_total = total_arr - total_lwbt
    if seen_total <= 0:
        p_admit = 0.15  # fallback
    else:
        p_admit = total_adm / seen_total

    pED_Hs = df['ED_Admit_IP_Surge'].sum() / seen_total
    pED_Hm = df['ED_Admit_MED_SURG_TELE'].sum() / seen_total
    pED_ICU = df['ED_Admit_ICU'].sum() / seen_total

    mean_service = mean_LOS_days - mean_wait_days - p_admit * mean_board_days
    gamma = 1.0 / mean_service

    mean_board_Hs = mean_board_days
    mean_board_ICU = mean_board_days* 1  # 0.7
    mean_board_Hm = mean_board_days * 1  # 1.2

    xi_I = 1.0 / max(mean_board_ICU, 1e-6)
    xi_Hm = 1.0 / max(mean_board_Hm, 1e-6)
    xi_Hs = 1.0 / max(mean_board_Hs, 1e-6)

    varphi_I = df['IP_Surge_TO_ICU'].sum() / df['OCC_BEDS_IP_SURGE'].sum()
    varphi_D = df['Discharges_IP_Surge'].sum() / df['OCC_BEDS_IP_SURGE'].sum()
    varphi_Hm = 0.42

    psi_I = df['MED_SURG_TELE_TO_ICU'].sum() / df['OCC_BEDS_MED_SURG_TELE'].sum()
    psi_D = df['Discharges_MED_SURG_TELE'].sum() / df['OCC_BEDS_MED_SURG_TELE'].sum()

    eps_Hs = df['ICU_TO_IP_Surge'].sum() / df['OCC_BEDS_ICU'].sum()
    eps_Hm = df['ICU_TO_MED_SURG_TELE'].sum() / df['OCC_BEDS_ICU'].sum()
    eps_D = df['Discharges_ICU'].sum() / df['OCC_BEDS_ICU'].sum()

    return {'sigma': sigma, 'omega': omega, 'psi': psi,  'gamma': gamma,
            'f_lwbt': f, 'mean_wait_days': mean_wait_days, 'mean_board_days': mean_board_days,
            'pED_Hs':pED_Hs, 'pED_Hm':pED_Hm, 'pED_ICU':pED_ICU, 'xi_I':xi_I, 'xi_Hm':xi_Hm, 'xi_Hs':xi_Hs,
            'varphi_I':varphi_I, 'varphi_D':varphi_D, 'varphi_Hm':varphi_Hm, 'psi_I':psi_I, 'psi_D':psi_D,
            'eps_Hs':eps_Hs, 'eps_Hm':eps_Hm, 'eps_D':eps_D}


def compute_week_parameters(df):
    df['weekday'] = df['Date'].dt.day_name()

    df['varphi_I_daily'] = df['IP_Surge_TO_ICU'] / df['OCC_BEDS_IP_SURGE'].replace(0, np.nan)
    df['varphi_D_daily'] = df['Discharges_IP_Surge'] / df['OCC_BEDS_IP_SURGE'].replace(0, np.nan)

    df['psi_I_daily'] = df['MED_SURG_TELE_TO_ICU'] / df['OCC_BEDS_MED_SURG_TELE'].replace(0, np.nan)
    df['psi_D_daily'] = df['Discharges_MED_SURG_TELE'] / df['OCC_BEDS_MED_SURG_TELE'].replace(0, np.nan)

    df['eps_Hs_daily'] = df['ICU_TO_IP_Surge'] / df['OCC_BEDS_ICU'].replace(0, np.nan)
    df['eps_Hm_daily'] = df['ICU_TO_MED_SURG_TELE'] / df['OCC_BEDS_ICU'].replace(0, np.nan)
    df['eps_D_daily'] = df['Discharges_ICU'] / df['OCC_BEDS_ICU'].replace(0, np.nan)

    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Compute mean per weekday for the rates with weekday effect
    varphi_I_weekday = df.groupby('weekday')['varphi_I_daily'].mean().reindex(weekday_order)
    varphi_D_weekday = df.groupby('weekday')['varphi_D_daily'].mean().reindex(weekday_order)
    psi_D_weekday = df.groupby('weekday')['psi_D_daily'].mean().reindex(weekday_order)
    psi_I_weekday = df.groupby('weekday')['psi_I_daily'].mean().reindex(weekday_order)

    eps_Hm_weekday = df.groupby('weekday')['eps_Hm_daily'].mean().reindex(weekday_order)
    eps_D_weekday = df.groupby('weekday')['eps_D_daily'].mean().reindex(weekday_order)
    eps_Hs_weekday = df.groupby('weekday')['eps_Hs_daily'].mean().reindex(weekday_order)

    return {'varphi_I_weekday': varphi_I_weekday, 'varphi_D_weekday': varphi_D_weekday, 'psi_D_weekday':psi_D_weekday,
            'psi_I_weekday':psi_I_weekday, 'eps_Hm_weekday':eps_Hm_weekday,'eps_D_weekday':eps_D_weekday, 'eps_Hs_weekday':eps_Hs_weekday}


