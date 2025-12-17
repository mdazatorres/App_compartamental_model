# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.colors import TwoSlopeNorm
# NOTE: Replace the following two lines by the actual import that gives you `df`, `fixed_params`, and the functions.
from model_full_equilibrium_points import (
    procces_data, compute_params_from_df, compute_equilibrium_data,transient_response_for_multi_surge,
     transient_response_for_surge,fixed_params, jacobian_at_equilibrium, Ad_Hs_mean, Ad_Hm_mean, Ad_ICU_mean, df
)
# ---- END initialization/import ----

plt.rcParams.update({'font.size': 18,        # general font size
    'axes.labelsize': 22,   # x and y labels
    'axes.titlesize': 22,   # title size
    'xtick.labelsize': 20,  # x tick labels
    'ytick.labelsize': 20,  # y tick labels
    'legend.fontsize': 22  # legend text
})


st.set_page_config(page_title="Hospital Equilibrium Explorer", layout="wide")

st.title("Hospital Equilibrium Explorer")

# --- load data & fixed params (should be fast; we assume procces_data is deterministic) ---
# init_day='2024-01-01'
@st.cache_data
def load_data(init_day, end_day):
    df = procces_data(init_day=init_day, end_day=end_day)
    fixed_params = compute_params_from_df(df)
    return df, fixed_params

# Sidebar controls for selecting mode
st.sidebar.header("Controls")
mode = st.sidebar.radio("Choose view", ["Equilibrium", "Transient surge", "Elasticity heatmap"])

# ---------- EQUILIBRIUM PAGE ----------
if mode == "Equilibrium":
    st.header("Baseline equilibrium and observed data")
    st.markdown("Baseline equilibrium computed from weekly means of arrivals/admissions and fixed parameters.")

    date_range = st.slider( "Select date range",
        min_value=dt.date(2020, 1, 1),
        max_value=dt.date(2025, 3, 31),
        value=(dt.date(2020, 1, 1), dt.date(2025, 3, 31)),
        format="YYYY-MM-DD")
    init_day, end_day = date_range
    init_day_str = init_day.strftime("%Y-%m-%d")
    end_day_str = end_day.strftime("%Y-%m-%d")

    df, fixed_params = load_data(init_day_str, end_day_str)

    arrivals = df['DAILY_ED_ARRIVALS'].values
    T = len(arrivals)
    t_data = np.arange(len(arrivals))

    # W, S, B_Hs, B_Hm, B_I, Hs, Hm, I, D
    y0 = [10, 10, 5, 5, 2, df['OCC_BEDS_IP_SURGE'].iloc[0] if len(df) > 0 else 10,
          df['OCC_BEDS_MED_SURG_TELE'].iloc[0] if len(df) > 0 else 20, df['OCC_BEDS_ICU'].iloc[0] if len(df) > 0 else 5, 0]

    obs_Hs = df['OCC_BEDS_IP_SURGE'].values
    obs_Hm = df['OCC_BEDS_MED_SURG_TELE'].values
    obs_ICU = df['OCC_BEDS_ICU'].values

    obs_Dis_Hs = df['Discharges_IP_Surge'].values
    obs_Dis_Hm = df['Discharges_MED_SURG_TELE'].values
    obs_Dis_ICU = df['Discharges_ICU'].values

    obs_ED_adm_Hs = df['ED_Admit_IP_Surge'].values
    obs_ED_adm_Hm = df['ED_Admit_MED_SURG_TELE'].values
    obs_ED_adm_ICU = df['ED_Admit_ICU'].values

    arrivals_mean = df['DAILY_ED_ARRIVALS'].values.mean()
    Ad_Hs_mean = df['DIRECT_Admt_IP_Surge'].values.mean()
    Ad_Hm_mean = df['DIRECT_Admt_MED_SURG_TELE'].values.mean()
    Ad_ICU_mean = df['DIRECT_Admt_ICU'].values.mean() + 0.5
    At_Hs_mean = df['TRNSFR_ADMT_IP_Surge'].values.mean()
    At_Hm_mean = df['TRNSFR_ADMT_MED_SURG_TELE'].values.mean()
    At_ICU_mean = df['TRNSFR_ADMT_ICU'].values.mean()

    # baseline equilibrium
    baseline_eq = compute_equilibrium_data(fixed_params, arrivals_mean, Ad_Hs_mean, Ad_Hm_mean, Ad_ICU_mean,
                                       At_Hs_mean, At_Hm_mean, At_ICU_mean)

    cap_ICU = int((df['TTL_BEDS_ICU'] - df['UNAVBL_BEDS_ICU']).median())
    cap_med_surg = int((df['TTL_BEDS_MED_SURG_TELE'] - df['UNAVBL_BEDS_MED_SURG_TELE']).median())
    cap_IP_surge = int((df['TTL_BEDS_IP_SURGE'] - df['UNAVBL_BEDS_IP_SURGE']).median())
    #st.subheader("Observed occupancy time series with equilibrium lines")
    ################
    fig = make_subplots( rows=3, cols=1, subplot_titles=('IP Surge (Hs)', 'MED/SURG/TELE (Hm)', 'ICU (I)'),
        shared_xaxes=True, vertical_spacing=0.05)

    dates = df['Date']

    # Hs plot
    fig.add_trace(go.Scatter(x=dates, y=df['OCC_BEDS_IP_SURGE'], mode='lines', name='Observed Hs', line=dict(color='blue')),
        row=1, col=1 )
    fig.add_hline(y=baseline_eq['Hs'], line_dash="dash", line_color="red",
                  annotation_text=f"Equilibrium Hs: {baseline_eq['Hs']:.2f}", row=1, col=1)
    fig.add_hline(y=cap_IP_surge, line_dash="dot", line_color="orange",
                  annotation_text=f"Capacity: {cap_IP_surge}", row=1, col=1)
    # Hm plot
    fig.add_trace(go.Scatter(x=dates, y=df['OCC_BEDS_MED_SURG_TELE'], mode='lines', name='Observed Hm', line=dict(color='green')),
        row=2, col=1)
    fig.add_hline(y=baseline_eq['Hm'], line_dash="dash", line_color="red",
                  annotation_text=f"Equilibrium Hm: {baseline_eq['Hm']:.2f}",
                  row=2, col=1)
    fig.add_hline(y=cap_med_surg, line_dash="dot", line_color="orange",
                  annotation_text=f"Capacity: {cap_med_surg}",
                  row=2, col=1)
    # I plot
    fig.add_trace(go.Scatter(x=dates, y=df['OCC_BEDS_ICU'], mode='lines', name='Observed I', line=dict(color='purple')),
        row=3, col=1)
    fig.add_hline(y=baseline_eq['I'], line_dash="dash", line_color="red",
                  annotation_text=f"Equilibrium I: {baseline_eq['I']:.2f}",
                  row=3, col=1)
    fig.add_hline(y=cap_ICU, line_dash="dot", line_color="orange",
                  annotation_text=f"Capacity: {cap_ICU}",
                  row=3, col=1)

    fig.update_layout(height=800, showlegend=True, title_text=f"Observed Occupancy Time Series (Data from {init_day_str} to {end_day_str})")
    fig.update_yaxes(title_text="Occupancy", row=1, col=1)
    fig.update_yaxes(title_text="Occupancy", row=2, col=1)
    fig.update_yaxes(title_text="Occupancy", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)


    fig = make_subplots(rows=3, cols=1, subplot_titles=('IP Surge (Hs)', 'MED/SURG/TELE (Hm)', 'ICU (I)'),
                        shared_xaxes=True, vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=dates, y=df['ED_Admit_IP_Surge'], mode='lines', name='Observed ED admissions Hs', line=dict(color='blue')),
                  row=1, col=1)
    fig.add_hline(y=fixed_params['pED_Hs']*fixed_params['gamma']*baseline_eq['S'], line_dash="dash", line_color="red",
                  annotation_text=f"Equilibrium Admit Hs: {fixed_params['pED_Hs']*fixed_params['gamma']*baseline_eq['S']:.2f}",
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df['ED_Admit_MED_SURG_TELE'], mode='lines', name='Observed ED admissions Hm',
                             line=dict(color='green')),row=2, col=1)
    fig.add_hline(y=fixed_params['pED_Hm'] * fixed_params['gamma'] * baseline_eq['S'], line_dash="dash",
                  line_color="red",
                  annotation_text=f"Equilibrium Admit Hm: {fixed_params['pED_Hm'] * fixed_params['gamma'] * baseline_eq['S']:.2f}",
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df['ED_Admit_ICU'], mode='lines', name='Observed ED admissions ICU',
                             line=dict(color='purple')), row=3, col=1)
    fig.add_hline(y=fixed_params['pED_ICU'] * fixed_params['gamma'] * baseline_eq['S'], line_dash="dash",
                  line_color="red",
                  annotation_text=f"Equilibrium Admit ICU: {fixed_params['pED_ICU'] * fixed_params['gamma'] * baseline_eq['S']:.2f}",
                  row=3, col=1)
    fig.update_layout(height=800, showlegend=True, title_text=f"Observed ED Admissions Time Series (Data from {init_day_str} to {end_day_str})")
    st.plotly_chart(fig, use_container_width=True)

    # Update layout
    st.subheader("Equilibrium values")
    eq_table = pd.DataFrame({
        "compartment": list(baseline_eq.keys()),
        "value": [baseline_eq[k] for k in baseline_eq.keys()]
    })
    st.dataframe(eq_table.style.format({"value": "{:.2f}"}), height=300)



# ---------- SURGE SWEEP PAGE ----------
elif mode=="Transient surge":

    #####################
    st.subheader("📈 Surge Scenarios")

    surge_specs = {'Hs': [], 'Hm': [], 'I': []}

    dt = 0.5

    for comp, label, base in [
        ('Hs', 'IP (Hs)', Ad_Hs_mean),
        ('Hm', 'Medical (Hm)', Ad_Hm_mean),
        ('I', 'ICU (I)', Ad_ICU_mean)
    ]:
        with st.expander(f"{label} surge events"):
            n_events = st.number_input(
                f"Number of {label} surge events",
                min_value=0, max_value=5, value=1, step=1,
                key=f"n_{comp}"
            )

            for k in range(n_events):
                col1, col2, col3 = st.columns(3)

                with col1:
                    t_on = st.number_input(
                        f"{label} surge {k + 1} start",
                        min_value=0.0, max_value=100.0, value=0.0, step=1.0,
                        key=f"{comp}_on_{k}"
                    )
                with col2:
                    t_off = st.number_input(
                        f"{label} surge {k + 1} end",
                        min_value=t_on, max_value=150.0, value=t_on + 5.0, step=1.0,
                        key=f"{comp}_off_{k}"
                    )
                with col3:
                    amp = st.number_input(
                        f"{label} surge {k + 1}: extra admissions per day",
                        min_value=0.0, max_value=50.0, value=1.0, step=1.0,
                        help="Total direct admissions during surge",
                        key=f"{comp}_amp_{k}"
                    )

                surge_specs[comp].append((t_on, t_off, amp))

    if all(len(v) == 0 for v in surge_specs.values()):
        st.warning("No surge events defined.")
        st.stop()

    t_end = max(w[1] for comp in surge_specs.values() for w in comp) + 80
    times = np.arange(0, t_end + dt, dt)



    res = transient_response_for_multi_surge(surge_specs, times)

    x_ts = res['x_ts']
    x0 = res['x0']

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True)

    colors = ['blue', 'green', 'purple']
    labels = ['Hs', 'Hm', 'ICU']

    for i in range(3):
        fig.add_trace(
            go.Scatter(
                x=times,
                y=x_ts[:, i],
                mode='lines',
                name=f"{labels[i]} occupancy",
                line=dict(color=colors[i])
            ),
            row=i + 1, col=1
        )

        fig.add_hline(
            y=x0[i],
            line_dash="dot",
            line_color=colors[i],
            row=i + 1, col=1
        )

    fig.update_xaxes(title_text="Days", row=3, col=1)

    fig.update_layout(
        title="Transient occupancy under multiple surge scenarios",
        yaxis_title="Beds",
        template="plotly_white",
        height=700
    )

    st.plotly_chart(fig, use_container_width=True)

    #####################


    # st.subheader('old option')
    #
    # col1, col2, col3, col4= st.columns(4)
    #
    # with col1:
    #     surge_Ad_Hs = st.number_input( "IP Surge", min_value=1.0, max_value=20.0, value=1.0, step=1.0,help="Increase in IP direct admissions")
    # with col2:
    #     surge_Ad_Hm = st.number_input("Med Surge", min_value=1.0, max_value=20.0, value=1.0, step=1.0,help="Increase in medical ward admissions")
    # with col3:
    #     surge_Ad_ICU = st.number_input( "ICU Surge",min_value=1.0, max_value=20.0, value=1.0, step=1.0, help="Increase in ICU admissions" )
    # with col4:
    #     t_surge = st.number_input("Days of the surge", min_value=0.0, max_value=50.0, value=5.0, step=1.0,
    #                               help="T surge")
    #
    # res=transient_response_for_surge(ad_hs= Ad_Hs_mean+surge_Ad_Hs, ad_hm= Ad_Hm_mean+surge_Ad_Hm, ad_icu= Ad_ICU_mean+surge_Ad_ICU, T_surge=t_surge, dt=0.5)
    #
    # times = res['times'];
    # x_ts = res['x_ts'];
    # x0 = res['x0']
    #
    #
    # fig = make_subplots(rows=3, cols=1)
    # fig.add_trace(go.Scatter(x=times, y=x_ts[:, 0], mode='lines', name='Hs (total)',
    #                          line=dict(color='blue')), row=1, col=1)
    # fig.add_trace(go.Scatter(x=times, y=x_ts[:, 1], mode='lines', name='Hm (total)',
    #                          line=dict(color='green')), row=2, col=1)
    # fig.add_trace(go.Scatter(x=times, y=x_ts[:, 2], mode='lines', name='ICU (I)',
    #                          line=dict(color='purple')), row=3, col=1)
    #
    # #fig.add_vline(x=14, line_width=2, line_dash="dash", line_color="black")
    #
    # # Baseline horizontal lines
    # fig.add_hline(y=x0[0], line_dash="dot", line_color="blue", row=1, col=1)
    # fig.add_hline(y=x0[1], line_dash="dot", line_color="green", row=2, col=1)
    # fig.add_hline(y=x0[2], line_dash="dot", line_color="purple", row=3, col=1)
    #
    # fig.update_xaxes(title_text="Days since surge start", row=3, col=1)
    #
    # fig.update_layout( title="Transient occupancy during admission surge",
    #     yaxis_title="Occupancy (beds)",
    #     template="plotly_white")
    #
    # st.plotly_chart(fig, use_container_width=True)

    extra_beds_over_time = x_ts - x0
    #extra_beddays_per_comps= res['extra_beddays_per_comp']
    peak_extra_beds_total = np.max(np.sum(extra_beds_over_time, axis=1))
    peak_extra_beds_per_comp = {
        'Hs': np.max(extra_beds_over_time[:, 0]),
        'Hm': np.max(extra_beds_over_time[:, 1]),
        'I': np.max(extra_beds_over_time[:, 2])
    }
    total_beds_needed = {
        'Hs': x0[0] + peak_extra_beds_per_comp['Hs'],
        'Hm': x0[1] + peak_extra_beds_per_comp['Hm'],
        'I': x0[2] + peak_extra_beds_per_comp['I'],
        'Total': np.sum(x0) + peak_extra_beds_total
    }

    # --- Output numbers ---
    col1, col2 = st.columns(2)

    with col1:
        st.write("### 🏥 Peak Bed Requirements")
        st.metric("Total Extra Beds", f"{peak_extra_beds_total:.1f}")
        # st.metric("Total Bed days Hs", f"{extra_beddays_per_comps['Hs']:.1f}")
        # st.metric("Total Bed days Hm", f"{extra_beddays_per_comps['Hm']:.1f}")
        # st.metric("Total Bed days ICU", f"{extra_beddays_per_comps['I']:.1f}")

        st.write("**By compartment:**")
        st.write(f"- Hs: {peak_extra_beds_per_comp['Hs']:.1f} extra")
        st.write(f"- Hm: {peak_extra_beds_per_comp['Hm']:.1f} extra")
        st.write(f"- ICU: {peak_extra_beds_per_comp['I']:.1f} extra")

    with col2:
        st.write("### 📊 Total Workload")
        st.metric("Extra Bed-Days (Total)", f"{res['extra_beddays_total']:.1f}")
        st.write("**By compartment:**")
        for comp, beddays in res['extra_beddays_per_comp'].items():
            st.write(f"- {comp}: {beddays:.1f} bed-days")

    st.write("### 👥 Staffing & Resource Planning")

    # Use different variable names for each set of columns
    st.write("#### IP Surge (Hs) ")
    hs_col1, hs_col2, hs_col3 = st.columns(3)
    with hs_col1:
        nurses_per_bed_Hs = st.number_input("Nurses per bed", min_value=0.1, max_value=2.0, value=0.3, step=0.1,
                                            help="Typically 0.3-0.5 nurses per bed (1 nurse handles 2-3 beds)",
                                            key="nurses_hs")
    with hs_col2:
        cost_per_bedday_Hs = st.number_input("Cost per bed-day ($)", min_value=100, max_value=2000, value=800, step=50,
                                             help="Includes staff, supplies, utilities", key="cost_hs")
    with hs_col3:
        shifts_per_day_Hs = st.number_input("Shifts per day", min_value=1, max_value=3, value=2, step=1,
                                            help="Typically 2-3 shifts per day", key="shifts_hs")

    st.write("#### Med Surge Tele (Hm) ")
    hm_col1, hm_col2, hm_col3 = st.columns(3)
    with hm_col1:
        nurses_per_bed_Hm = st.number_input("Nurses per bed", min_value=0.1, max_value=2.0, value=0.3, step=0.1,
                                            help="Typically 0.3-0.5 nurses per bed (1 nurse handles 2-3 beds)",
                                            key="nurses_hm")
    with hm_col2:
        cost_per_bedday_Hm = st.number_input("Cost per bed-day ($)", min_value=100, max_value=2000, value=800, step=50,
                                             help="Includes staff, supplies, utilities", key="cost_hm")
    with hm_col3:
        shifts_per_day_Hm = st.number_input("Shifts per day", min_value=1, max_value=3, value=2, step=1,
                                            help="Typically 2-3 shifts per day", key="shifts_hm")

    st.write("#### ICU ")
    icu_col1, icu_col2, icu_col3 = st.columns(3)
    with icu_col1:
        nurses_per_bed_ICU = st.number_input("Nurses per bed", min_value=0.1, max_value=2.0, value=0.3, step=0.1,
                                             help="Typically 0.3-0.5 nurses per bed (1 nurse handles 2-3 beds)",
                                             key="nurses_icu")
    with icu_col2:
        cost_per_bedday_ICU = st.number_input("Cost per bed-day ($)", min_value=100, max_value=2000, value=800, step=50,
                                              help="Includes staff, supplies, utilities", key="cost_icu")
    with icu_col3:
        shifts_per_day_ICU = st.number_input("Shifts per day", min_value=1, max_value=3, value=2, step=1,
                                             help="Typically 2-3 shifts per day", key="shifts_icu")

    if st.button("Calculate Staffing Needs"):
        # CORRECTED: Use the per-component dictionary, not the total
        extra_nurse_shifts_Hs = res['extra_beddays_per_comp']['Hs'] * nurses_per_bed_Hs * shifts_per_day_Hs
        total_cost_Hs = res['extra_beddays_per_comp']['Hs'] * cost_per_bedday_Hs

        extra_nurse_shifts_Hm = res['extra_beddays_per_comp']['Hm'] * nurses_per_bed_Hm * shifts_per_day_Hm
        total_cost_Hm = res['extra_beddays_per_comp']['Hm'] * cost_per_bedday_Hm

        extra_nurse_shifts_ICU = res['extra_beddays_per_comp']['I'] * nurses_per_bed_ICU * shifts_per_day_ICU
        total_cost_ICU = res['extra_beddays_per_comp']['I'] * cost_per_bedday_ICU

        st.success("**Staffing & Budget Requirements:**")

        st.write("**IP Surge (Hs):**")
        st.write(f"🧑‍⚕️ Extra nurse shifts needed: {extra_nurse_shifts_Hs:.0f}")
        st.write(f"💰 Total surge cost: ${total_cost_Hs:,.0f}")

        st.write("**Med Surge (Hm):**")
        st.write(f"🧑‍⚕️ Extra nurse shifts needed: {extra_nurse_shifts_Hm:.0f}")
        st.write(f"💰 Total surge cost: ${total_cost_Hm:,.0f}")

        st.write("**ICU:**")
        st.write(f"🧑‍⚕️ Extra nurse shifts needed: {extra_nurse_shifts_ICU:.0f}")
        st.write(f"💰 Total surge cost: ${total_cost_ICU:,.0f}")

        st.write(f"📅 **Total extra workload:** {res['extra_beddays_total']:.1f} bed-days")

        # ---- TOTAL COST SUMMATION ----
        total_cost_all = total_cost_Hs + total_cost_Hm + total_cost_ICU

        st.markdown("---")
        st.subheader("💵 Total Surge Cost Summary")
        st.write(f"**Total cost across all units:** ${total_cost_all:,.0f}")
        st.write(f"📅 **Total extra workload:** {res['extra_beddays_total']:.1f} bed-days")

    # Add this section to your Streamlit app
