# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from func_to_plot import plot_occupancy,plot_ED_admissions
from matplotlib.colors import TwoSlopeNorm
# NOTE: Replace the following two lines by the actual import that gives you `df`, `fixed_params`, and the functions.
from model_full_equilibrium_points import (
    procces_data, compute_params_from_df, compute_equilibrium_data,transient_response_for_multi_surge,
     fixed_params, jacobian_at_equilibrium, Ad_Hs_mean, Ad_Hm_mean, Ad_ICU_mean, df)
# ---- END initialization/import ----

plt.rcParams.update({'font.size': 18,        # general font size
    'axes.labelsize': 22,   # x and y labels
    'axes.titlesize': 22,   # title size
    'xtick.labelsize': 20,  # x tick labels
    'ytick.labelsize': 20,  # y tick labels
    'legend.fontsize': 22 })


st.set_page_config(page_title="Hospital Equilibrium Explorer", layout="wide")



# --- load data & fixed params (should be fast; we assume procces_data is deterministic) ---
# init_day='2024-01-01'
@st.cache_data
def load_data(init_day, end_day):
    df = procces_data(init_day=init_day, end_day=end_day)
    fixed_params = compute_params_from_df(df)
    return df, fixed_params

# Sidebar controls for selecting mode
st.sidebar.header("🔧 Analysis Mode")
mode = st.sidebar.radio("Select analysis view", ["Equilibrium", "Transient surge"])

# ---------- EQUILIBRIUM PAGE ----------
if mode == "Equilibrium":
    st.header("🏥 Hospital Capacity & Surge Planning Explorer")
    st.markdown("Model-based analysis of baseline equilibrium, surge scenarios, workload, and resource planning" )

    st.markdown("#### Baseline Equilibrium Analysis")
    st.markdown("Baseline equilibrium computed from weekly means of arrivals/admissions and fixed parameters.")

    #st.markdown("📅 Analysis Period")
    date_range = st.slider( "Select date range for baseline estimation", min_value=dt.date(2020, 1, 1),  max_value=dt.date(2025, 3, 31),
        value=(dt.date(2020, 1, 1), dt.date(2025, 3, 31)), format="YYYY-MM-DD")

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

    #st.subheader("📈 Observed Occupancy vs. Equilibrium")
    dates = df['Date']
    fig = plot_occupancy(df, baseline_eq, cap_IP_surge, cap_med_surg, cap_ICU, init_day_str, end_day_str)
    st.plotly_chart(fig, use_container_width=True)

    fig = plot_ED_admissions(df, fixed_params, baseline_eq,init_day_str,end_day_str)
    st.plotly_chart(fig, use_container_width=True)

    # Update layout
    #st.subheader("⚖️ Equilibrium Occupancy Levels")
    st.markdown("#### Equilibrium Occupancy Levels")
    eq_table = pd.DataFrame({ "compartment": list(baseline_eq.keys()), "value": [baseline_eq[k] for k in baseline_eq.keys()] })
    st.dataframe(eq_table.style.format({"value": "{:.2f}"}), height=300)




else : #mode=="Transient surge":
    # ---------- SURGE scenario selection ----------
    st.subheader("📈 Surge Scenarios Analysis")
    st.markdown("Specify the timing, duration, and intensity of admission surges by unit." )
    surge_specs = {'Hs': [], 'Hm': [], 'I': []}

    dt = 1.0
    for comp, label, base in [('Hs', 'IP (Hs)', Ad_Hs_mean), ('Hm', 'Medical (Hm)', Ad_Hm_mean),
                              ('I', 'ICU (I)', Ad_ICU_mean)]:
        # Number of surge events for this unit
        with st.expander(f"{label} surge events"):
            n_events = st.number_input(f"Number of {label} surge events",
                min_value=0, max_value=5, value=1, step=1, key=f"n_{comp}")

        # Collect parameters for each surge event
            for k in range(n_events):
                col1, col2, col3 = st.columns(3)
                with col1:  # Surge start time (days)
                    t_on = st.number_input( f"{label} surge {k + 1} start",
                        min_value=0.0, max_value=100.0, value=0.0, step=1.0, key=f"{comp}_on_{k}")
                with col2:  # Surge end time (days)
                    t_off = st.number_input(f"{label} surge {k + 1} end",
                        min_value=t_on, max_value=150.0, value=t_on + 5.0, step=1.0, key=f"{comp}_off_{k}")
                with col3:  # Surge amplitude (extra admissions per day)
                    amp = st.number_input(
                        f"{label} surge {k + 1}: extra admissions per day", min_value=0.0, max_value=50.0, value=1.0, step=1.0,
                        help="Total direct admissions during surge", key=f"{comp}_amp_{k}")

                surge_specs[comp].append((t_on, t_off, amp))   # Store surge event
    # Stop execution if no surge events are defined across all units
    if all(len(v) == 0 for v in surge_specs.values()):
        st.warning("No surge events defined.")
        st.stop()

    # --------------------------------------------------------------------
    # Determine simulation horizon:
    # take the latest surge end time across all units and extend it by 70 days to allow the system to relax back to equilibrium
    t_end = max(w[1] for comp in surge_specs.values() for w in comp) + 70
    times = np.arange(0, t_end + dt, dt) # Time grid for simulation (daily resolution)

    # Compute transient system response to multiple surge events
    res = transient_response_for_multi_surge(surge_specs, times)

    x_ts = res['x_ts'] # Time series of state variables (beds by unit)
    x0 = res['x0']     # Baseline equilibrium (no-surge steady state)

    extra_beds_over_time = x_ts - x0 # Compute daily excess bed occupancy relative to baseline

    # We define "active surge impact" as any time when at least one unit
    # exceeds the baseline by more than the specified threshold.
    threshold = 0.1  # minimum extra beds considered meaningful

    max_extra = np.max(extra_beds_over_time, axis=1) # Maximum excess occupancy across units at each time point

    active_idx = np.where(max_extra >= threshold)[0] # Indices where the system is still above the threshold

    # Last time index with meaningful surge impact
    if len(active_idx) > 0:
        t_cut = active_idx[-1] + 1
    else:
        t_cut = 1   # No meaningful deviation from equilibrium detected

    # --------------------------------------------------------------------
    # Truncate the time series to exclude periods where excess occupancy is negligible (< threshold).
    # All downstream workload and cost calculations are based on this truncated series.
    times_plot = times[:t_cut]
    x_ts_plot= x_ts[:t_cut]

    # Plot transient bed occupancy by unit, together with the baseline (no-surge) equilibrium level for reference.
    # fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
    # colors = ['blue', 'green', 'purple']; labels = ['Hs', 'Hm', 'ICU']
    # for i in range(3):
    #     fig.add_trace(go.Scatter( x=times_plot, y=x_ts_plot[:, i],  mode='lines', name=f"{labels[i]} occupancy", line=dict(color=colors[i]) ), row=i + 1, col=1 )
    #     fig.add_hline(y=x0[i], line_dash="dot",line_color=colors[i], row=i + 1, col=1 )
    # fig.update_xaxes(title_text="Days", row=3, col=1)
    # fig.update_layout( title="Transient Bed Occupancy Under Surge Scenarios", yaxis_title="Beds",  template="plotly_white", height=500 )
    # st.plotly_chart(fig, use_container_width=True)

    #########
    from plotly.subplots import make_subplots
    #def occupancy_under_scenarios
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        specs=[[{"secondary_y": True}],
               [{"secondary_y": True}],
               [{"secondary_y": True}]]
    )

    colors = ['blue', 'green', 'purple']
    labels = ['Hs', 'Hm', 'ICU']

    extra_beds = x_ts - x0  # extra beds relative to equilibrium
    extra_beds_plot= extra_beds[:t_cut]
    for i in range(3):
        # --- absolute occupancy (left axis) ---
        fig.add_trace(go.Scatter(x=times_plot, y=x_ts_plot[:, i], mode="lines", name=f"{labels[i]} occupancy",
                line=dict(color=colors[i])), row=i + 1, col=1,  secondary_y=False )

        # equilibrium line
        fig.add_hline( y=x0[i], line_dash="dot", line_color=colors[i],  row=i + 1, col=1 )

        # --- extra beds (right axis, starts at 0) ---
        fig.add_trace( go.Scatter(x=times_plot, y=extra_beds_plot[:, i], mode="lines",
                name=f"{labels[i]} extra beds",line=dict(color=colors[i], dash="dash"), visible="legendonly",
                showlegend=False ), row=i + 1,col=1, secondary_y=True)

        # --- Extra beds (bars = discrete daily beds) ---
        fig.add_trace(go.Bar( x=times_plot,
                y=extra_beds_plot[:, i],
                name=f"{labels[i]} extra beds (daily)",
                marker_color=colors[i],
                opacity=0.35,
                showlegend=False
            ),
            row=i + 1, col=1, secondary_y=True
        )

        # right axis formatting
        fig.update_yaxes( title_text="Extra beds", secondary_y=True,
            row=i + 1, col=1,rangemode="tozero" )



    # axis labels
    fig.update_xaxes(title_text="Days", row=3, col=1)
    fig.update_yaxes(title_text="Beds", secondary_y=False)
    fig.update_layout(
        title="Transient Bed Occupancy Under Surge Scenarios",
        template="plotly_white",
        height=600,
        legend=dict(orientation="h", y=-0.15))

    st.plotly_chart(fig, use_container_width=True)

    ########

    # --------------------------------------------------------------------
    # Compute peak excess bed demand:
    #  total peak across all units,  unit-specific peaks for capacity planning

    peak_extra_beds_total = np.max(np.sum(extra_beds_over_time, axis=1))
    peak_extra_beds_per_comp = {
        'Hs': np.max(extra_beds_over_time[:, 0]),
        'Hm': np.max(extra_beds_over_time[:, 1]),
        'I': np.max(extra_beds_over_time[:, 2]) }

    # --- Output summary statistics ---
    col1, col2 = st.columns(2)

    with col1:
        st.write("### 🏥 Peak Additional Bed Requirements")
        st.metric("Total Extra Beds", f"{peak_extra_beds_total:.1f}")
        st.write("**By compartment:**")
        st.write(f"- Hs: {peak_extra_beds_per_comp['Hs']:.1f} extra")
        st.write(f"- Hm: {peak_extra_beds_per_comp['Hm']:.1f} extra")
        st.write(f"- ICU: {peak_extra_beds_per_comp['I']:.1f} extra")

    with col2:
        st.write("### 📊 Cumulative Bed-Days (Total Workload)")
        extra_beds_cut = extra_beds_over_time[:t_cut]
        extra_beddays_per_comp_cut = {comp: extra_beds_cut[:, j].sum() for j, comp in enumerate(["Hs", "Hm", "I"])}
        extra_beddays_total_cut = sum(extra_beddays_per_comp_cut.values())
        #### look here
        res["extra_beddays_per_comp_cut"] = extra_beddays_per_comp_cut
        res["extra_beddays_total_cut"] = extra_beddays_total_cut

        st.metric("Extra Bed-Days (Total)", f"{res["extra_beddays_total_cut"]:.1f}")
        st.write("**By compartment:**")
        for comp, beddays in res["extra_beddays_per_comp_cut"].items():
            st.write(f"- {comp}: {beddays:.1f} bed-days")


    st.write("### 🗓️ Weekly Bed-Day Workload")
    st.caption("Weekly aggregation of extra occupied beds (bed-days)")

    extra_beds_over_time_cut = x_ts[:t_cut] - x0
    n_days, n_units = extra_beds_over_time_cut.shape
    n_weeks = int(np.ceil(n_days / 7))

    # pad with zeros so days is a multiple of 7
    pad = n_weeks * 7 - n_days
    extra_padded = np.pad(extra_beds_over_time_cut,((0, pad), (0, 0)), mode="constant")

    weeks_workload = extra_padded.reshape(n_weeks, 7, n_units).sum(axis=1)
    week_labels = [f"Week {i + 1}" for i in range(n_weeks)]; unit_labels = ["Hs", "Hm", "ICU"]

    df_weeks = pd.DataFrame( weeks_workload,index=week_labels,columns=unit_labels )

    # total per week (row-wise)
    df_weeks["Total"] = df_weeks.sum(axis=1)
    df_weeks.loc["Total"] = df_weeks.sum(axis=0)

    st.dataframe(df_weeks)

    ############
    st.write("### 👥 Staffing & Resource Planning")
    st.caption("Estimate staffing needs and operational costs based on cumulative bed-day demand")

    st.write("#### Nursing Assumptions by Unit")
    st.write("##### IP Surge (Hs) ")
    hs_col1, hs_col2, hs_col3 = st.columns(3)
    with hs_col1:
        beds_per_nurse_Hs = st.number_input("Beds per nurse",  min_value=1.0, max_value=10.0, value=4.0, step=0.5,
            help="Typical ranges: Med/Surg 4–6, ICU 1–2", key="beds_per_nurse_hs")
        nurses_per_bed_Hs = 1.0 / beds_per_nurse_Hs

    with hs_col2:
        cost_per_bedday_Hs = st.number_input("Cost per day ($)", min_value=100, max_value=2000, value=800, step=50,
                                             help="Includes staff, supplies, utilities", key="cost_hs")
    with hs_col3:
        shifts_per_day_Hs = st.number_input("Shifts per day", min_value=1, max_value=3, value=2, step=1,
                                            help="Typically 2-3 shifts per day", key="shifts_hs")

    st.write("##### Med Surge Tele (Hm) ")
    hm_col1, hm_col2, hm_col3 = st.columns(3)
    with hm_col1:
        beds_per_nurse_Hm = st.number_input( "Beds per nurse", min_value=1.0, max_value=10.0, value=5.0, step=0.5,
            help="Typical: 4–6 beds per nurse", key="beds_per_nurse_hm" )
        nurses_per_bed_Hm = 1 / beds_per_nurse_Hm
    with hm_col2:
        cost_per_bedday_Hm = st.number_input("Cost per bed ($)", min_value=100, max_value=2000, value=800, step=50,
                                             help="Includes staff, supplies, utilities", key="cost_hm")
    with hm_col3:
        shifts_per_day_Hm = st.number_input("Shifts per day", min_value=1, max_value=3, value=2, step=1,
                                            help="Typically 2-3 shifts per day", key="shifts_hm")

    st.write("##### ICU ")
    icu_col1, icu_col2, icu_col3 = st.columns(3)
    with icu_col1:
        beds_per_nurse_ICU = st.number_input( "Beds per nurse", min_value=1.0, max_value=4.0, value=2.0, step=0.5,
            help="Typical ICU ratio: 1–2 beds per nurse", key="beds_per_nurse_icu")
        nurses_per_bed_ICU = 1/beds_per_nurse_ICU
    with icu_col2:
        cost_per_bedday_ICU = st.number_input("Cost per day ($)", min_value=100, max_value=2000, value=800, step=50,
                                              help="Includes staff, supplies, utilities", key="cost_icu")
    with icu_col3:
        shifts_per_day_ICU = st.number_input("Shifts per day", min_value=1, max_value=3, value=2, step=1,
                                             help="Typically 2-3 shifts per day", key="shifts_icu")

    ############Add resource ###################
    if "resources" not in st.session_state:
        st.session_state.resources = []

    st.markdown("#### ➕ Add Staffing / Resource Type")
    st.caption("Optional resources beyond nursing (e.g., respiratory therapy, technicians). ")

    with st.form("add_resource_form", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            res_name = st.text_input("Resource name", placeholder="e.g. Respiratory Therapist")

        with col2:
            res_unit = st.selectbox("Applies to unit", ["Hs", "Hm", "I", "All"])

        with col3:
            beds_per_staff = st.number_input( "Beds per staff member", min_value=0.5,max_value=20.0,
                value=4.0, step=0.5, help="How many beds one staff member can cover (e.g. ICU 1–2, Med/Surg 4–6)")

            staff_per_bed = 1/beds_per_staff  #st.number_input("Staff per bed", min_value=0.0, value=0.1, step=0.05)

        col4, col5 = st.columns(2)
        with col4:
            cost_per_day = st.number_input("Cost per day-bed ($)", min_value=0.0, value=400.0, step=50.0)

        with col5:
            shifts_per_day = st.number_input("Shifts per day", min_value=1, max_value=3, value=2)

        submitted = st.form_submit_button("Add resource")

        if submitted and res_name:
            st.session_state.resources.append({
                "name": res_name,
                "unit": res_unit,
                "staff_per_bed": staff_per_bed,
                "cost_per_bed-day": cost_per_day,
                "shifts_per_day": shifts_per_day
            })

    st.markdown("##### 📋 Added Resources")

    if len(st.session_state.resources) == 0:
        st.info("No additional resources added yet.")
    else:
        for i, r in enumerate(st.session_state.resources):
            col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 0.5])

            col1.write(f"**{r['name']}**")
            col2.write(f"Unit: {r['unit']}")
            col3.write(f"{r['staff_per_bed']} / bed")
            col4.write(f"${r['cost_per_bed-day']} / bed")

            # check here
            if col5.button("❌", key=f"del_{i}"):
                st.session_state.resources.pop(i)
                st.experimental_rerun()

    ################################

    if st.button("Calculate Staffing Needs"):
        # CORRECTED: Use the per-component dictionary, not the total
        #extra_beddays_per_comp_cut
        extra_nurse_shifts_Hs = res['extra_beddays_per_comp_cut']['Hs'] * nurses_per_bed_Hs * shifts_per_day_Hs
        total_cost_Hs = res['extra_beddays_per_comp_cut']['Hs']* nurses_per_bed_Hs * cost_per_bedday_Hs

        extra_nurse_shifts_Hm = res['extra_beddays_per_comp_cut']['Hm'] * nurses_per_bed_Hm * shifts_per_day_Hm
        total_cost_Hm = res['extra_beddays_per_comp_cut']['Hm'] * nurses_per_bed_Hm * cost_per_bedday_Hm

        extra_nurse_shifts_ICU = res['extra_beddays_per_comp_cut']['I'] * nurses_per_bed_ICU * shifts_per_day_ICU
        total_cost_ICU = res['extra_beddays_per_comp_cut']['I']* nurses_per_bed_ICU * cost_per_bedday_ICU

        st.success("**Staffing and Cost Estimates:**")

        st.markdown("#### Nursing Requirements")
        st.write("**IP Surge (Hs):**")
        st.write(f"🧑‍⚕️ Extra nurse shifts needed: {extra_nurse_shifts_Hs:.0f}")
        st.write(f"💰 Total surge cost: ${total_cost_Hs:,.0f}")

        st.write("**Med Surge (Hm):**")
        st.write(f"🧑‍⚕️ Extra nurse shifts needed: {extra_nurse_shifts_Hm:.0f}")
        st.write(f"💰 Total surge cost: ${total_cost_Hm:,.0f}")

        st.write("**ICU:**")
        st.write(f"🧑‍⚕️ Extra nurse shifts needed: {extra_nurse_shifts_ICU:.0f}")
        st.write(f"💰 Total surge cost: ${total_cost_ICU:,.0f}")

        #st.write(f"📅 **Total extra workload:** {res['extra_beddays_total']:.1f} bed-days")
        st.markdown("#### Additional Staffing Requirements")
        total_extra_cost_resources = 0.0

        for r in st.session_state.resources:
            if r["unit"] == "All":
                beddays = res['extra_beddays_total_cut']
            else:
                beddays = res['extra_beddays_per_comp_cut'][r["unit"]]

            extra_shifts = beddays * r["staff_per_bed"]* r["shifts_per_day"]

            cost = beddays * r["staff_per_bed"]  * r["cost_per_bed-day"]
            #cost = beddays * r["cost_per_shift"]

            total_extra_cost_resources += cost

            st.write(f"**{r['unit']}:**")
            st.write(f"🧑‍⚕️ Extra {r['name']} shifts needed: {extra_shifts:.0f}")
            st.write(f"💰 Total surge cost: ${cost:,.0f}")
            #st.write( f"**{r['name']} ({r['unit']})** — " f"🧑‍⚕️ {extra_shifts:.0f} shifts, " f"💰 ${cost:,.0f}" )
            #st.write( f"**{r['name']} ({r['unit']})** — " f"🧑‍⚕️ {extra_shifts:.0f} shifts, " f"💰 ${cost:,.0f}" )

        # ---- TOTAL COST SUMMATION ----

        st.markdown("---")
        st.subheader("💵 Total Surge Cost Summary")
        total_cost_all = total_cost_Hs + total_cost_Hm + total_cost_ICU + total_extra_cost_resources

        st.write(f"**Total cost across all units:** ${total_cost_all:,.0f}")
        #st.write(f"📅 **Total extra workload:** {res['extra_beddays_total']:.1f} bed-days")

        st.write("#### Weekly Surge Cost Breakdown")

        # cost per bed-day dictionary
        cost_per_bedday = {
            "Hs": cost_per_bedday_Hs,
            "Hm": cost_per_bedday_Hm,
            "ICU": cost_per_bedday_ICU
        }

        nurses_per_bed = {
            "Hs": nurses_per_bed_Hs,
            "Hm": nurses_per_bed_Hm,
            "ICU": nurses_per_bed_ICU
        }

        # compute weekly cost per compartment
        df_cost_weeks = df_weeks.copy()

        for unit in ["Hs", "Hm", "ICU"]:
            df_cost_weeks[unit] = df_cost_weeks[unit] * nurses_per_bed[unit] * cost_per_bedday[unit]
        df_cost_weeks["Total"] = df_cost_weeks[["Hs", "Hm", "ICU"]].sum(axis=1)

        ###########
        # Weekly cost from added resources (cost per bed-day)
        df_resource_cost_weeks = pd.DataFrame(
            0.0,
            index=df_weeks.index,
            columns=["Total"]
        )

        for r in st.session_state.resources:

            if r["unit"] == "All":
                weekly_beddays = df_weeks["Total"]
            else:
                weekly_beddays = df_weeks[r["unit"]]

            # ✅ cost per bed-day → direct multiplication
            weekly_cost = weekly_beddays * r["cost_per_bed-day"] * r["staff_per_bed"]

            df_resource_cost_weeks["Total"] += weekly_cost

        df_cost_weeks_all = df_cost_weeks.copy()
        df_cost_weeks_all["Total"] += df_resource_cost_weeks["Total"]


        st.dataframe( df_cost_weeks_all.style.format("${:,.0f}"))



        ##########


        # st.dataframe(
        #     df_cost_weeks.style.format("${:,.0f}"),
        #     use_container_width=True
        # )

        #st.subheader("💵 Total Surge Resource Cost")
        #st.write(f"**${total_extra_cost_resources:,.0f}**")
    # Add this section to your Streamlit app


    #
    #
    # init_day_str=  '2024-01-01'
    # end_day_str= '2025-03-31'
    #
    # surge_specs = { "Hs": [],
    #     "Hm": [(3.0, 5.0, 1.0)],  # ONLY medical beds surge
    #     "I": [] }
    #
    # dt = 1.0
    # t_end = 50
    # times = np.arange(0, t_end + dt, dt)
    #
    # res = transient_response_for_multi_surge(surge_specs, times)
    #
    # x_ts = res["x_ts"]
    # x0 = res["x0"]
    # extra_beds_over_time = x_ts - x0


#Hm 6 extra bed