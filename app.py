# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from func_to_plot import plot_occupancy,plot_ED_admissions, plot_occupancy_under_scenarios
from model_full_equilibrium_points import (
    procces_data, compute_params_from_df, compute_equilibrium_data,transient_response_for_multi_surge,
     fixed_params, jacobian_at_equilibrium, Ad_Hs_mean, Ad_Hm_mean, Ad_ICU_mean, df)

# General font size
plt.rcParams.update({'font.size': 18,
    'axes.labelsize': 22,   # x and y labels
    'axes.titlesize': 22,   # title size
    'xtick.labelsize': 20,  # x tick labels
    'ytick.labelsize': 20,  # y tick labels
    'legend.fontsize': 22 })


st.set_page_config(page_title="Hospital Equilibrium Explorer", layout="wide")


# --------------------------------------------------------------------
#  Load data & fixed params (should be fast; we assume procces_data is deterministic)
# --------------------------------------------------------------------
# init_day='2024-01-01'
@st.cache_data
def load_data(init_day, end_day):
    df = procces_data(init_day=init_day, end_day=end_day)
    fixed_params = compute_params_from_df(df)
    return df, fixed_params
# --------------------------------------------------------------------
# Sidebar controls for selecting mode
# --------------------------------------------------------------------
st.sidebar.header("🔧 Analysis Mode")
mode = st.sidebar.radio("Select analysis view", ["Equilibrium", "Transient surge"])

# --------------------------------------------------------------------
#         Equilibrium Page
# --------------------------------------------------------------------
if mode == "Equilibrium":
    st.header("🏥 Hospital Capacity & Surge Planning Explorer")
    st.markdown("Model-based analysis of baseline equilibrium, surge scenarios, workload, and resource planning" )

    st.markdown("#### Baseline Equilibrium Analysis")
    st.markdown("Baseline equilibrium computed from weekly means of arrivals/admissions and fixed parameters.")

    date_range = st.slider( "Select date range for baseline estimation", min_value=dt.date(2020, 1, 1),  max_value=dt.date(2025, 3, 31),
        value=(dt.date(2024, 1, 1), dt.date(2025, 3, 31)), format="YYYY-MM-DD")

    init_day, end_day = date_range
    init_day_str = init_day.strftime("%Y-%m-%d")
    end_day_str = end_day.strftime("%Y-%m-%d")

    df, fixed_params = load_data(init_day_str, end_day_str) # Read data set!
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

    # Compute the Baseline Equilibrium
    baseline_eq = compute_equilibrium_data(fixed_params, arrivals_mean, Ad_Hs_mean, Ad_Hm_mean, Ad_ICU_mean,
                                       At_Hs_mean, At_Hm_mean, At_ICU_mean)
    # Hospital Capacity #   CHECK HERE: if these computations are good!!!!
    cap_ICU = int((df['TTL_BEDS_ICU'] - df['UNAVBL_BEDS_ICU']).median())
    cap_med_surg = int((df['TTL_BEDS_MED_SURG_TELE'] - df['UNAVBL_BEDS_MED_SURG_TELE']).median())
    cap_IP_surge = int((df['TTL_BEDS_IP_SURGE'] - df['UNAVBL_BEDS_IP_SURGE']).median())

    # --------------------------------------------------------------------
    #  Plot of observed occupancy time series with equilibrium lines
    # --------------------------------------------------------------------
    dates = df['Date']
    fig = plot_occupancy(df, baseline_eq, cap_IP_surge, cap_med_surg, cap_ICU, init_day_str, end_day_str)
    st.plotly_chart(fig, use_container_width=True)

    fig = plot_ED_admissions(df, fixed_params, baseline_eq,init_day_str,end_day_str)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Equilibrium Occupancy Levels")
    eq_table = pd.DataFrame({ "compartment": list(baseline_eq.keys()), "value": [baseline_eq[k] for k in baseline_eq.keys()] })
    st.dataframe(eq_table.style.format({"value": "{:.2f}"}), height=300)



# --------------------------------------------------------------------
#         Transient surge Page
# --------------------------------------------------------------------
else :
    # ---------- Surge scenario selection ----------
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

    # --------------  Determine simulation horizon ----------------------
    # take the latest surge end time across all units and extend it by 70 days to allow the system to relax back to equilibrium
    t_end = max(w[1] for comp in surge_specs.values() for w in comp) + 70
    times = np.arange(0, t_end + dt, dt) # Time grid for simulation (daily resolution)

    #--------------- Compute transient system response to multiple surge events ------
    res = transient_response_for_multi_surge(surge_specs, times)
    x_ts = res['x_ts'] # Time series of state variables (beds by unit)
    x0 = res['x0']     # Baseline equilibrium (no-surge steady state)

    extra_beds_over_time = x_ts - x0 # Compute daily excess bed occupancy relative to baseline

    # -------------- Apply threshold to the time series output --------------
    # We define "active surge impact" as any time when at least one unit exceeds the baseline by more than the specified threshold.

    threshold = 0.1  # minimum extra beds considered meaningful
    max_extra = np.max(extra_beds_over_time, axis=1) # Maximum excess occupancy across units at each time point
    active_idx = np.where(max_extra >= threshold)[0] # Indices where the system is still above the threshold

    # Last time index with meaningful surge impact
    if len(active_idx) > 0:
        t_cut = active_idx[-1] + 1
    else:
        t_cut = 1   # No meaningful deviation from equilibrium detected

    # Truncate the time series to exclude periods where excess occupancy is negligible (< threshold).
    # All downstream workload and cost calculations are based on this truncated series.
    times_plot = times[:t_cut]
    x_ts_plot= x_ts[:t_cut]

    fig_occ_under_scenarios = plot_occupancy_under_scenarios(x_ts, x0,t_cut, times_plot,x_ts_plot)
    st.plotly_chart(fig_occ_under_scenarios, use_container_width=True)

    # --------------- Compute peak excess bed demand ------------------------------
    # Compute peak excess bed demand: total peak across all units,  unit-specific peaks for capacity planning
    peak_extra_beds_total = np.max(np.sum(extra_beds_over_time, axis=1))
    peak_extra_beds_per_comp = { 'Hs': np.max(extra_beds_over_time[:, 0]),  'Hm': np.max(extra_beds_over_time[:, 1]),
                                 'I': np.max(extra_beds_over_time[:, 2]) }

    #------------------------------------------------------
    #          Output summary statistics
    # ------------------------------------------------------
    col1, col2 = st.columns(2)
    # ---------------- Peak demand metrics ----------------
    with col1:
        st.write("### 🏥 Peak Additional Bed Requirements")
        st.metric("Total Extra Beds", f"{peak_extra_beds_total:.1f}")
        st.write("**By compartment:**")
        st.write(f"- Hs: {peak_extra_beds_per_comp['Hs']:.1f} extra")
        st.write(f"- Hm: {peak_extra_beds_per_comp['Hm']:.1f} extra")
        st.write(f"- ICU: {peak_extra_beds_per_comp['I']:.1f} extra")

    # ---------------- Cumulative workload (bed-days) ----------------
    with col2:
        st.write("### 📊 Cumulative Bed-Days (Total Workload)")
        # Restrict extra beds to the active surge period only
        extra_beds_over_time_cut = extra_beds_over_time[:t_cut]

        # Total bed-days per unit (discrete daily sum)
        extra_beddays_per_comp_cut = {comp: extra_beds_over_time_cut[:, j].sum() for j, comp in enumerate(["Hs", "Hm", "I"])}
        extra_beddays_total_cut = sum(extra_beddays_per_comp_cut.values()) # Total system-wide bed-days

        # Store results for downstream staffing and cost calculations
        res["extra_beddays_per_comp_cut"] = extra_beddays_per_comp_cut
        res["extra_beddays_total_cut"] = extra_beddays_total_cut

        st.metric("Extra Bed-Days (Total)", f"{res["extra_beddays_total_cut"]:.1f}")
        st.write("**By compartment:**")
        for comp, beddays in res["extra_beddays_per_comp_cut"].items():
            st.write(f"- {comp}: {beddays:.1f} bed-days")

    # --------------------------------------------------------------------
    # Weekly aggregation of bed-day workload
    # --------------------------------------------------------------------
    st.write("### 🗓️ Weekly Bed-Day Workload")
    st.caption("Weekly aggregation of extra occupied beds (bed-days)")

    n_days, n_units = extra_beds_over_time_cut.shape
    n_weeks = int(np.ceil(n_days / 7))

    # Pad with zeros so the number of days is divisible by 7
    pad = n_weeks * 7 - n_days
    extra_padded = np.pad(extra_beds_over_time_cut,((0, pad), (0, 0)), mode="constant")

    # Reshape into (weeks × days × units) and sum across days
    weeks_workload = extra_padded.reshape(n_weeks, 7, n_units).sum(axis=1)

    # Build weekly summary table
    week_labels = [f"Week {i + 1}" for i in range(n_weeks)]; unit_labels = ["Hs", "Hm", "I"]
    df_weeks = pd.DataFrame( weeks_workload,index=week_labels,columns=unit_labels )

    # Add totals:
    df_weeks["Total"] = df_weeks.sum(axis=1)      #  - Row-wise: total workload per week
    df_weeks.loc["Total"] = df_weeks.sum(axis=0)  #  - Column-wise: total workload per unit across all weeks

    st.dataframe(df_weeks)

    # --------------------------------------------------------------------
    # Staffing & Resource Planning Inputs
    # --------------------------------------------------------------------
    # This section estimates: 1) Nursing workload (number of nurse shifts required) 2) Total nursing cost
    #  IMPORTANT MODELING ASSUMPTIONS:
    # - Costs are specified PER OCCUPIED BED PER DAY (bed-day) → These costs already aggregate all nursing shifts in a day
    #  - The number of shifts per day is used ONLY to estimate staffing volume,   NOT to scale cost
    #  - All calculations are based on cumulative extra bed-days
    st.write("### 👥 Staffing & Resource Planning")
    st.caption("Estimate staffing needs and operational costs based on cumulative bed-day demand")

    # --------------------------------------------------------------------
    # Nursing assumptions by unit
    #--------------------------------------------------------------------
    # For each unit, the user specifies:
    #   - Beds per nurse (converted internally to nurses per bed)
    #   - Nursing cost per occupied bed per day (all shifts included)
    #   - Number of nursing shifts per day (used only for staffing volume estimates)
    # --------------------------------------------------------------------

    st.write("#### Nursing Assumptions by Unit")
    # ------------------- Inpatient Surge (Hs)----------------------------
    st.write("##### IP Surge (Hs) ")
    hs_col1, hs_col2, hs_col3 = st.columns(3)
    with hs_col1:
        beds_per_nurse_Hs = st.number_input("Beds per nurse",  min_value=1.0, max_value=10.0, value=4.0, step=0.5,
            help="Typical ranges: Med/Surg 4–6, ICU 1–2", key="beds_per_nurse_hs")
        nurses_per_bed_Hs = 1.0 / beds_per_nurse_Hs # Convert to nurses required per occupied bed
    with hs_col2:
        nurse_cost_per_bed_day_Hs = st.number_input("Cost per day ($)", min_value=100, max_value=2000, value=800, step=50,
                                             help="Includes staff, supplies, utilities", key="cost_hs")
    with hs_col3:
        nurse_shifts_per_day_Hs = st.number_input("Shifts per day", min_value=1, max_value=3, value=2, step=1,
                                            help="Typically 2-3 shifts per day", key="shifts_hs")
    # -------------------  Med/Surg Telemetry (Hm) ----------------------------
    st.write("##### Med Surge Tele (Hm) ")
    hm_col1, hm_col2, hm_col3 = st.columns(3)
    with hm_col1:
        beds_per_nurse_Hm = st.number_input( "Beds per nurse", min_value=1.0, max_value=10.0, value=5.0, step=0.5,
            help="Typical: 4–6 beds per nurse", key="beds_per_nurse_hm" )
        nurses_per_bed_Hm = 1 / beds_per_nurse_Hm # Convert to nurses required per occupied bed
    with hm_col2:
        nurse_cost_per_bed_day_Hm = st.number_input("Cost per bed ($)", min_value=100, max_value=2000, value=800, step=50,
                                             help="Includes staff, supplies, utilities", key="cost_hm")
    with hm_col3:
        nurse_shifts_per_day_Hm = st.number_input("Shifts per day", min_value=1, max_value=3, value=2, step=1,
                                            help="Typically 2-3 shifts per day", key="shifts_hm")
    # -------------------  Intensive Care Unit (I / ICU) ----------------------------
    st.write("##### ICU ")
    icu_col1, icu_col2, icu_col3 = st.columns(3)
    with icu_col1:
        beds_per_nurse_ICU = st.number_input( "Beds per nurse", min_value=1.0, max_value=4.0, value=2.0, step=0.5,
            help="Typical ICU ratio: 1–2 beds per nurse", key="beds_per_nurse_icu")
        nurses_per_bed_ICU = 1/beds_per_nurse_ICU # Convert to nurses required per occupied bed
    with icu_col2:
        nurse_cost_per_bed_day_ICU = st.number_input("Cost per day ($)", min_value=100, max_value=2000, value=800, step=50,
                                              help="Includes staff, supplies, utilities", key="cost_icu")
    with icu_col3:
        nurse_shifts_per_day_ICU = st.number_input("Shifts per day", min_value=1, max_value=3, value=2, step=1,
                                             help="Typically 2-3 shifts per day", key="shifts_icu")

    #--------------------------------------------------------------------
    #   Additional Staffing / Resource Types (Non-nursing)
    #--------------------------------------------------------------------
    # This section allows the user to define optional staff categories beyond nurses (e.g., respiratory therapists, technicians).
    #
    # Each added resource is stored in Streamlit session state and includes:
    #   - Name of the resource, Unit it applies to (Hs, Hm, ICU, or all),  Staffing intensity (staff per occupied bed)
    #   - Cost per occupied bed per day,  Number of shifts per day (for staffing volume only)

    if "resources" not in st.session_state: # Initialize resource list if it does not exist
        st.session_state.resources = []
    st.markdown("#### ➕ Add Staffing / Resource Type")
    st.caption("Optional resources beyond nursing (e.g., respiratory therapy, technicians). ")

    with st.form("add_resource_form", clear_on_submit=True): # Form used to add a new resource entry
        col1, col2, col3 = st.columns(3)

        with col1: # Descriptive name of the staff/resource
            res_name = st.text_input("Resource name", placeholder="e.g. Respiratory Therapist")

        with col2: # Unit(s) where this resource applies
            res_unit = st.selectbox("Applies to unit", ["Hs", "Hm", "I", "All"])

        with col3: # Beds covered by one staff member
            beds_per_staff = st.number_input( "Beds per staff member", min_value=0.5,max_value=20.0,
                value=4.0, step=0.5, help="How many beds one staff member can cover (e.g. ICU 1–2, Med/Surg 4–6)")
            staff_per_bed = 1/beds_per_staff  # Convert to staff required per occupied bed

        col4, col5 = st.columns(2)
        with col4: # Cost per occupied bed per day for this resource
            staff_cost_per_day = st.number_input("Cost per day-bed ($)", min_value=0.0, value=400.0, step=50.0)

        with col5: # Number of shifts per day (used for staffing volume only)
            staff_shifts_per_day = st.number_input("Shifts per day", min_value=1, max_value=3, value=2)

        submitted = st.form_submit_button("Add resource")     # Submit button for the form

        if submitted and res_name:     # Store resource in session state if valid
            st.session_state.resources.append({ "name": res_name, "unit": res_unit, "staff_per_bed": staff_per_bed,
                "staff_cost_per_bed_day": staff_cost_per_day, "staff_shifts_per_day": staff_shifts_per_day})

    #--------------------------------------------------------------------
    #  Display and manage added resources
    #--------------------------------------------------------------------
    # Lists all added resources and allows deletion
    st.markdown("##### 📋 Added Resources")
    if len(st.session_state.resources) == 0:
        st.info("No additional resources added yet.")
    else:
        for i, r in enumerate(st.session_state.resources):
            # Layout for resource summary + delete button
            col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 0.5])

            col1.write(f"**{r['name']}**")
            col2.write(f"Unit: {r['unit']}")
            col3.write(f"{r['staff_per_bed']} / bed")
            col4.write(f"${r['staff_cost_per_bed_day']} / bed")
            if col5.button("❌", key=f"del_{i}"): # Remove resource entry
                st.session_state.resources.pop(i)
                st.experimental_rerun()


    #--------------------------------------------------------------------
    #  Result Calculation of Staffing Needs
    #--------------------------------------------------------------------
    if st.button("Calculate Staffing Needs"):
        extra_nurse_shifts_Hs = res['extra_beddays_per_comp_cut']['Hs'] * nurses_per_bed_Hs * nurse_shifts_per_day_Hs
        total_nurse_cost_Hs = res['extra_beddays_per_comp_cut']['Hs']* nurses_per_bed_Hs * nurse_cost_per_bed_day_Hs

        extra_nurse_shifts_Hm = res['extra_beddays_per_comp_cut']['Hm'] * nurses_per_bed_Hm * nurse_shifts_per_day_Hm
        total_nurse_cost_Hm = res['extra_beddays_per_comp_cut']['Hm'] * nurses_per_bed_Hm * nurse_cost_per_bed_day_Hm

        extra_nurse_shifts_ICU = res['extra_beddays_per_comp_cut']['I'] * nurses_per_bed_ICU * nurse_shifts_per_day_ICU
        total_nurse_cost_ICU = res['extra_beddays_per_comp_cut']['I']* nurses_per_bed_ICU * nurse_cost_per_bed_day_ICU

        st.success("**Staffing and Cost Estimates:**")

        st.markdown("#### Nursing Requirements")
        st.write("**IP Surge (Hs):**")
        st.write(f"🧑‍⚕️ Extra nurse shifts needed: {extra_nurse_shifts_Hs:.0f}")
        st.write(f"💰 Total surge cost: ${total_nurse_cost_Hs:,.0f}")

        st.write("**Med Surge (Hm):**")
        st.write(f"🧑‍⚕️ Extra nurse shifts needed: {extra_nurse_shifts_Hm:.0f}")
        st.write(f"💰 Total surge cost: ${total_nurse_cost_Hm:,.0f}")

        st.write("**ICU:**")
        st.write(f"🧑‍⚕️ Extra nurse shifts needed: {extra_nurse_shifts_ICU:.0f}")
        st.write(f"💰 Total surge cost: ${total_nurse_cost_ICU:,.0f}")

        st.markdown("#### Additional Staffing Requirements")
        total_extra_cost_resources = 0.0

        for r in st.session_state.resources:
            if r["unit"] == "All":
                beddays = res['extra_beddays_total_cut']
            else:
                beddays = res['extra_beddays_per_comp_cut'][r["unit"]]
            extra_staff_shifts = beddays * r["staff_per_bed"]* r["staff_shifts_per_day"]

            staff_cost = beddays * r["staff_per_bed"]  * r["staff_cost_per_bed_day"]
            total_extra_cost_resources += staff_cost

            st.write(f"**{r['unit']}:**")
            st.write(f"🧑‍⚕️ Extra {r['name']} shifts needed: {extra_staff_shifts:.0f}")
            st.write(f"💰 Total surge cost: ${staff_cost:,.0f}")
        # --------------------------------------------------------------------
        #                 TOTAL COST SUMMATION
        # --------------------------------------------------------------------
        st.markdown("---")
        st.subheader("💵 Total Surge Cost Summary")
        total_nurse_cost_all = total_nurse_cost_Hs + total_nurse_cost_Hm + total_nurse_cost_ICU + total_extra_cost_resources

        st.write(f"**Total cost across all units:** ${total_nurse_cost_all:,.0f}")

        # --------------------------------------------------------------------
        #                 Weekly Surge Cost Breakdown
        # --------------------------------------------------------------------
        st.write("#### Weekly Surge Cost Breakdown")
        nurse_cost_per_bed_day = { "Hs": nurse_cost_per_bed_day_Hs,"Hm": nurse_cost_per_bed_day_Hm, "I": nurse_cost_per_bed_day_ICU}
        nurses_per_bed = { "Hs": nurses_per_bed_Hs, "Hm": nurses_per_bed_Hm, "I": nurses_per_bed_ICU}

        #-------- Compute weekly cost per compartment
        df_cost_weeks = df_weeks.copy()

        for unit in ["Hs", "Hm", "I"]:
            df_cost_weeks[unit] = df_cost_weeks[unit] * nurses_per_bed[unit] * nurse_cost_per_bed_day[unit]
        df_cost_weeks["Total"] = df_cost_weeks[["Hs", "Hm", "I"]].sum(axis=1)

        # Weekly cost from added resources (cost per bed-day)
        df_resource_cost_weeks = pd.DataFrame(0.0, index=df_weeks.index, columns=["Total"])

        for r in st.session_state.resources:
            if r["unit"] == "All":
                weekly_beddays = df_weeks["Total"]
            else:
                weekly_beddays = df_weeks[r["unit"]]

            # ✅ cost per bed-day → direct multiplication
            weekly_cost = weekly_beddays * r["staff_cost_per_bed_day"] * r["staff_per_bed"]

            df_resource_cost_weeks["Total"] += weekly_cost

        df_cost_weeks_all = df_cost_weeks.copy()
        df_cost_weeks_all["Total"] += df_resource_cost_weeks["Total"]
        st.dataframe( df_cost_weeks_all.style.format("${:,.0f}"))


