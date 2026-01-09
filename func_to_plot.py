import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_occupancy(df, baseline_eq, cap_IP_surge, cap_med_surg, cap_ICU, init_day_str, end_day_str):

    dates = df['Date']
    fig = make_subplots( rows=3, cols=1, subplot_titles=('IP Surge (Hs)', 'MED/SURG/TELE (Hm)', 'ICU (I)'),
                         shared_xaxes=True, vertical_spacing=0.05)

    # Hs plot
    fig.add_trace(go.Scatter(x=dates, y=df['OCC_BEDS_IP_SURGE'], mode='lines', name='Observed Hs', line=dict(color='blue')),
                  row=1, col=1 )
    fig.add_hline(y=baseline_eq['Hs'], line_dash="dash", line_color="red",
                  annotation_text=f"Equilibrium Hs: {baseline_eq['Hs']:.2f}", row=1, col=1)
    fig.add_hline(y=cap_IP_surge, line_dash="dot", line_color="orange",
                  annotation_text=f"Capacity: {cap_IP_surge}", row=1, col=1)
    # Hm plot
    fig.add_trace (go.Scatter(x=dates, y=df['OCC_BEDS_MED_SURG_TELE'], mode='lines', name='Observed Hm', line=dict(color='green')),
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
    return fig
    #st.plotly_chart(fig, use_container_width=True)

def plot_ED_admissions(df, fixed_params, baseline_eq,init_day_str,end_day_str):
    dates = df['Date']
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
    return fig