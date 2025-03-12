import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder
import datetime

# Set page config
st.set_page_config(layout="wide", page_title="Loan Portfolio Daily Monitoring Dashboard")

# Generate time series data for key metrics
def generate_time_series_data(days=90):
    np.random.seed(42)
    
    # Create date range for the last 90 days
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days)
    date_range = [start_date + datetime.timedelta(days=i) for i in range(days+1)]
    
    # Base values for metrics
    base_delinquency = 0.05  # 5% delinquency rate
    base_loss_rate = 0.02    # 2% loss rate
    base_prepayment = 0.10   # 10% prepayment rate
    base_balance = 100000000 # $100M portfolio
    
    # Create random variations with trend
    delinquency_trend = np.linspace(0, 0.01, days+1)  # Slight upward trend
    loss_trend = np.linspace(0, 0.005, days+1)        # Slight upward trend
    prepayment_trend = np.linspace(0, -0.02, days+1)  # Slight downward trend
    balance_trend = np.linspace(0, 5000000, days+1)   # Growing portfolio
    
    # Add random noise
    delinquency_rates = base_delinquency + delinquency_trend + np.random.normal(0, 0.003, days+1)
    loss_rates = base_loss_rate + loss_trend + np.random.normal(0, 0.002, days+1)
    prepayment_rates = base_prepayment + prepayment_trend + np.random.normal(0, 0.005, days+1)
    portfolio_balance = base_balance + balance_trend + np.random.normal(0, 1000000, days+1)
    
    # Ensure values are within reasonable ranges
    delinquency_rates = np.clip(delinquency_rates, 0.01, 0.15)
    loss_rates = np.clip(loss_rates, 0.005, 0.05)
    prepayment_rates = np.clip(prepayment_rates, 0.05, 0.20)
    portfolio_balance = np.clip(portfolio_balance, 90000000, 120000000)
    
    # Create breakdown by loan type
    loan_types = ['Auto Purchase', 'Debt Consolidation', 'HVAC', 'RV', 'Solar Roofing']
    
    # Data by loan type
    type_data = []
    for loan_type in loan_types:
        # Each loan type has slightly different characteristics
        if loan_type == 'Auto Purchase':
            type_multiplier = 0.8  # Lower delinquency
            balance_pct = 0.207
        elif loan_type == 'Debt Consolidation':
            type_multiplier = 1.5  # Higher delinquency
            balance_pct = 0.193
        elif loan_type == 'HVAC':
            type_multiplier = 0.9
            balance_pct = 0.199
        elif loan_type == 'RV':
            type_multiplier = 0.85
            balance_pct = 0.193
        else:  # Solar Roofing
            type_multiplier = 0.7  # Lowest delinquency
            balance_pct = 0.208
            
        for i, date in enumerate(date_range):
            type_data.append({
                'date': date,
                'loan_type': loan_type,
                'delinquency_rate': delinquency_rates[i] * type_multiplier,
                'loss_rate': loss_rates[i] * type_multiplier,
                'prepayment_rate': prepayment_rates[i] * (1/type_multiplier),
                'balance': portfolio_balance[i] * balance_pct
            })
    
    # Combine all data
    all_data = []
    for i, date in enumerate(date_range):
        all_data.append({
            'date': date,
            'delinquency_rate': delinquency_rates[i],
            'loss_rate': loss_rates[i],
            'prepayment_rate': prepayment_rates[i],
            'balance': portfolio_balance[i],
            'rollrate_current_to_30': 0.02 + np.random.normal(0, 0.002),
            'rollrate_30_to_60': 0.30 + np.random.normal(0, 0.02),
            'rollrate_60_to_90': 0.40 + np.random.normal(0, 0.03),
            'rollrate_90_to_default': 0.60 + np.random.normal(0, 0.04)
        })
    
    df_overall = pd.DataFrame(all_data)
    df_by_type = pd.DataFrame(type_data)
    
    return df_overall, df_by_type

# Load data
@st.cache_data
def load_data():
    return generate_time_series_data(90)

df_overall, df_by_type = load_data()

# Sidebar filters
st.sidebar.header("Time Range")
date_range = st.sidebar.slider(
    "Select Date Range",
    min_value=df_overall['date'].min().date(),
    max_value=df_overall['date'].max().date(),
    value=(df_overall['date'].min().date(), df_overall['date'].max().date())
)

# Convert date_range back to datetime for filtering
start_date = datetime.datetime.combine(date_range[0], datetime.time.min)
end_date = datetime.datetime.combine(date_range[1], datetime.time.max)

# Filter data based on date range
filtered_overall = df_overall[(df_overall['date'] >= start_date) & (df_overall['date'] <= end_date)]
filtered_by_type = df_by_type[(df_by_type['date'] >= start_date) & (df_by_type['date'] <= end_date)]

# Main dashboard
st.title("Loan Portfolio Daily Performance Monitoring")

# Current metrics (latest date)
latest_date = filtered_overall['date'].max()
latest_data = filtered_overall[filtered_overall['date'] == latest_date].iloc[0]

st.header("Current Portfolio Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Portfolio Balance", 
        f"${latest_data['balance']:,.0f}",
        delta=f"{(latest_data['balance'] / filtered_overall.iloc[0]['balance'] - 1) * 100:.1f}%"
    )

with col2:
    st.metric(
        "Delinquency Rate", 
        f"{latest_data['delinquency_rate']:.2%}",
        delta=f"{(latest_data['delinquency_rate'] - filtered_overall.iloc[0]['delinquency_rate']) * 100:.2f}pp", 
        delta_color="inverse"
    )

with col3:
    st.metric(
        "Loss Rate", 
        f"{latest_data['loss_rate']:.2%}",
        delta=f"{(latest_data['loss_rate'] - filtered_overall.iloc[0]['loss_rate']) * 100:.2f}pp", 
        delta_color="inverse"
    )

with col4:
    st.metric(
        "Prepayment Rate", 
        f"{latest_data['prepayment_rate']:.2%}",
        delta=f"{(latest_data['prepayment_rate'] - filtered_overall.iloc[0]['prepayment_rate']) * 100:.2f}pp"
    )

# Time series charts
st.header("Daily Performance Trends")

tab1, tab2, tab3 = st.tabs(["Delinquency & Loss", "Portfolio Balance", "Roll Rates"])

with tab1:
    # Delinquency and loss rate over time
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered_overall['date'], 
        y=filtered_overall['delinquency_rate'],
        mode='lines',
        name='Delinquency Rate',
        line=dict(color='orange', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=filtered_overall['date'], 
        y=filtered_overall['loss_rate'],
        mode='lines',
        name='Loss Rate',
        line=dict(color='red', width=2)
    ))
    fig.update_layout(
        title='Delinquency and Loss Rate Trends',
        xaxis_title='Date',
        yaxis_title='Rate',
        yaxis_tickformat='.2%',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Portfolio balance over time
    fig = px.line(
        filtered_overall, 
        x='date', 
        y='balance',
        title='Daily Portfolio Balance',
        labels={'date': 'Date', 'balance': 'Balance ($)'}
    )
    fig.update_layout(yaxis_tickformat='$,.0f')
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Roll rates over time
    roll_rate_df = filtered_overall[['date', 'rollrate_current_to_30', 'rollrate_30_to_60', 
                                      'rollrate_60_to_90', 'rollrate_90_to_default']]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=roll_rate_df['date'], 
        y=roll_rate_df['rollrate_current_to_30'],
        mode='lines',
        name='Current to 30 Days',
        line=dict(color='green', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=roll_rate_df['date'], 
        y=roll_rate_df['rollrate_30_to_60'],
        mode='lines',
        name='30 to 60 Days',
        line=dict(color='yellow', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=roll_rate_df['date'], 
        y=roll_rate_df['rollrate_60_to_90'],
        mode='lines',
        name='60 to 90 Days',
        line=dict(color='orange', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=roll_rate_df['date'], 
        y=roll_rate_df['rollrate_90_to_default'],
        mode='lines',
        name='90+ to Default',
        line=dict(color='red', width=2)
    ))
    fig.update_layout(
        title='Roll Rate Trends',
        xaxis_title='Date',
        yaxis_title='Roll Rate',
        yaxis_tickformat='.2%',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# Performance by loan type
st.header("Performance by Collateral Type")

# Get the most recent data for each loan type
latest_by_type = filtered_by_type[filtered_by_type['date'] == latest_date]

# Pivot for comparison
pivot_delinq = filtered_by_type.pivot(index='date', columns='loan_type', values='delinquency_rate')

# Plot delinquency by loan type
fig = px.line(
    pivot_delinq,
    x=pivot_delinq.index,
    y=pivot_delinq.columns,
    title='Delinquency Rate by Collateral Type',
    labels={'value': 'Delinquency Rate', 'variable': 'Collateral Type'}
)
fig.update_layout(yaxis_tickformat='.2%', legend_title='Collateral Type')
st.plotly_chart(fig, use_container_width=True)

# Current delinquency comparison
fig = px.bar(
    latest_by_type,
    x='loan_type',
    y='delinquency_rate',
    title='Current Delinquency Rate by Collateral Type',
    labels={'loan_type': 'Collateral Type', 'delinquency_rate': 'Delinquency Rate'},
    color='loan_type',
    text_auto='.2%'
)
fig.update_layout(yaxis_tickformat='.2%')
st.plotly_chart(fig, use_container_width=True)

# Show data table
st.header("Daily Data Table")
st.write("Overall Portfolio Metrics")

# Format the data for display
display_df = filtered_overall.copy()
display_df['date'] = display_df['date'].dt.date
display_df['delinquency_rate'] = display_df['delinquency_rate'].apply(lambda x: f"{x:.2%}")
display_df['loss_rate'] = display_df['loss_rate'].apply(lambda x: f"{x:.2%}")
display_df['prepayment_rate'] = display_df['prepayment_rate'].apply(lambda x: f"{x:.2%}")
display_df['balance'] = display_df['balance'].apply(lambda x: f"${x:,.0f}")
display_df['rollrate_current_to_30'] = display_df['rollrate_current_to_30'].apply(lambda x: f"{x:.2%}")
display_df['rollrate_30_to_60'] = display_df['rollrate_30_to_60'].apply(lambda x: f"{x:.2%}")
display_df['rollrate_60_to_90'] = display_df['rollrate_60_to_90'].apply(lambda x: f"{x:.2%}")
display_df['rollrate_90_to_default'] = display_df['rollrate_90_to_default'].apply(lambda x: f"{x:.2%}")

# Rename columns for better readability
display_df.columns = ['Date', 'Delinquency Rate', 'Loss Rate', 'Prepayment Rate', 'Balance',
                     'Roll Rate: Current→30d', 'Roll Rate: 30d→60d', 'Roll Rate: 60d→90d', 'Roll Rate: 90d→Default']

# Configure grid
gb = GridOptionsBuilder.from_dataframe(display_df)
gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=10)
gb.configure_default_column(sortable=True, filterable=True)

grid_options = gb.build()
AgGrid(
    display_df,
    gridOptions=grid_options,
    enable_enterprise_modules=True,
    update_on_grid_options_change=True
)
