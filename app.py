import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import JsCode
import datetime

# Set page config
st.set_page_config(layout="wide", page_title="Loan Portfolio Monitoring Dashboard")

# Sample data generation function
def generate_loan_data(n_loans=1000):
    np.random.seed(42)
    
    # Create loan IDs
    loan_ids = [f'LOAN-{i:06d}' for i in range(1, n_loans + 1)]
    
    # Create dates (loans originated in last 3 years)
    start_date = datetime.datetime.now() - datetime.timedelta(days=365*3)
    dates = [start_date + datetime.timedelta(days=np.random.randint(0, 365*3)) for _ in range(n_loans)]
    
    # Collateral types with distribution matching the case study
    collateral_types = np.random.choice(
        ['Auto Purchase', 'Debt Consolidation', 'HVAC', 'RV', 'Solar Roofing'],
        size=n_loans,
        p=[0.207, 0.193, 0.199, 0.193, 0.208]
    )
    
    # Loan amounts based on collateral type
    loan_amounts = []
    for ctype in collateral_types:
        if ctype == 'Auto Purchase':
            loan_amounts.append(np.random.uniform(5000, 35000))
        elif ctype == 'Debt Consolidation':
            loan_amounts.append(np.random.uniform(10000, 50000))
        elif ctype == 'HVAC':
            loan_amounts.append(np.random.uniform(8000, 20000))
        elif ctype == 'RV':
            loan_amounts.append(np.random.uniform(20000, 100000))
        else:  # Solar Roofing
            loan_amounts.append(np.random.uniform(15000, 40000))
    
    # Loan terms in months
    loan_terms = []
    for ctype in collateral_types:
        if ctype == 'Auto Purchase':
            loan_terms.append(np.random.choice([36, 48, 60, 72]))
        elif ctype == 'Debt Consolidation':
            loan_terms.append(np.random.choice([36, 48, 60]))
        elif ctype == 'HVAC':
            loan_terms.append(np.random.choice([60, 84, 120]))
        elif ctype == 'RV':
            loan_terms.append(np.random.choice([60, 84, 120, 180]))
        else:  # Solar Roofing
            loan_terms.append(np.random.choice([120, 180, 240]))
    
    # Interest rates based on collateral type
    interest_rates = []
    for ctype in collateral_types:
        if ctype == 'Auto Purchase':
            interest_rates.append(np.random.uniform(0.039, 0.089))
        elif ctype == 'Debt Consolidation':
            interest_rates.append(np.random.uniform(0.069, 0.149))
        elif ctype == 'HVAC':
            interest_rates.append(np.random.uniform(0.049, 0.099))
        elif ctype == 'RV':
            interest_rates.append(np.random.uniform(0.054, 0.089))
        else:  # Solar Roofing
            interest_rates.append(np.random.uniform(0.039, 0.079))
    
    # FICO scores based on collateral type
    fico_scores = []
    for ctype in collateral_types:
        if ctype == 'Auto Purchase':
            fico_scores.append(np.random.randint(620, 800))
        elif ctype == 'Debt Consolidation':
            fico_scores.append(np.random.randint(640, 780))
        elif ctype == 'HVAC':
            fico_scores.append(np.random.randint(660, 820))
        elif ctype == 'RV':
            fico_scores.append(np.random.randint(680, 820))
        else:  # Solar Roofing
            fico_scores.append(np.random.randint(680, 830))
    
    # Delinquency status (current, 30 days, 60 days, 90+ days, default)
    delinquency_probs = {
        'Auto Purchase': [0.94, 0.03, 0.01, 0.01, 0.01],
        'Debt Consolidation': [0.88, 0.05, 0.03, 0.02, 0.02],
        'HVAC': [0.95, 0.02, 0.01, 0.01, 0.01],
        'RV': [0.96, 0.02, 0.01, 0.005, 0.005],
        'Solar Roofing': [0.97, 0.01, 0.01, 0.005, 0.005]
    }
    
    delinquency_status = []
    for ctype in collateral_types:
        delinquency_status.append(np.random.choice(
            ['Current', '30 Days', '60 Days', '90+ Days', 'Default'],
            p=delinquency_probs[ctype]
        ))
    
    # Purchase price as % of principal
    purchase_prices = []
    for ctype in collateral_types:
        if ctype == 'Auto Purchase':
            purchase_prices.append(np.random.uniform(0.92, 0.97))
        elif ctype == 'Debt Consolidation':
            purchase_prices.append(np.random.uniform(0.85, 0.95))
        elif ctype == 'HVAC':
            purchase_prices.append(np.random.uniform(0.90, 0.98))
        elif ctype == 'RV':
            purchase_prices.append(np.random.uniform(0.93, 0.98))
        else:  # Solar Roofing
            purchase_prices.append(np.random.uniform(0.94, 0.99))
    
    # Remaining balance
    remaining_balance = []
    for i in range(n_loans):
        # Calculate months elapsed
        months_elapsed = (datetime.datetime.now() - dates[i]).days / 30
        # Calculate remaining balance based on simple amortization
        if months_elapsed >= loan_terms[i] or delinquency_status[i] == 'Default':
            remaining_balance.append(0)
        else:
            total_payments = loan_terms[i]
            payments_made = min(int(months_elapsed), total_payments)
            remaining_pct = (total_payments - payments_made) / total_payments
            remaining_balance.append(loan_amounts[i] * remaining_pct)
    
    # Calculated IRR
    irrs = []
    for i in range(n_loans):
        base_irr = (interest_rates[i] - 0.01)  # Base IRR is interest rate minus servicing fee
        
        # Adjust for purchase price
        discount_factor = (1 - purchase_prices[i]) * 2  # Convert discount to IRR boost
        
        # Adjust for delinquency
        delinq_adjustment = {
            'Current': 0,
            '30 Days': -0.02,
            '60 Days': -0.05,
            '90+ Days': -0.08,
            'Default': -0.15
        }
        
        adjusted_irr = base_irr + discount_factor + delinq_adjustment[delinquency_status[i]]
        irrs.append(max(adjusted_irr, -0.10))  # Floor at -10%
    
    # Calculated WAL
    wals = []
    for i in range(n_loans):
        base_wal = loan_terms[i] / 2  # Simple approximation
        
        # Adjust for collateral type (prepayment behavior)
        prepay_factor = {
            'Auto Purchase': 0.8,  # Faster prepayment
            'Debt Consolidation': 0.9,
            'HVAC': 1.1,  # Slower prepayment
            'RV': 1.0,
            'Solar Roofing': 1.2  # Slowest prepayment
        }
        
        # Adjust for interest rate environment
        rate_factor = 1 + (interest_rates[i] - 0.06) * 3  # Higher rates = slower prepayment
        
        adjusted_wal = base_wal * prepay_factor[collateral_types[i]] * rate_factor
        wals.append(max(min(adjusted_wal, loan_terms[i]), 1))  # Constrain between 1 and term
    
    # Loan vintage (year and quarter)
    vintages = [f"{d.year} Q{(d.month-1)//3+1}" for d in dates]
    
    # Create DataFrame
    df = pd.DataFrame({
        'loan_id': loan_ids,
        'origination_date': dates,
        'collateral_type': collateral_types,
        'loan_amount': loan_amounts,
        'loan_term': loan_terms,
        'interest_rate': interest_rates,
        'fico_score': fico_scores,
        'delinquency_status': delinquency_status,
        'purchase_price_pct': purchase_prices,
        'remaining_balance': remaining_balance,
        'irr': irrs,
        'wal': wals,
        'vintage': vintages
    })
    
    return df

# Generate sample data
@st.cache_data
def load_data():
    return generate_loan_data(2000)

df = load_data()

# Sidebar filters
st.sidebar.header("Portfolio Filters")

# Filter by collateral type
collateral_options = ['All'] + sorted(df['collateral_type'].unique().tolist())
selected_collateral = st.sidebar.multiselect(
    "Collateral Type",
    options=collateral_options,
    default=['All']
)

# Filter by vintage
vintage_options = ['All'] + sorted(df['vintage'].unique().tolist())
selected_vintage = st.sidebar.multiselect(
    "Vintage",
    options=vintage_options,
    default=['All']
)

# Filter by delinquency status
delinquency_options = ['All'] + sorted(df['delinquency_status'].unique().tolist())
selected_delinquency = st.sidebar.multiselect(
    "Delinquency Status",
    options=delinquency_options,
    default=['All']
)

# FICO score range
min_fico, max_fico = int(df['fico_score'].min()), int(df['fico_score'].max())
fico_range = st.sidebar.slider(
    "FICO Score Range",
    min_value=min_fico,
    max_value=max_fico,
    value=(min_fico, max_fico)
)

# Apply filters
filtered_df = df.copy()

if 'All' not in selected_collateral:
    filtered_df = filtered_df[filtered_df['collateral_type'].isin(selected_collateral)]
    
if 'All' not in selected_vintage:
    filtered_df = filtered_df[filtered_df['vintage'].isin(selected_vintage)]
    
if 'All' not in selected_delinquency:
    filtered_df = filtered_df[filtered_df['delinquency_status'].isin(selected_delinquency)]

filtered_df = filtered_df[
    (filtered_df['fico_score'] >= fico_range[0]) & 
    (filtered_df['fico_score'] <= fico_range[1])
]

# Main dashboard
st.title("Loan Portfolio Monitoring Dashboard")

# Portfolio Overview metrics
st.header("Portfolio Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Portfolio Value", 
        f"${filtered_df['remaining_balance'].sum():,.2f}",
        delta=f"{(filtered_df['remaining_balance'].sum() / df['loan_amount'].sum() - 1) * 100:.1f}%"
    )

with col2:
    weighted_irr = (filtered_df['irr'] * filtered_df['remaining_balance']).sum() / filtered_df['remaining_balance'].sum()
    st.metric(
        "Weighted Average IRR", 
        f"{weighted_irr:.2%}",
        delta=f"{(weighted_irr - (df['irr'] * df['remaining_balance']).sum() / df['remaining_balance'].sum()) * 100:.2f}pp"
    )

with col3:
    weighted_wal = (filtered_df['wal'] * filtered_df['remaining_balance']).sum() / filtered_df['remaining_balance'].sum()
    st.metric(
        "Weighted Average Life", 
        f"{weighted_wal:.2f} months",
        delta=f"{(weighted_wal - (df['wal'] * df['remaining_balance']).sum() / df['remaining_balance'].sum()):.2f} mo"
    )

with col4:
    delinquency_rate = filtered_df[filtered_df['delinquency_status'] != 'Current'].shape[0] / filtered_df.shape[0]
    st.metric(
        "Delinquency Rate", 
        f"{delinquency_rate:.2%}",
        delta=f"{-(delinquency_rate - df[df['delinquency_status'] != 'Current'].shape[0] / df.shape[0]) * 100:.2f}pp", 
        delta_color="inverse"
    )

# Performance by Collateral Type
st.subheader("Performance by Collateral Type")
col1, col2 = st.columns(2)

with col1:
    # Portfolio Composition
    collateral_counts = filtered_df['collateral_type'].value_counts().reset_index()
    collateral_counts.columns = ['Collateral Type', 'Count']
    
    fig = px.pie(
        collateral_counts, 
        values='Count', 
        names='Collateral Type',
        title='Portfolio Composition by Collateral Type',
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # IRR by Collateral Type
    irr_by_collateral = filtered_df.groupby('collateral_type').apply(
        lambda x: (x['irr'] * x['remaining_balance']).sum() / x['remaining_balance'].sum()
    ).reset_index()
    irr_by_collateral.columns = ['Collateral Type', 'Weighted IRR']
    
    fig = px.bar(
        irr_by_collateral,
        x='Collateral Type',
        y='Weighted IRR',
        title='Weighted Average IRR by Collateral Type',
        color='Collateral Type',
        color_discrete_sequence=px.colors.qualitative.Set2,
        text_auto='.2%'
    )
    fig.update_layout(yaxis_tickformat='.2%')
    st.plotly_chart(fig, use_container_width=True)

# Delinquency Analysis
st.subheader("Delinquency Analysis")
col1, col2 = st.columns(2)

with col1:
    # Delinquency Status Distribution
    delinq_counts = filtered_df['delinquency_status'].value_counts().reset_index()
    delinq_counts.columns = ['Delinquency Status', 'Count']
    
    # Set order for delinquency status
    status_order = ['Current', '30 Days', '60 Days', '90+ Days', 'Default']
    delinq_counts['Delinquency Status'] = pd.Categorical(
        delinq_counts['Delinquency Status'], 
        categories=status_order, 
        ordered=True
    )
    delinq_counts = delinq_counts.sort_values('Delinquency Status')
    
    fig = px.bar(
        delinq_counts,
        x='Delinquency Status',
        y='Count',
        title='Loan Count by Delinquency Status',
        color='Delinquency Status',
        color_discrete_map={
            'Current': 'green',
            '30 Days': 'yellow',
            '60 Days': 'orange',
            '90+ Days': 'red',
            'Default': 'black'
        }
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Delinquency by Collateral Type
    delinq_by_collateral = filtered_df.pivot_table(
        index='collateral_type',
        columns='delinquency_status',
        values='loan_id',
        aggfunc='count',
        fill_value=0
    )
    
    # Ensure all delinquency statuses are present
    for status in status_order:
        if status not in delinq_by_collateral.columns:
            delinq_by_collateral[status] = 0
    
    delinq_by_collateral = delinq_by_collateral[status_order]
    
    # Calculate percentages
    delinq_pct = delinq_by_collateral.div(delinq_by_collateral.sum(axis=1), axis=0)
    
    # Convert to long format for plotting
    delinq_pct_long = delinq_pct.reset_index().melt(
        id_vars='collateral_type',
        value_vars=status_order,
        var_name='Delinquency Status',
        value_name='Percentage'
    )
    
    fig = px.bar(
        delinq_pct_long,
        x='collateral_type',
        y='Percentage',
        color='Delinquency Status',
        title='Delinquency Status Distribution by Collateral Type',
        labels={'collateral_type': 'Collateral Type', 'Percentage': 'Percentage of Loans'},
        color_discrete_map={
            'Current': 'green',
            '30 Days': 'yellow',
            '60 Days': 'orange',
            '90+ Days': 'red',
            'Default': 'black'
        }
    )
    fig.update_layout(yaxis_tickformat='.0%')
    st.plotly_chart(fig, use_container_width=True)

# Vintage Analysis
st.subheader("Vintage Analysis")

# Prepare data for vintage analysis
vintage_delinq = filtered_df.pivot_table(
    index='vintage',
    columns='delinquency_status',
    values='loan_id',
    aggfunc='count',
    fill_value=0
)

# Ensure all delinquency statuses are present
for status in status_order:
    if status not in vintage_delinq.columns:
        vintage_delinq[status] = 0

vintage_delinq = vintage_delinq[status_order]

# Calculate percentages
vintage_delinq_pct = vintage_delinq.div(vintage_delinq.sum(axis=1), axis=0)

# Create combined metric (delinquency index)
vintage_delinq_pct['Delinquency Index'] = (
    vintage_delinq_pct['30 Days'] * 1 +
    vintage_delinq_pct['60 Days'] * 2 +
    vintage_delinq_pct['90+ Days'] * 3 +
    vintage_delinq_pct['Default'] * 4
)

# Sort by vintage
vintage_delinq_pct = vintage_delinq_pct.reset_index()
vintage_delinq_pct['vintage_sort'] = pd.Categorical(
    vintage_delinq_pct['vintage'],
    categories=sorted(vintage_delinq_pct['vintage'].unique()),
    ordered=True
)
vintage_delinq_pct = vintage_delinq_pct.sort_values('vintage_sort')

# Plot vintage delinquency index
fig = px.line(
    vintage_delinq_pct,
    x='vintage',
    y='Delinquency Index',
    title='Delinquency Index by Vintage (Higher = Worse Performance)',
    markers=True
)
st.plotly_chart(fig, use_container_width=True)

# IRR and WAL Analysis
st.subheader("Risk-Adjusted Return Analysis")
col1, col2 = st.columns(2)

with col1:
    # Scatter plot of IRR vs WAL
    fig = px.scatter(
        filtered_df,
        x='wal',
        y='irr',
        color='collateral_type',
        size='remaining_balance',
        hover_data=['loan_id', 'fico_score', 'delinquency_status', 'interest_rate'],
        title='IRR vs WAL by Collateral Type',
        labels={'wal': 'Weighted Average Life (months)', 'irr': 'Internal Rate of Return'},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(yaxis_tickformat='.2%')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # FICO Score vs IRR
    fig = px.scatter(
        filtered_df,
        x='fico_score',
        y='irr',
        color='delinquency_status',
        size='remaining_balance',
        hover_data=['loan_id', 'collateral_type', 'interest_rate'],
        title='FICO Score vs IRR by Delinquency Status',
        labels={'fico_score': 'FICO Score', 'irr': 'Internal Rate of Return'},
        color_discrete_map={
            'Current': 'green',
            '30 Days': 'yellow',
            '60 Days': 'orange',
            '90+ Days': 'red',
            'Default': 'black'
        }
    )
    fig.update_layout(yaxis_tickformat='.2%')
    st.plotly_chart(fig, use_container_width=True)

# Loan-level Data with AgGrid
st.subheader("Loan-Level Data")

# Format data for display
display_df = filtered_df.copy()
display_df['origination_date'] = display_df['origination_date'].dt.date
display_df['interest_rate'] = display_df['interest_rate'].apply(lambda x: f"{x:.2%}")
display_df['purchase_price_pct'] = display_df['purchase_price_pct'].apply(lambda x: f"{x:.2%}")
display_df['irr'] = display_df['irr'].apply(lambda x: f"{x:.2%}")
display_df['remaining_balance'] = display_df['remaining_balance'].apply(lambda x: f"${x:,.2f}")
display_df['loan_amount'] = display_df['loan_amount'].apply(lambda x: f"${x:,.2f}")

# Select columns for display
display_columns = [
    'loan_id', 'origination_date', 'vintage', 'collateral_type', 
    'loan_amount', 'remaining_balance', 'loan_term', 'wal',
    'interest_rate', 'irr', 'fico_score', 'delinquency_status', 'purchase_price_pct'
]
display_df = display_df[display_columns]

# Configure AgGrid
gb = GridOptionsBuilder.from_dataframe(display_df)
gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=15)
gb.configure_column('loan_id', headerCheckboxSelection=True)
gb.configure_selection('multiple', use_checkbox=True)
gb.configure_default_column(
    groupable=True, 
    value=True, 
    enableRowGroup=True, 
    aggFunc='count', 
    editable=False
)

# Cell styling
delinquency_cellstyle = JsCode("""
function(params) {
    if (params.value === 'Current') {
        return {'color': 'white', 'backgroundColor': 'green'};
    } else if (params.value === '30 Days') {
        return {'color': 'black', 'backgroundColor': 'yellow'};
    } else if (params.value === '60 Days') {
        return {'color': 'black', 'backgroundColor': 'orange'};
    } else if (params.value === '90+ Days') {
        return {'color': 'white', 'backgroundColor': 'red'};
    } else if (params.value === 'Default') {
        return {'color': 'white', 'backgroundColor': 'black'};
    }
    return {'color': 'black', 'backgroundColor': 'white'};
}
""")

gb.configure_column('delinquency_status', cellStyle=delinquency_cellstyle)

grid_options = gb.build()
ag_grid = AgGrid(
    display_df,
    gridOptions=grid_options,
    enable_enterprise_modules=True,
    allow_unsafe_jscode=True,
    update_on_grid_options_change=True
)

# Show selected rows
selected_rows = ag_grid['selected_rows']
if selected_rows:
    st.write(f"Selected {len(selected_rows)} loans for detailed analysis")
    
    # Show detailed info for selected loans
    selected_df = pd.DataFrame(selected_rows)
    
    # Calculate summary metrics
    total_balance = selected_df['remaining_balance'].str.replace('$', '').str.replace(',', '').astype(float).sum()
    
    st.metric(
        "Selected Portfolio Value", 
        f"${total_balance:,.2f}",
        delta=f"{total_balance / filtered_df['remaining_balance'].str.replace('$', '').str.replace(',', '').astype(float).sum():.2%} of filtered portfolio"
    )