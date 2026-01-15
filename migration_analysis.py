import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import RadioButtons
import os

# ==========================================
# CONFIGURATION & STYLE
# ==========================================
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Fix FutureWarning for downcasting
pd.set_option('future.no_silent_downcasting', True)

FILES = {
    'stock': 'migr_resvalid.xlsx',
    'inflow': 'migr_resfirst.xlsx',
    'naturalization': 'migr_acq.xlsx'
}

# ==========================================
# STEP 1: DATA LOADING & CLEANING
# ==========================================
def load_and_process_file(filepath, value_name):
    """
    Loads a CSV, cleans it using vectorized operations, and melts it into Long Format.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # Load data
    # Column 0 is Country, Columns 1..N are Years
    df = pd.read_excel(filepath)
    
    # Rename first column to 'Country' for consistency
    df.rename(columns={df.columns[0]: 'Country'}, inplace=True)
    
    # Melt: Turn Year columns into rows
    year_cols = df.columns[1:]
    df_melted = df.melt(id_vars=['Country'], value_vars=year_cols, 
                        var_name='Year', value_name=value_name)
    
    # Vectorized Cleaning
    # 1. Replace ':' with NaN
    # 2. Convert to numeric (coercing any errors to NaN)
    df_melted[value_name] = pd.to_numeric(
        df_melted[value_name].replace(':', np.nan), 
        errors='coerce'
    )
    
    # Clean Year (ensure it's integer, handling potential string headers)
    df_melted['Year'] = pd.to_numeric(df_melted['Year'], errors='coerce').astype('Int64')
    
    # Drop rows where Year parsing failed (if any)
    df_melted = df_melted.dropna(subset=['Year'])
    
    return df_melted

# ==========================================
# STEP 2: CALCULATION LOGIC
# ==========================================
def calculate_theoretical_extended(g):
    """
    Splits the 'Gap' into Naturalization (Integration) and Emigration (Loss).
    """
    g = g.sort_values('Year')
    valid_stock = g['Stock'].dropna()
    if valid_stock.empty:
        # Initialize columns with NaN if no stock data
        g['Theoretical_Max'] = np.nan
        g['Theoretical_Adj'] = np.nan
        return g
    
    start_year = g.loc[valid_stock.index[0], 'Year']
    base_stock = valid_stock.iloc[0]
    
    # Calculate cumulative flows ONLY after the base year
    mask = g['Year'] > start_year
    inflow_cum = g['Inflow'].fillna(0).where(mask, 0).cumsum()
    nat_cum = g['Naturalization'].fillna(0).where(mask, 0).cumsum()
    
    # Level 1: Theoretical Max (Nobody leaves, nobody changes passport)
    g['Theoretical_Max'] = base_stock + inflow_cum
    
    # Level 2: Citizenship Adjusted (Nobody leaves, but some get passports)
    # S_adj = S_max - Cumulative_Naturalization
    g['Theoretical_Adj'] = g['Theoretical_Max'] - nat_cum
    
    return g

def process_data():
    """
    Loads, merges, filters, and calculates metrics.
    """
    print("Loading data...")
    df_s = load_and_process_file(FILES['stock'], 'Stock')
    df_i = load_and_process_file(FILES['inflow'], 'Inflow')
    df_n = load_and_process_file(FILES['naturalization'], 'Naturalization')
    
    # Merge
    print("Merging and processing...")
    df = df_s.merge(df_i, on=['Country', 'Year'], how='outer') \
             .merge(df_n, on=['Country', 'Year'], how='outer')
    
    # Filter Timeframe (Year >= 2008)
    df = df[df['Year'] >= 2008]
    
    # Interpolate small gaps in Stock for better visualization
    # (Group by country first to avoid cross-country interpolation)
    df['Stock'] = df.groupby('Country')['Stock'].transform(lambda x: x.interpolate(method='linear', limit=1))

    # Apply Gap Decomposition Logic
    # Explicitly select columns to avoid DeprecationWarning in pandas 2.1+ and TypeError in <2.2
    cols = ['Year', 'Stock', 'Inflow', 'Naturalization']
    df = df.groupby('Country', group_keys=True)[cols].apply(calculate_theoretical_extended)
    df = df.reset_index(level='Country')
    
    return df

# ==========================================
# STEP 2.5: REPORTING LOGIC
# ==========================================
def calculate_retention_ranking(df):
    """
    Calculates the Aggregate Retention Rate (CR) for each country.
    CR = (Delta_Stock + Sum_Naturalization) / Sum_Inflow
    """
    results = []
    for country, group in df.groupby('Country'):
        # Filter for valid range (start to end)
        group = group.sort_values('Year')
        valid_stock = group['Stock'].dropna()
        
        if valid_stock.empty:
            continue
            
        start_val = valid_stock.iloc[0]
        end_val = valid_stock.iloc[-1]
        
        # Sum flows over the period where stock exists
        start_year = group.loc[valid_stock.index[0], 'Year']
        end_year = group.loc[valid_stock.index[-1], 'Year']
        
        period_mask = (group['Year'] > start_year) & (group['Year'] <= end_year)
        sum_inflow = group.loc[period_mask, 'Inflow'].sum()
        sum_nat = group.loc[period_mask, 'Naturalization'].sum()
        
        if sum_inflow < 1000: # Filter noise
            continue
            
        delta_stock = end_val - start_val
        
        # Retention Rate Formula
        cr = ((delta_stock + sum_nat) / sum_inflow) * 100
        
        # Implied Emigration (Total Attrition)
        # E_implied = Sum_Inflow - (Delta_Stock + Sum_Nat)
        e_implied = sum_inflow - (delta_stock + sum_nat)
        
        status = "Anchor" if cr >= 80 else "Transit"
        
        results.append({
            'Country': country,
            'Retention_Rate': cr,
            'Implied_Emigration': e_implied,
            'Status': status
        })
        
    return pd.DataFrame(results).sort_values('Retention_Rate', ascending=False)

# ==========================================
# STEP 3: VISUALIZATION
# ==========================================
def plot_gap_decomposition(ax_main, ax_bottom, country, df):
    """
    Renders the Gap Decomposition chart for a specific country.
    """
    data = df[df['Country'] == country].sort_values('Year')
    
    if data.empty or data['Stock'].isna().all():
        ax_main.text(0.5, 0.5, f"Insufficient data for {country}", 
                ha='center', va='center', transform=ax_main.transAxes)
        ax_bottom.clear()
        ax_bottom.axis('off')
        return

    ax_main.clear()
    ax_bottom.clear()
    ax_bottom.axis('on')
    
    years = data['Year']
    
    # --- MAIN PLOT (Stock & Gap) ---
    # Layer 1: Theoretical Max (Grey Line)
    ax_main.plot(years, data['Theoretical_Max'], color='grey', linewidth=1.5, label='Theoretical Max (Zero Exit)')
    
    # Layer 2: Theoretical Adj (Gold Dashed)
    ax_main.plot(years, data['Theoretical_Adj'], color='gold', linestyle='--', linewidth=2, label='Theoretical (Citizenship Adj.)')
    
    # Layer 3: Actual Stock (Blue Solid)
    ax_main.plot(years, data['Stock'], color='#003366', linewidth=2.5, label='Actual Resident Stock')
    
    # Area 1: Naturalized (Integration) - Gold Fill
    ax_main.fill_between(years, data['Theoretical_Max'], data['Theoretical_Adj'],
                     color='gold', alpha=0.2, label='Naturalized (Integration)')
    
    # Area 2: Emigration (Loss) - Red Fill (Where Adj > Stock)
    ax_main.fill_between(years, data['Theoretical_Adj'], data['Stock'],
                     where=(data['Theoretical_Adj'] > data['Stock']),
                     color='tab:red', alpha=0.3, label='Emigration (Loss)')
    
    # Annotations (Geopolitical Events)
    for year in [2014, 2022]:
        if year in years.values:
            # Main Plot Annotation
            ax_main.axvline(x=year, color='black', linestyle=':', alpha=0.6)
            # Use current ylim for text placement
            y_max = ax_main.get_ylim()[1]
            ax_main.text(year, y_max*0.95, f' {year}', rotation=90, va='top')
            
            # Bottom Plot Annotation
            ax_bottom.axvline(x=year, color='black', linestyle=':', alpha=0.6)
            
    ax_main.set_title(f'Migration Gap Decomposition: {country}', fontsize=16, fontweight='bold')
    ax_main.set_ylabel('Population Stock', fontsize=12)
    ax_main.yaxis.set_label_position("right")
    ax_main.yaxis.tick_right()
    ax_main.legend(loc='upper left', frameon=True)
    ax_main.grid(True, linestyle=':', alpha=0.6)
    
    # Hide X labels on main plot (shared axis)
    plt.setp(ax_main.get_xticklabels(), visible=False)

    # --- BOTTOM PLOT (Inflow) ---
    # Layer 4: Inflow (Teal)
    ax_bottom.plot(years, data['Inflow'], color='teal', linestyle='-', linewidth=2, label='Inflow (First Permits)')
    
    # Explicitly scale Y-axis to fit data (start at 0)
    if not data['Inflow'].dropna().empty:
        ax_bottom.set_ylim(bottom=0, top=data['Inflow'].max() * 1.1)
    
    ax_bottom.set_ylabel('Inflow', fontsize=12)
    ax_bottom.yaxis.set_label_position("right")
    ax_bottom.yaxis.tick_right()
    ax_bottom.set_xlabel('Year', fontsize=12)
    ax_bottom.legend(loc='upper left', frameon=True)
    ax_bottom.grid(True, linestyle=':', alpha=0.6)

def main():
    try:
        df = process_data()
        
        # Get list of countries with valid data
        valid_countries = df.dropna(subset=['Stock']).groupby('Country')['Stock'].max().sort_values(ascending=False).index.tolist()
        
        if not valid_countries:
            print("No valid data found.")
            return

        # Limit to top 15 for UI clarity
        top_countries = valid_countries[:15]

        # Print Demographic Report
        print("\n=== DEMOGRAPHIC REPORT (Retention Ranking) ===")
        df_ranking = calculate_retention_ranking(df)
        for _, row in df_ranking.iterrows():
            if row['Country'] in top_countries:
                print(f"Country: {row['Country']:<15} | Retention Rate: {row['Retention_Rate']:>6.1f}% | "
                      f"Implied Emigration: {row['Implied_Emigration']:>8.0f} | Status: {row['Status']}")
        print("==============================================\n")

        print("\nStarting Dashboard...")
        print(f"Loaded {len(top_countries)} countries. Check the popup window.")
        
        # Create Figure and Axes with GridSpec
        fig = plt.figure()
        gs = fig.add_gridspec(4, 1)
        
        # Top subplot (3/4 height)
        ax_main = fig.add_subplot(gs[0:3, :])
        
        # Bottom subplot (1/4 height), sharing X axis
        ax_bottom = fig.add_subplot(gs[3, :], sharex=ax_main)
        
        plt.subplots_adjust(left=0.3) # Make room for sidebar
        
        # Sidebar for RadioButtons
        rax = plt.axes([0.02, 0.2, 0.25, 0.6], facecolor='#f0f0f0')
        radio = RadioButtons(rax, top_countries)
        
        def update(label):
            plot_gap_decomposition(ax_main, ax_bottom, label, df)
            fig.canvas.draw_idle()

        radio.on_clicked(update)

        # Initial plot
        update(top_countries[0])

        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()