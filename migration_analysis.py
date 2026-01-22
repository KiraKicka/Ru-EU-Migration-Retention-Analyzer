import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import RadioButtons
import os
from functools import reduce

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
    'naturalization': 'migr_acq.xlsx',
    'asylum_pending': 'migr_asypenctzm.xlsx',
    'change_status': 'migr_reschange.xlsx',
    'long_term': 'migr_reslong.xlsx',
    'blue_card': 'migr_resbc13.xlsx'
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
    
    # Handle Monthly Data (e.g., 2008M12) for Asylum Pending
    # Check if columns contain 'M' (excluding the Country column)
    data_cols = df.columns[1:]
    if any('M' in str(c) for c in data_cols):
        # Filter for December (M12) to get year-end stock
        m12_cols = [c for c in data_cols if str(c).endswith('M12')]
        df = df[['Country'] + m12_cols]
        # Rename columns: "2008M12" -> "2008"
        df.columns = ['Country'] + [str(c).replace('M12', '') for c in m12_cols]

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
    data_frames = [
        load_and_process_file(FILES['stock'], 'Stock'),
        load_and_process_file(FILES['inflow'], 'Inflow'),
        load_and_process_file(FILES['naturalization'], 'Naturalization'),
        load_and_process_file(FILES['asylum_pending'], 'Pending'),
        load_and_process_file(FILES['change_status'], 'ChangeStatus'),
        load_and_process_file(FILES['long_term'], 'LongTerm'),
        load_and_process_file(FILES['blue_card'], 'BlueCard')
    ]
    
    # Merge all datasets
    print("Merging and processing...")
    df = reduce(lambda left, right: pd.merge(left, right, on=['Country', 'Year'], how='outer'), data_frames)
    
    # Filter Timeframe (Year >= 2008)
    df = df[df['Year'] >= 2008]
    
    # Interpolate small gaps
    # Stock: Do NOT fillna(0) to allow dynamic start year (e.g. if 2008 is missing)
    df['Stock'] = df.groupby('Country')['Stock'].transform(
        lambda x: x.interpolate(method='linear', limit=1)
    )

    # Others: Fill NaNs with 0 as they are likely zero if missing
    for col in ['Pending', 'LongTerm']:
        df[col] = df.groupby('Country')[col].transform(
            lambda x: x.interpolate(method='linear', limit=1).fillna(0)
        )

    # Calculate Total De Facto Stock (Official + Pending)
    df['Total_De_Facto'] = df['Stock'] + df['Pending']

    # Apply Gap Decomposition Logic
    # Explicitly select columns to preserve them after apply
    cols = ['Year', 'Stock', 'Inflow', 'Naturalization', 'Pending', 
            'Total_De_Facto', 'LongTerm', 'ChangeStatus', 'BlueCard']
    df = df.groupby('Country', group_keys=True)[cols].apply(calculate_theoretical_extended)
    df = df.reset_index(level='Country')
    
    return df

# ==========================================
# STEP 2.5: REPORTING LOGIC
# ==========================================
def calculate_retention_ranking(df):
    """
    Calculates:
    1. Standard Retention Rate (CR)
    2. Forensic Retention Rate (De Facto CR)
    3. Anchoring Ratio (Long-Term / Stock)
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
        
        # 1. Standard Retention Rate
        cr = ((delta_stock + sum_nat) / sum_inflow) * 100
        
        # 2. Forensic Retention Rate (Using Total De Facto Stock)
        start_defacto = group.loc[valid_stock.index[0], 'Total_De_Facto']
        end_defacto = group.loc[valid_stock.index[-1], 'Total_De_Facto']
        delta_defacto = end_defacto - start_defacto
        
        cr_forensic = ((delta_defacto + sum_nat) / sum_inflow) * 100

        # 3. Anchoring Ratio (Latest available)
        last_long_term = group.loc[valid_stock.index[-1], 'LongTerm']
        anchoring_ratio = (last_long_term / end_val) * 100 if end_val > 0 else 0

        # Implied Emigration (Standard)
        e_implied = sum_inflow - (delta_stock + sum_nat)
        
        status = "Anchor" if cr >= 80 else "Transit"
        
        results.append({
            'Country': country,
            'Retention_Rate': cr,
            'Forensic_Rate': cr_forensic,
            'Anchoring_Ratio': anchoring_ratio
        })
        
    return pd.DataFrame(results).sort_values('PCAR', ascending=True) # Sort by lowest attrition

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
    
    # Layer 3: Actual Stock (Blue Solid) - Base
    ax_main.plot(years, data['Stock'], color='#003366', linewidth=2, label='Official Resident Stock')
    
    # Layer 4: Pending Asylum (Stacked on Stock)
    ax_main.plot(years, data['Total_De_Facto'], color='purple', linewidth=1, linestyle=':', label='De Facto (Incl. Pending)')
    ax_main.fill_between(years, data['Stock'], data['Total_De_Facto'], color='purple', alpha=0.3, label='Pending Asylum (Invisible)')
    
    # Area 1: Naturalized (Integration) - Gold Fill
    ax_main.fill_between(years, data['Theoretical_Max'], data['Theoretical_Adj'],
                     color='gold', alpha=0.2, label='Naturalized (Integration)')
    
    # Area 2: Emigration (Loss) - Red Fill
    # Gap is now between Theoretical_Adj and Total_De_Facto (Forensic View)
    ax_main.fill_between(years, data['Theoretical_Adj'], data['Total_De_Facto'],
                     where=(data['Theoretical_Adj'] > data['Total_De_Facto']),
                     color='tab:red', alpha=0.3, label='Implied Emigration')
    
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
    
    # Layer 5: Naturalization (Gold)
    ax_bottom.plot(years, data['Naturalization'], color='gold', linestyle='-', linewidth=2, label='Naturalization (Passports)')
    
    # Explicitly scale Y-axis to fit data (start at 0)
    max_val = data[['Inflow', 'Naturalization']].max().max()
    if not pd.isna(max_val) and max_val > 0:
        ax_bottom.set_ylim(bottom=0, top=max_val * 1.1)
    
    ax_bottom.set_ylabel('Annual Flows', fontsize=12)
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

        # Calculate ranking data for GUI
        df_ranking = calculate_retention_ranking(df)

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
        
        # Stats Panel (New)
        stats_ax = plt.axes([0.02, 0.05, 0.25, 0.15], facecolor='#f0f0f0')
        stats_ax.axis('off')
        
        def update(label):
            plot_gap_decomposition(ax_main, ax_bottom, label, df)
            
            # Update Stats Panel
            stats_ax.clear()
            stats_ax.axis('off')
            
            country_stats = df_ranking[df_ranking['Country'] == label]
            if not country_stats.empty:
                row = country_stats.iloc[0]
                text_str = (f"Metrics for {label}:\n\n"
                            f"Std CR:      {row['Retention_Rate']:.1f}%\n"
                            f"Forensic CR: {row['Forensic_Rate']:.1f}%\n"
                            f"Anchoring:   {row['Anchoring_Ratio']:.1f}%")
            else:
                text_str = f"Metrics for {label}:\nN/A"
                
            stats_ax.text(0.05, 0.9, text_str, transform=stats_ax.transAxes, 
                          fontsize=10, verticalalignment='top', fontfamily='monospace')
            
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