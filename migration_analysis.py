import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# CONFIGURATION & STYLE
# ==========================================
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

FILES = {
    'stock': 'migr_resvalid.xlsx',
    'inflow': 'migr_resfirst.xlsx',
    'naturalization': 'migr_acq.xlsx',
    'emigration_off': 'migr_emi1ctz.xlsx'
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
def calculate_demographics(df_stock, df_inflow, df_nat, df_emi_off):
    """
    Merges dataframes and calculates derived metrics based on the 
    Demographic Balancing Equation.
    """
    # Merge all dataframes on Country and Year
    df = df_stock.merge(df_inflow, on=['Country', 'Year'], how='outer') \
                 .merge(df_nat, on=['Country', 'Year'], how='outer') \
                 .merge(df_emi_off, on=['Country', 'Year'], how='outer')
    
    # Sort strictly by Country and Year for correct lag calculations
    df.sort_values(by=['Country', 'Year'], inplace=True)
    
    # 1. Calculate S_{t-1} (Previous Year Stock)
    df['Stock_Prev'] = df.groupby('Country')['Stock'].shift(1)
    
    # 2. Derived Total Outflow (E_total)
    # Formula: E_total = S_{t-1} + I_t - N_t - S_t
    df['Emigration_Total'] = df['Stock_Prev'] + df['Inflow'] - df['Naturalization'] - df['Stock']
    
    # 3. Hidden Emigration
    # Formula: E_hidden = E_total - E_official
    # Note: We allow negative values as they indicate data anomalies or unrecorded inflows
    df['Emigration_Hidden'] = df['Emigration_Total'] - df['Emigration_Official']
    
    # 4. Theoretical Stock (Migration Gap Analysis)
    # S_theo,t = S_base + Sum(Inflow)
    # We define S_base as the Stock at the first available year for that country.
    def calculate_theoretical(g):
        g = g.sort_values('Year')
        # Find first year with valid Stock data
        valid_stock = g['Stock'].dropna()
        if valid_stock.empty:
            g['Theoretical_Stock'] = np.nan
            return g
        
        # Base Year & Stock
        start_idx = valid_stock.index[0]
        base_stock = valid_stock.iloc[0]
        start_year = g.loc[start_idx, 'Year']
        
        # Cumulative Inflow (only count inflows occurring AFTER the base year measurement)
        inflows_future = g['Inflow'].fillna(0).where(g['Year'] > start_year, 0)
        g['Theoretical_Stock'] = base_stock + inflows_future.cumsum()
        return g

    df = df.groupby('Country', group_keys=False).apply(calculate_theoretical)
    
    return df

# ==========================================
# STEP 3: VISUALIZATION
# ==========================================
def plot_migration_gap_dashboard(df, country):
    """
    Generates a 2-subplot dashboard for the 'Italian Gap' approach.
    1. Dynamics: Stock vs Inflow (Dual Axis)
    2. Gap Analysis: Actual vs Theoretical Stock
    """
    country_data = df[df['Country'] == country]
    if country_data.empty:
        print(f"No data found for {country}")
        return

    # Ensure data is sorted
    country_data = country_data.sort_values('Year')
    years = country_data['Year']

    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # --- SUBPLOT 1: DYNAMICS (Stock vs Flow) ---
    # Left Axis: Inflow (Bar)
    ax1.bar(years, country_data['Inflow'], color='tab:red', alpha=0.6, label='Inflow ($I_t$)')
    ax1.set_ylabel('Annual Inflow (Persons)', color='tab:red', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='tab:red')
    
    # Right Axis: Stock (Line)
    ax2 = ax1.twinx()
    ax2.plot(years, country_data['Stock'], color='darkblue', linewidth=2, label='Stock ($S_t$)')
    ax2.set_ylabel('Total Stock (Persons)', color='darkblue', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='darkblue')
    
    ax1.set_title(f'Migration Dynamics: {country}', fontsize=14, fontweight='bold')
    
    # Annotations (Geopolitical Events)
    for year in [2014, 2022]:
        if year in years.values:
            ax1.axvline(x=year, color='black', linestyle='--', alpha=0.5)
            ax1.text(year, ax1.get_ylim()[1], f' {year}', rotation=90, verticalalignment='top')

    # --- SUBPLOT 2: THE GAP (Actual vs Theoretical) ---
    # Plot Actual
    ax3.plot(years, country_data['Stock'], color='tab:blue', linewidth=2, label='Actual Stock')
    
    # Plot Theoretical
    ax3.plot(years, country_data['Theoretical_Stock'], color='grey', linestyle='--', label='Theoretical Stock ($S_{theo}$)')
    
    # Fill the Gap
    # Handle NaNs for fill_between by dropping them temporarily for the fill logic if needed, 
    # but matplotlib usually handles NaNs by breaking the fill.
    ax3.fill_between(years, 
                     country_data['Stock'], 
                     country_data['Theoretical_Stock'], 
                     color='grey', alpha=0.3, 
                     label='Attrition (Departure/Naturalization)')
    
    ax3.set_ylabel('Population Stock', fontsize=12)
    ax3.set_xlabel('Year', fontsize=12)
    ax3.set_title('Migration Gap Analysis', fontsize=14)
    ax3.legend(loc='upper left')
    ax3.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

def main():
    try:
        # 1. Load Data
        print("Loading and cleaning data...")
        df_s = load_and_process_file(FILES['stock'], 'Stock')
        df_i = load_and_process_file(FILES['inflow'], 'Inflow')
        df_n = load_and_process_file(FILES['naturalization'], 'Naturalization')
        df_e = load_and_process_file(FILES['emigration_off'], 'Emigration_Official')
        
        # 2. Calculate Metrics
        print("Calculating demographic metrics...")
        df_main = calculate_demographics(df_s, df_i, df_n, df_e)
        
        if not df_main.empty:
            # 3. Filter Top 10 Countries by Maximum Stock
            print("Identifying Top 10 countries by Stock...")
            top_countries = df_main.groupby('Country')['Stock'].max().nlargest(10).index.tolist()
            print(f"Top Countries: {top_countries}")
            
            # 4. Generate Dashboards
            print("Generating Migration Gap Dashboards...")
            for country in top_countries:
                plot_migration_gap_dashboard(df_main, country)
            
            print("Analysis Complete.")
        else:
            print("DataFrame is empty after processing.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()