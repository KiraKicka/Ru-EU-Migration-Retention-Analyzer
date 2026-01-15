import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display
import os

# ==========================================
# CONFIGURATION & STYLE
# ==========================================
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

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
    df = df.groupby('Country', group_keys=False).apply(calculate_theoretical_extended)
    
    return df

# ==========================================
# STEP 3: VISUALIZATION
# ==========================================
def plot_gap_decomposition(country, df):
    """
    Renders the Gap Decomposition chart for a specific country.
    """
    data = df[df['Country'] == country].sort_values('Year')
    
    if data.empty or data['Stock'].isna().all():
        print(f"Insufficient data for {country}")
        return

    plt.figure(figsize=(12, 7))
    
    years = data['Year']
    
    # Layer 1: Theoretical Max (Grey Line)
    plt.plot(years, data['Theoretical_Max'], color='grey', linewidth=1.5, label='Theoretical Max (Zero Exit)')
    
    # Layer 2: Theoretical Adj (Gold Dashed)
    plt.plot(years, data['Theoretical_Adj'], color='gold', linestyle='--', linewidth=2, label='Theoretical (Citizenship Adj.)')
    
    # Layer 3: Actual Stock (Blue Solid)
    plt.plot(years, data['Stock'], color='#003366', linewidth=2.5, label='Actual Resident Stock')
    
    # Area 1: Naturalized (Integration) - Gold Fill
    plt.fill_between(years, data['Theoretical_Max'], data['Theoretical_Adj'],
                     color='gold', alpha=0.2, label='Naturalized (Integration)')
    
    # Area 2: Emigration (Loss) - Red Fill (Where Adj > Stock)
    plt.fill_between(years, data['Theoretical_Adj'], data['Stock'],
                     where=(data['Theoretical_Adj'] > data['Stock']),
                     color='tab:red', alpha=0.3, label='Emigration (Loss)')
    
    # Area 3: Net Gain/Refugee (Green Fill) - (Where Stock > Adj)
    plt.fill_between(years, data['Stock'], data['Theoretical_Adj'],
                     where=(data['Stock'] > data['Theoretical_Adj']),
                     color='tab:green', alpha=0.3, label='Net Gain/Refugee Effect')
    
    # Annotations (Geopolitical Events)
    for year in [2014, 2022]:
        if year in years.values:
            plt.axvline(x=year, color='black', linestyle=':', alpha=0.6)
            plt.text(year, plt.ylim()[1]*0.95, f' {year}', rotation=90, va='top')
            
    plt.title(f'Migration Gap Decomposition: {country}', fontsize=16, fontweight='bold')
    plt.ylabel('Population Stock', fontsize=12)
    plt.xlabel('Year', fontsize=12)
    plt.legend(loc='upper left', frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

def main():
    try:
        df = process_data()
        
        # Get list of countries with valid data
        valid_countries = df.dropna(subset=['Stock']).groupby('Country')['Stock'].max().sort_values(ascending=False).index.tolist()
        
        if not valid_countries:
            print("No valid data found.")
            return

        print("\nStarting Interactive Dashboard...")
        print("Select a country from the dropdown below.")
        
        # Create Interactive Widget
        dropdown = widgets.Dropdown(
            options=valid_countries,
            value=valid_countries[0] if valid_countries else None,
            description='Country:',
            disabled=False,
        )
        
        # Use interactive_output to link the dropdown to the plot function
        # We pass 'df' as a fixed argument
        ui = widgets.VBox([dropdown])
        out = widgets.interactive_output(plot_gap_decomposition, {'country': dropdown, 'df': widgets.fixed(df)})
        
        display(ui, out)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()