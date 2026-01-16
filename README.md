<h1 align="center">Ru-EU Migration Retention Analyzer (DBNA Model)</h1>

<p align="center">
    <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python 3.8+">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
    <img src="https://img.shields.io/badge/Status-For%20Fun-purple" alt="Status: For Fun">
    <img src="https://img.shields.io/badge/Pandas-2.x-blue.svg" alt="Pandas">
    <img src="https://img.shields.io/badge/Matplotlib-3.x-orange.svg" alt="Matplotlib">
</p>

This project is a tool for demographic analysis of migration processes between Russia and the European Union countries. Instead of simply counting arrivals, it focuses on the **Retention Rate** — a key indicator that shows what percentage of immigrants remain in the destination country on a long-term basis.

The analysis covers the key countries of attraction for Russian citizens: **Germany, Spain, France, and Italy** for the period since 2008.

## 📊 Methodology: DBNA (Demographic Balancing with Naturalization Adjustment)

Standard "inflow minus outflow" analysis methods do not work for EU migration statistics due to two problems:
1.  **Unreliable emigration data (`migr_emi`)**: Many countries, including France, hardly keep any records of foreigners who have left.
2.  **Distortion due to naturalization (`migr_acq`)**: When an immigrant obtains citizenship, they disappear from the "foreign population" statistics, which can be mistakenly interpreted as emigration.

To solve these problems, the **Demographic Balancing with Naturalization Adjustment (DBNA) Method** is used. We reconstruct the "hidden" emigration based on the annual balance.

**The final formula for the Retention Rate (CR):**

$$CR = \left( 1 - \frac{\max(0, E_{implied})}{P_t + I_{(t)}} \right) \times 100\%$$

Where:
-   $P_t$: Number of residents at the beginning of the year (`migr_resvalid`).
-   $I_{(t)}$: Inflow of new immigrants during the year (`migr_resfirst`).
-   $E_{implied}$: "Implied emigration," calculated as the difference between the expected and actual population figures, adjusted for naturalization.

This approach allows for an assessment of the real stability of the migrant contingent, considering the acquisition of citizenship not as a loss, but as the highest form of retention.

## 🖼️ Demonstration

The script launches an interactive `Matplotlib` panel where you can visually assess the gap between the theoretical and actual population for each country.

-   **Left panel**: Allows you to select a country for analysis.
-   **Top chart (Gap Decomposition)**:
    -   `Actual Resident Stock`: The real number of Russians with residence permits.
    -   `Theoretical (Citizenship Adj.)`: The expected number of residents if no one had left (adjusted for naturalization).
    -   **Gold area**: "Integration" — residents who have become citizens.
    -   **Red area**: "Losses" — presumed emigration.
-   **Bottom chart**: Dynamics of the annual inflow (issued first residence permits).

## 🛠️ Installation and requirements

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-repository/migration-analyzer.git
    cd migration-analyzer
    ```

2.  **Install dependencies:**
    The script requires Python 3.8+ and the following libraries. Install them using `pip`:
    ```bash
    pip install pandas numpy matplotlib seaborn openpyxl
    ```

3.  **Prepare the data:**
    Download the necessary datasets from the Eurostat database and place them in the root directory of the project under the following names:
    -   `migr_resvalid.xlsx`: Number of valid residence permits (Stock).
    -   `migr_resfirst.xlsx`: Issued first residence permits (Inflow).
    -   `migr_acq.xlsx`: Acquisition of citizenship (Naturalization).

## 🚀 Usage

To run the analysis, execute the script from the command line:

```bash
python migration_analysis.py
```

Execution process:
1.  **Console report**: A ranking of countries by **Retention Rate** will be printed to the terminal.
    ```
    === DEMOGRAPHIC REPORT (Retention Ranking) ===
    Country: France          | Retention Rate:  99.8% | Implied Emigration:       98 | Status: Anchor
    Country: Germany         | Retention Rate:  98.9% | Implied Emigration:     2347 | Status: Anchor
    Country: Italy           | Retention Rate:  98.1% | Implied Emigration:      570 | Status: Anchor
    Country: Spain           | Retention Rate:  97.4% | Implied Emigration:     2648 | Status: Anchor
    ==============================================
    ```
2.  **Interactive panel**: A `Matplotlib` window with visualizations will automatically open, where you can switch between countries.

## 📄 License

This project is distributed under the MIT License.