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

## 📊 Methodology: DBNA & Forensic Analysis

Standard "inflow minus outflow" analysis methods do not work for EU migration statistics due to two problems:
1.  **Unreliable emigration data (`migr_emi`)**: Many countries, including France, hardly keep any records of foreigners who have left.
2.  **Distortion due to naturalization (`migr_acq`)**: When an immigrant obtains citizenship, they disappear from the "foreign population" statistics, which can be mistakenly interpreted as emigration.
3.  **Invisible Populations**: Standard models ignore asylum seekers whose applications are pending. They are physically present but not yet in the "Valid Residence Permit" stock.

To solve these problems, the **Demographic Balancing with Naturalization Adjustment (DBNA) Method** is used, now upgraded to a **Forensic Model**.

### Key Metrics

1.  **Standard Retention Rate (CR)**:
    The classic metric based on official residence permits.
    $$CR = \left( 1 - \frac{\max(0, E_{implied})}{P_t + I_{(t)}} \right) \times 100\%$$

2.  **Forensic Retention Rate (De Facto CR)**:
    Adjusts the population stock to include **Pending Asylum Seekers** (`migr_asypenctzm`). This reveals the "De Facto" retention, often higher than official statistics suggest because these individuals have not left, even if they don't have a permit yet.

3.  **Anchoring Ratio**:
    Measures the depth of integration.
    $$Anchoring = \frac{\text{Long-Term Residents}}{\text{Total Valid Stock}} \times 100\%$$
    A high ratio indicates a settled community, while a low ratio suggests a transient "transit" population.

Where:
-   $P_t$: Number of residents (`migr_resvalid`).
-   $I_{(t)}$: Inflow (`migr_resfirst`).
-   $E_{implied}$: "Implied emigration," calculated as the difference between the expected and actual population figures.

## 💾 Data Acquisition & Setup

This analysis relies exclusively on official data from the **Eurostat Database**, ensuring reliability and comparability across countries. The script now requires **7 specific datasets** to perform the forensic reconstruction.

1.  **Clone the repository and install dependencies:**
    ```bash
    git clone https://github.com/KiraKicka/Ru-EU-Migration-Retention-Analyzer
    cd migration-analyzer
    pip install pandas numpy matplotlib seaborn openpyxl
    ```

2.  **Download the data from Eurostat:**
    Navigate to the [Eurostat Database](https://ec.europa.eu/eurostat/web/main/data/database) and download the following tables. For each table, use the **"Data Explorer"** to apply the filters below, then download the data as a **"Spreadsheet"** (`.xlsx` file).

    *   **Filters to apply for ALL tables:**
        *   **Citizen (CITIZEN):** Russia (RU)
        *   **Sex (SEX):** Total
        *   **Age (AGE):** Total
        *   **Unit (UNIT):** Person / Number
        *   **Destination (GEO):** Select your countries of interest (e.g., Germany, France, Italy, Spain).

    *   **Required Tables:**

        | Eurostat Code | Purpose | Why it's needed | Filename to use |
        | :--- | :--- | :--- | :--- |
        | **`migr_resvalid`** | Stock of Residents | Provides the baseline number of Russian citizens with valid residence permits at the end of each year. This is our ground truth. | `migr_resvalid.xlsx` |
        | **`migr_resfirst`** | Inflow of Immigrants | Tracks the number of newly issued first residence permits each year. This is the primary input driving population growth. | `migr_resfirst.xlsx` |
        | **`migr_acq`** | Naturalization | Accounts for residents who acquire citizenship. They haven't emigrated but are removed from the `migr_resvalid` stock, so we must track them to avoid misinterpreting their exit as a "loss". | `migr_acq.xlsx` |
        | **`migr_asypenctzm`** | **Pending Asylum** | **(New)** Tracks asylum seekers waiting for a decision. They are physically present but invisible in standard stock data. | `migr_asypenctzm.xlsx` |
        | **`migr_reschange`** | Change of Status | **(New)** Used to analyze churn and integration pathways. | `migr_reschange.xlsx` |
        | **`migr_reslong`** | Long-Term Residents | **(New)** Used to calculate the Anchoring Ratio. | `migr_reslong.xlsx` |
        | **`migr_resbc13`** | EU Blue Cards | **(New)** Tracks high-skilled migration presence. | `migr_resbc13.xlsx` |

    > **Note:** The script automatically handles Monthly data (e.g., `2023M12`) often found in the Asylum dataset (`migr_asypenctzm`). It filters for December to align with annual stocks.

3.  **Place the files:**
    Move the downloaded `.xlsx` files into the root directory of the project, ensuring their names match the "Filename to use" column above.

## 📈 Understanding the Gap Analysis

The core of this project is the **Gap Decomposition** chart, which visualizes the difference between a theoretical "perfect retention" scenario and the observed reality.

Here is how to interpret each element of the chart:

*   **THE "GAP"**: This is the overall space between the top grey line (the maximum possible population) and the bottom blue line (the actual population). The chart explains what this gap is made of.

*   **Lines:**
    *   <span style="color:grey">▬</span> **Grey Line (`Theoretical Max`):** Represents the theoretical maximum number of residents if **no one ever left** and **no one acquired citizenship**. It's calculated as `Initial Population + Cumulative Inflow`.
    *   <span style="color:gold">▬</span> **Gold Dashed Line (`Theoretical Adj.`):** This is a more realistic theoretical line. It subtracts the cumulative number of naturalized citizens from the `Theoretical Max`. This line shows what the resident stock *should* be if the only "exit" was acquiring a passport.
    *   <span style="color:purple">····</span> **Purple Dotted Line (`Total De Facto`):** The **Forensic View**. This is `Official Stock` + `Pending Asylum`.
    *   <span style="color:#003366">▬</span> **Blue Line (`Official Stock`):** The number of valid residence permits.

*   **Shaded Areas:**
    *   <span style="color:gold;opacity:0.5">■</span> **Gold Area (`Naturalized / Integration`):** The gap between the grey and gold lines. This area represents the portion of the original cohort that has **successfully integrated** by becoming citizens of the host country. From a retention perspective, this is a positive outcome, not a loss.
    *   <span style="color:purple;opacity:0.3">■</span> **Purple Area (`Pending / Invisible`):** People physically present (Asylum seekers) but not in the official stock.
    *   <span style="color:red;opacity:0.5">■</span> **Red Area (`Implied Emigration`):** The "unexplained" loss calculated from the **De Facto** line (Forensic view).

*   **Bottom Chart (Flows):**
    *   <span style="color:teal">▬</span> **Teal Line (`Inflow`):** New arrivals (First Permits).
    *   <span style="color:gold">▬</span> **Gold Line (`Naturalization`):** Passport issuance. Comparing these two shows if a country is importing new people or integrating existing ones.

## 🚀 Usage

To run the analysis, execute the script from the command line:

```bash
python migration_analysis.py
```

Execution process:
1.  **Console report**: A ranking of countries by **Retention Rate** will be printed to the terminal.
    ```
    === DEMOGRAPHIC REPORT (Retention Ranking) ===
    Country         |   Std CR | Forensic CR | Anchoring
    -------------------------------------------------------
    France          |    99.8% |      101.2% |     45.2%
    Germany         |    98.9% |       99.5% |     62.1%
    Italy           |    98.1% |       98.3% |     55.0%
    Spain           |    97.4% |       97.9% |     38.4%
    ==============================================
    ```
2.  **Interactive panel**: A `Matplotlib` window with visualizations will automatically open, where you can switch between countries.

## 📄 License

This project is distributed under the MIT License.
