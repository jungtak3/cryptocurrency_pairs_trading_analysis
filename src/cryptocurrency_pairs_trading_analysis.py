import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from collections import OrderedDict
import os
from datetime import datetime
# Import will be done conditionally to avoid circular imports

# Define cryptos and their file paths
CRYPTOS_FILES = OrderedDict([
    ("BTC", "./data/BTCUSDT_20170817_20250430.csv"),
    ("ETH", "./data/ETHUSDT_20170817_20250430.csv"),
    ("LTC", "./data/LTCUSDT_20171213_20250430.csv"),
    ("NEO", "./data/NEOUSDT_20171120_20250430.csv"),
])

# Define panels (start_date, end_date)
PANELS = OrderedDict([
    ("Panel A", ("2018-01-01", "2018-06-30")),
    ("Panel B", ("2018-07-01", "2018-12-31")),
    ("Panel C", ("2019-01-01", "2019-06-30")),
    ("Panel D", ("2019-07-01", "2019-12-31")),
])

# Define pairs for analysis
PAIRS = [
    ("BTC", "ETH"),
    ("BTC", "LTC"),
    ("BTC", "NEO"),
    ("ETH", "LTC"),
    ("ETH", "NEO"),
    ("LTC", "NEO"),
]

def load_data(file_path, start_date_str, end_date_str):
    try:
        df = pd.read_csv(file_path)
        df['time'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df.set_index('time')
        df = df[['open', 'close']]
        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame()

def display_table_for_console(df, title=""): # Renamed for clarity
    print(f"\n--- {title} ---")
    print(df.to_string())

def perform_adf_test_on_series(series, regression_type='ct'):
    if series.empty or series.isnull().all() or len(series.dropna()) < 4 :
            return np.nan, np.nan
    try:
        series_cleaned = series.dropna()
        if np.isinf(series_cleaned).any():
            series_cleaned = series_cleaned.replace([np.inf, -np.inf], np.nan).dropna()
        if len(series_cleaned) < 4: 
            return np.nan, np.nan
        result = adfuller(series_cleaned, regression=regression_type)
        return result[0], result[1]
    except Exception as e:
        return np.nan, np.nan

def calculate_stochastic_differential_residuals(crypto_data_dict):
    """
    Calculate stochastic differential residuals according to the paper's methodology.
    Implements equations (4), (6), and (7) from the paper.
    """
    try:
        # Step 1: Calculate daily returns for each crypto
        crypto_returns = {}
        market_caps = {'BTC': 4, 'ETH': 3, 'LTC': 2, 'NEO': 1}  # Arbitrary weights based on typical market cap ranking
        
        for crypto_symbol, df in crypto_data_dict.items():
            if 'close' in df.columns and not df['close'].empty:
                prices = df['close'].dropna()
                if len(prices) > 1:
                    returns = prices.pct_change().dropna()
                    crypto_returns[crypto_symbol] = returns
        
        if len(crypto_returns) < 2:
            return {}
        
        # Step 2: Create market benchmark return (weighted average)
        # Align all return series first
        aligned_returns = pd.DataFrame(crypto_returns)
        aligned_returns = aligned_returns.dropna()
        
        if aligned_returns.empty:
            return {}
        
        # Calculate weighted market return using market cap weights
        total_weight = sum(market_caps[crypto] for crypto in aligned_returns.columns if crypto in market_caps)
        market_return = pd.Series(0.0, index=aligned_returns.index)
        
        for crypto in aligned_returns.columns:
            weight = market_caps.get(crypto, 1) / total_weight
            market_return += weight * aligned_returns[crypto]
        
        # Step 3: Calculate market betas using OLS regression (equation 7)
        crypto_betas = {}
        for crypto in aligned_returns.columns:
            if not aligned_returns[crypto].empty and not market_return.empty:
                try:
                    # Add constant for OLS regression
                    X = sm.add_constant(market_return)
                    y = aligned_returns[crypto]
                    
                    # Align X and y properly
                    aligned_data = pd.concat([y, market_return], axis=1, join='inner').dropna()
                    if len(aligned_data) > 3:  # Need enough data points
                        y_clean = aligned_data.iloc[:, 0]
                        X_clean = sm.add_constant(aligned_data.iloc[:, 1])
                        
                        model = sm.OLS(y_clean, X_clean).fit()
                        crypto_betas[crypto] = model.params.iloc[1]  # Beta coefficient
                    else:
                        crypto_betas[crypto] = 1.0  # Default beta
                except:
                    crypto_betas[crypto] = 1.0  # Default beta if regression fails
        
        # Step 4: Calculate stochastic differential residuals for each pair (equation 4)
        stochastic_residuals = {}
        
        for crypto1_sym, crypto2_sym in PAIRS:
            if crypto1_sym in aligned_returns.columns and crypto2_sym in aligned_returns.columns:
                try:
                    # Get returns and betas
                    gamma_x = aligned_returns[crypto1_sym]
                    gamma_y = aligned_returns[crypto2_sym]
                    beta_x = crypto_betas.get(crypto1_sym, 1.0)
                    beta_y = crypto_betas.get(crypto2_sym, 1.0)
                    delta = beta_x - beta_y
                    
                    # Calculate stochastic differential: Dt = γxt - γyt - δγmt
                    stochastic_diff = gamma_x - gamma_y - delta * market_return
                    stochastic_residuals[(crypto1_sym, crypto2_sym)] = stochastic_diff.dropna()
                    
                except Exception as e:
                    # If calculation fails, return empty series
                    stochastic_residuals[(crypto1_sym, crypto2_sym)] = pd.Series(dtype=float)
            else:
                stochastic_residuals[(crypto1_sym, crypto2_sym)] = pd.Series(dtype=float)
        
        return stochastic_residuals
        
    except Exception as e:
        # Return empty dict if anything fails
        return {}

def run_analysis(): # Renamed main to run_analysis for clarity when importing
    print("Starting cryptocurrency pairs trading analysis (data processing)...")
    all_panel_data = OrderedDict()
    panel_names_list = list(PANELS.keys())

    for panel_name, (start_date, end_date) in PANELS.items():
        # This print can be removed if the new script handles all output
        # print(f"\nProcessing data for {panel_name} ({start_date} to {end_date})")
        panel_crypto_data = OrderedDict()
        for crypto_symbol, file_path in CRYPTOS_FILES.items():
            data = load_data(file_path, start_date, end_date)
            if not data.empty:
                panel_crypto_data[crypto_symbol] = data
        all_panel_data[panel_name] = panel_crypto_data
    
    # Initialize results dictionaries
    table1_data = OrderedDict()
    table2_data = OrderedDict()
    table3_data = OrderedDict()
    table4_data = OrderedDict()
    all_cointegration_betas = OrderedDict()
    all_cointegration_residuals = OrderedDict()
    all_stochastic_diff_residuals = OrderedDict()  # Add storage for stochastic differential residuals
    table5_data = OrderedDict()
    table6_data = OrderedDict()

    # --- Table 1: Summary Statistics ---
    for panel_name, crypto_data_dict in all_panel_data.items():
        if not crypto_data_dict: continue
        panel_stats = []
        for crypto_symbol, df in crypto_data_dict.items():
            if 'close' not in df.columns or df['close'].empty:
                stats_series = pd.Series({"Crypto": crypto_symbol, "Mean": np.nan, "Median": np.nan, "Maximum": np.nan, "Minimum": np.nan, "SD": np.nan, "Skewness": np.nan, "Kurtosis": np.nan, "Jarque-Bera": f"{np.nan:.2f}"})
            else:
                close_prices = df['close']
                jb_stat, jb_p_value = stats.jarque_bera(close_prices.dropna())
                jb_sig = "*" if jb_p_value < 0.01 else ""
                stats_series = pd.Series({"Crypto": crypto_symbol, "Mean": close_prices.mean(), "Median": close_prices.median(), "Maximum": close_prices.max(), "Minimum": close_prices.min(), "SD": close_prices.std(), "Skewness": close_prices.skew(), "Kurtosis": close_prices.kurtosis(), "Jarque-Bera": f"{jb_stat:.2f}{jb_sig}"})
            panel_stats.append(stats_series)
        if panel_stats: table1_data[panel_name] = pd.DataFrame(panel_stats).set_index('Crypto')

    # --- Table 2: Correlation and Distance ---
    for panel_name, crypto_data_dict in all_panel_data.items():
        if not crypto_data_dict or len(crypto_data_dict) < 2: continue
        panel_pair_metrics = []
        panel_closes = {sym: df['close'].dropna() for sym, df in crypto_data_dict.items() if 'close' in df and not df['close'].dropna().empty}
        if len(panel_closes) < 2: continue
        for crypto1_sym, crypto2_sym in PAIRS:
            if crypto1_sym in panel_closes and crypto2_sym in panel_closes:
                prices1, prices2 = panel_closes[crypto1_sym], panel_closes[crypto2_sym]
                aligned_prices1, aligned_prices2 = prices1.align(prices2, join='inner')
                if aligned_prices1.empty or len(aligned_prices1) < 2: correlation, distance = np.nan, np.nan
                else:
                    correlation = aligned_prices1.corr(aligned_prices2)
                    norm_prices1, norm_prices2 = (aligned_prices1 - aligned_prices1.mean()) / aligned_prices1.std(), (aligned_prices2 - aligned_prices2.mean()) / aligned_prices2.std()
                    distance = np.sum((norm_prices1 - norm_prices2)**2)
                panel_pair_metrics.extend([{"Pair": f"{crypto1_sym} and {crypto2_sym}", "Measure": "Correlation", "Value": correlation}, {"Pair": f"{crypto1_sym} and {crypto2_sym}", "Measure": "Distance", "Value": distance}])
            else: panel_pair_metrics.extend([{"Pair": f"{crypto1_sym} and {crypto2_sym}", "Measure": "Correlation", "Value": np.nan}, {"Pair": f"{crypto1_sym} and {crypto2_sym}", "Measure": "Distance", "Value": np.nan}])
        if panel_pair_metrics:
            pivot_df = pd.DataFrame(panel_pair_metrics).pivot_table(index="Measure", columns="Pair", values="Value")
            ordered_cols = [f"{p[0]} and {p[1]}" for p in PAIRS if f"{p[0]} and {p[1]}" in pivot_df.columns]
            table2_data[panel_name] = pivot_df[ordered_cols] if ordered_cols else pivot_df
    
    # --- Table 3: ADF Tests on Prices ---
    for panel_name, crypto_data_dict in all_panel_data.items():
        if not crypto_data_dict: continue
        adf_stats_panel = []
        for crypto_symbol, df in crypto_data_dict.items():
            if 'close' not in df.columns or df['close'].empty:
                adf_stats_panel.extend([{"Crypto": crypto_symbol, "Test Type": "Level", "ADF Statistic": "N/A", "P-value": "[N/A]"}, {"Crypto": crypto_symbol, "Test Type": "First Difference", "ADF Statistic": "N/A", "P-value": "[N/A]"}])
                continue
            close_prices = df['close'].dropna()
            adf_stat_level, p_val_level = perform_adf_test_on_series(close_prices, 'ct')
            sig_level = ""
            if pd.notna(p_val_level): sig_level = "*" if p_val_level < 0.01 else ("**" if p_val_level < 0.05 else ("***" if p_val_level < 0.1 else ""))
            first_diff = close_prices.diff().dropna()
            adf_stat_diff, p_val_diff = perform_adf_test_on_series(first_diff, 'ct')
            sig_diff = ""
            if pd.notna(p_val_diff): sig_diff = "*" if p_val_diff < 0.01 else ("**" if p_val_diff < 0.05 else ("***" if p_val_diff < 0.1 else ""))
            adf_stats_panel.extend([{"Crypto": crypto_symbol, "Test Type": "Level", "ADF Statistic": f"{adf_stat_level:.4f}{sig_level}" if pd.notna(adf_stat_level) else "N/A", "P-value": f"[{p_val_level:.4f}]" if pd.notna(p_val_level) else "[N/A]"}, {"Crypto": crypto_symbol, "Test Type": "First Difference", "ADF Statistic": f"{adf_stat_diff:.4f}{sig_diff}" if pd.notna(adf_stat_diff) else "N/A", "P-value": f"[{p_val_diff:.4f}]" if pd.notna(p_val_diff) else "[N/A]"}])
        if adf_stats_panel:
            pivot_adf = pd.DataFrame(adf_stats_panel).set_index(['Test Type', 'Crypto'])[['ADF Statistic', 'P-value']].unstack('Crypto')
            ordered_crypto_names = list(CRYPTOS_FILES.keys())
            final_cols = [(m, c) for c in ordered_crypto_names for m in ['ADF Statistic', 'P-value'] if (m,c) in pivot_adf.columns]
            table3_data[panel_name] = pivot_adf.reindex(columns=final_cols) if final_cols else pivot_adf

    # --- Table 4: Engle-Granger Cointegration ---
    for panel_name, crypto_data_dict in all_panel_data.items():
        if not crypto_data_dict or len(crypto_data_dict) < 2: continue
        panel_coint_metrics = []
        panel_closes = {sym: df['close'].dropna() for sym, df in crypto_data_dict.items() if 'close' in df and not df['close'].dropna().empty}
        if len(panel_closes) < 2: continue
        for crypto1_sym, crypto2_sym in PAIRS:
            pair_sorted = tuple(sorted((crypto1_sym, crypto2_sym)))
            beta_key = (crypto1_sym, crypto2_sym) 
            if crypto1_sym in panel_closes and crypto2_sym in panel_closes:
                p1_orig, p2_orig = panel_closes[crypto1_sym], panel_closes[crypto2_sym]
                p1_aligned, p2_aligned = p1_orig.align(p2_orig, join='inner')
                if p1_aligned.empty or len(p1_aligned) < 20: const, coeff, r2, dw, sig_coeff, residuals = np.nan, np.nan, np.nan, np.nan, "", pd.Series(dtype=float)
                else:
                    dep_var, ind_var = (p1_aligned, p2_aligned) if p1_aligned.mean() < p2_aligned.mean() else (p2_aligned, p1_aligned)
                    beta_key = (dep_var.name if hasattr(dep_var,'name') else (crypto1_sym if dep_var is p1_aligned else crypto2_sym), ind_var.name if hasattr(ind_var,'name') else (crypto1_sym if ind_var is p1_aligned else crypto2_sym))
                    if dep_var is p1_aligned : beta_key = (crypto1_sym, crypto2_sym)
                    else: beta_key = (crypto2_sym, crypto1_sym)
                    ind_var_const = sm.add_constant(ind_var)
                    try:
                        res = sm.OLS(dep_var, ind_var_const).fit()
                        const, coeff, r2, dw, p_val_coeff = res.params.iloc[0], res.params.iloc[1], res.rsquared, sm.stats.stattools.durbin_watson(res.resid), res.pvalues.iloc[1]
                        sig_coeff = ""
                        if pd.notna(p_val_coeff): sig_coeff = "*" if p_val_coeff < 0.01 else ("**" if p_val_coeff < 0.05 else ("***" if p_val_coeff < 0.1 else ""))
                        residuals = res.resid
                    except Exception: const, coeff, r2, dw, sig_coeff, residuals = np.nan, np.nan, np.nan, np.nan, "", pd.Series(dtype=float)
                all_cointegration_betas[(panel_name, beta_key)] = coeff
                all_cointegration_residuals[(panel_name, pair_sorted)] = residuals
                panel_coint_metrics.extend([{"Pair": f"{crypto1_sym} and {crypto2_sym}", "Metric": m, "Value": v} for m,v in [("Constant", f"{const:.3f}" if pd.notna(const) else "N/A"), ("Coefficient", f"{coeff:.3f}{sig_coeff}" if pd.notna(coeff) else "N/A"), ("R2", f"{r2:.3f}" if pd.notna(r2) else "N/A"), ("DW", f"{dw:.3f}" if pd.notna(dw) else "N/A")]])
            else:
                all_cointegration_betas[(panel_name, beta_key)] = np.nan
                all_cointegration_residuals[(panel_name, pair_sorted)] = pd.Series(dtype=float)
                panel_coint_metrics.extend([{"Pair": f"{crypto1_sym} and {crypto2_sym}", "Metric": m, "Value": "N/A"} for m in ["Constant", "Coefficient", "R2", "DW"]])
        if panel_coint_metrics:
            pivot_coint = pd.DataFrame(panel_coint_metrics).pivot_table(index="Metric", columns="Pair", values="Value", aggfunc='first')
            metric_order = ["Constant", "Coefficient", "R2", "DW"]
            ordered_cols = [f"{p[0]} and {p[1]}" for p in PAIRS if f"{p[0]} and {p[1]}" in pivot_coint.columns]
            table4_data[panel_name] = pivot_coint.reindex(index=metric_order, columns=ordered_cols) if ordered_cols else pivot_coint.reindex(index=metric_order)

    # --- Calculate Stochastic Differential Residuals for Table 5 ---
    for panel_name, crypto_data_dict in all_panel_data.items():
        if not crypto_data_dict or len(crypto_data_dict) < 2:
            continue
        
        # Calculate stochastic differential residuals for this panel
        stochastic_residuals = calculate_stochastic_differential_residuals(crypto_data_dict)
        
        # Store residuals for Table 5
        for pair_key, residuals in stochastic_residuals.items():
            all_stochastic_diff_residuals[(panel_name, pair_key)] = residuals

    # --- Table 5: ADF Tests on Residuals ---
    for panel_name, crypto_data_dict in all_panel_data.items():
        if not crypto_data_dict or len(crypto_data_dict) < 2: continue
        panel_adf_res_metrics = []
        panel_closes = {sym: df['close'].dropna() for sym, df in crypto_data_dict.items() if 'close' in df and not df['close'].dropna().empty}
        if len(panel_closes) < 2: continue
        for crypto1_sym, crypto2_sym in PAIRS:
            pair_sorted = tuple(sorted((crypto1_sym, crypto2_sym)))
            pair_disp = f"{crypto1_sym} and {crypto2_sym}"
            coint_res = all_cointegration_residuals.get((panel_name, pair_sorted), pd.Series(dtype=float))
            adf_stat_c, p_val_c = perform_adf_test_on_series(coint_res, 'n')
            sig_c = ""
            if pd.notna(p_val_c): sig_c = "*" if p_val_c < 0.01 else ("**" if p_val_c < 0.05 else ("***" if p_val_c < 0.1 else ""))
            panel_adf_res_metrics.append({"Pair": pair_disp, "Res Type": "Cointegration", "ADF Stat": f"{adf_stat_c:.3f}{sig_c}" if pd.notna(adf_stat_c) else "N/A", "P-value": f"({p_val_c:.3f})" if pd.notna(p_val_c) else "(N/A)"})
            if crypto1_sym in panel_closes and crypto2_sym in panel_closes:
                p1, p2 = panel_closes[crypto1_sym], panel_closes[crypto2_sym]
                ap1, ap2 = p1.align(p2, join='inner')
                if not ap1.empty and len(ap1) >= 2:
                    norm_p1, norm_p2 = (ap1 - ap1.mean()) / ap1.std(), (ap2 - ap2.mean()) / ap2.std()
                    dist_res = norm_p1 - norm_p2
                    adf_stat_d, p_val_d = perform_adf_test_on_series(dist_res, 'n')
                    sig_d = ""
                    if pd.notna(p_val_d): sig_d = "*" if p_val_d < 0.01 else ("**" if p_val_d < 0.05 else ("***" if p_val_d < 0.1 else ""))
                    panel_adf_res_metrics.append({"Pair": pair_disp, "Res Type": "Distance", "ADF Stat": f"{adf_stat_d:.3f}{sig_d}" if pd.notna(adf_stat_d) else "N/A", "P-value": f"({p_val_d:.3f})" if pd.notna(p_val_d) else "(N/A)"})
                else: panel_adf_res_metrics.append({"Pair": pair_disp, "Res Type": "Distance", "ADF Stat": "N/A", "P-value": "(N/A)"})
            else: panel_adf_res_metrics.append({"Pair": pair_disp, "Res Type": "Distance", "ADF Stat": "N/A", "P-value": "(N/A)"})
            # Stochastic differential residuals
            stoch_res = all_stochastic_diff_residuals.get((panel_name, (crypto1_sym, crypto2_sym)), pd.Series(dtype=float))
            if stoch_res.empty:
                # Try reverse order
                stoch_res = all_stochastic_diff_residuals.get((panel_name, (crypto2_sym, crypto1_sym)), pd.Series(dtype=float))
            
            if not stoch_res.empty and len(stoch_res) > 3:
                adf_stat_s, p_val_s = perform_adf_test_on_series(stoch_res, 'n')
                sig_s = ""
                if pd.notna(p_val_s): sig_s = "*" if p_val_s < 0.01 else ("**" if p_val_s < 0.05 else ("***" if p_val_s < 0.1 else ""))
                panel_adf_res_metrics.append({"Pair": pair_disp, "Res Type": "Stochastic Diff.", "ADF Stat": f"{adf_stat_s:.3f}{sig_s}" if pd.notna(adf_stat_s) else "N/A", "P-value": f"({p_val_s:.3f})" if pd.notna(p_val_s) else "(N/A)"})
            else:
                panel_adf_res_metrics.append({"Pair": pair_disp, "Res Type": "Stochastic Diff.", "ADF Stat": "N/A", "P-value": "(N/A)"})
        if panel_adf_res_metrics:
            pivot_adf_res = pd.DataFrame(panel_adf_res_metrics).set_index(['Res Type', 'Pair'])[['ADF Stat', 'P-value']].unstack('Pair')
            final_cols_res = [(m, f"{p[0]} and {p[1]}") for p_sym1, p_sym2 in PAIRS for p in [((p_sym1,p_sym2))] for m in ['ADF Stat', 'P-value'] if (m, f"{p[0]} and {p[1]}") in pivot_adf_res.columns]
            table5_data[panel_name] = pivot_adf_res.reindex(columns=final_cols_res) if final_cols_res else pivot_adf_res
    
    # --- Table 6: Profit Payoffs ---
    table6_results_list = []
    formation_panel_map = {"Panel A": "Panel A", "Panel B": "Panel A", "Panel C": "Panel B", "Panel D": "Panel C"}
    for trading_panel_name in panel_names_list:
        formation_panel_name = formation_panel_map[trading_panel_name]
        if trading_panel_name not in all_panel_data or formation_panel_name not in all_panel_data: continue
        trading_day_data = all_panel_data[trading_panel_name]
        for crypto1_sym, crypto2_sym in PAIRS:
            pair_name_display = f"{crypto1_sym} and {crypto2_sym}"
            dep_sym_beta, ind_sym_beta, beta = None, None, np.nan
            beta1, beta2 = all_cointegration_betas.get((formation_panel_name, (crypto1_sym, crypto2_sym))), all_cointegration_betas.get((formation_panel_name, (crypto2_sym, crypto1_sym)))
            if pd.notna(beta1): dep_sym_beta, ind_sym_beta, beta = crypto1_sym, crypto2_sym, beta1
            elif pd.notna(beta2): dep_sym_beta, ind_sym_beta, beta = crypto2_sym, crypto1_sym, beta2
            daily_profits_ls = pd.Series(dtype=float)
            if dep_sym_beta and ind_sym_beta and pd.notna(beta) and dep_sym_beta in trading_day_data and ind_sym_beta in trading_day_data:
                open_dep, close_dep = trading_day_data[dep_sym_beta]['open'], trading_day_data[dep_sym_beta]['close']
                open_ind, close_ind = trading_day_data[ind_sym_beta]['open'], trading_day_data[ind_sym_beta]['close']
                df_aligned = pd.concat([open_dep.rename('open_dep'), close_dep.rename('close_dep'), open_ind.rename('open_ind'), close_ind.rename('close_ind')], axis=1).dropna()
                if not df_aligned.empty: daily_profits_ls = (df_aligned['close_dep'] - df_aligned['open_dep']) - beta * (df_aligned['close_ind'] - df_aligned['open_ind'])
            cum_profit_ls, risk_ls = (daily_profits_ls.sum() if not daily_profits_ls.empty else 0), (daily_profits_ls.std() if not daily_profits_ls.empty and len(daily_profits_ls) > 1 else 0)
            profit_per_risk_ls = (cum_profit_ls / risk_ls) if risk_ls != 0 else 0
            table6_results_list.extend([{"Panel": trading_panel_name, "Pair": pair_name_display, "Strategy": "Pair Trading", "Measure": m, "Value": v} for m,v in [("Cumulative Profit (USD)", cum_profit_ls), ("SD in USD (Risk)", risk_ls), ("Profit Per Unit of Risk(USD)", profit_per_risk_ls)]])
            daily_profits_lo = pd.Series(dtype=float)
            if crypto1_sym in trading_day_data and crypto2_sym in trading_day_data:
                profit_c1, profit_c2 = (trading_day_data[crypto1_sym]['close'] - trading_day_data[crypto1_sym]['open']).dropna(), (trading_day_data[crypto2_sym]['close'] - trading_day_data[crypto2_sym]['open']).dropna()
                if not profit_c1.empty and not profit_c2.empty: aligned_profit_c1, aligned_profit_c2 = profit_c1.align(profit_c2, join='outer', fill_value=0); daily_profits_lo = aligned_profit_c1 + aligned_profit_c2
                elif not profit_c1.empty: daily_profits_lo = profit_c1
                elif not profit_c2.empty: daily_profits_lo = profit_c2
            cum_profit_lo, risk_lo = (daily_profits_lo.sum() if not daily_profits_lo.empty else 0), (daily_profits_lo.std() if not daily_profits_lo.empty and len(daily_profits_lo) > 1 else 0)
            profit_per_risk_lo = (cum_profit_lo / risk_lo) if risk_lo != 0 else 0
            table6_results_list.extend([{"Panel": trading_panel_name, "Pair": pair_name_display, "Strategy": "Portfolio", "Measure": m, "Value": v} for m,v in [("Cumulative Profit (USD)", cum_profit_lo), ("SD in USD (Risk)", risk_lo), ("Profit Per Unit of Risk(USD)", profit_per_risk_lo)]])
    if table6_results_list:
        df_table6_full = pd.DataFrame(table6_results_list)
        for panel_name_key in panel_names_list: # Use panel_name_key to avoid conflict
            panel_df = df_table6_full[df_table6_full["Panel"] == panel_name_key]
            if not panel_df.empty:
                pivot_table6 = panel_df.pivot_table(index=["Strategy", "Measure"], columns="Pair", values="Value")
                metric_order_table6 = ["Cumulative Profit (USD)", "SD in USD (Risk)", "Profit Per Unit of Risk(USD)"]
                strategy_order = ["Pair Trading", "Portfolio"]
                pivot_table6 = pivot_table6.reindex(index=pd.MultiIndex.from_product([strategy_order, metric_order_table6], names=['Strategy', 'Measure']))
                ordered_pair_cols_t6 = [f"{p[0]} and {p[1]}" for p in PAIRS if f"{p[0]} and {p[1]}" in pivot_table6.columns]
                table6_data[panel_name_key] = pivot_table6[ordered_pair_cols_t6] if ordered_pair_cols_t6 else pivot_table6
    
    return {
        "table1": table1_data, "table2": table2_data, "table3": table3_data,
        "table4": table4_data, "table5": table5_data, "table6": table6_data
    }

if __name__ == "__main__":
    processed_tables = run_analysis() # Call the renamed analysis function

    if "table1" in processed_tables:
        for panel_name, stats_df in processed_tables["table1"].items():
            display_table_for_console(stats_df, title=f"TABLE 1: SUMMARY STATISTICS - {panel_name}")
    
    if "table2" in processed_tables:
        for panel_name, metrics_df in processed_tables["table2"].items():
            display_table_for_console(metrics_df, title=f"TABLE 2: CORRELATION AND DISTANCE - {panel_name}")

    if "table3" in processed_tables:
        for panel_name, adf_df in processed_tables["table3"].items():
            display_table_for_console(adf_df, title=f"TABLE 3: ADF TESTS ON PRICES - {panel_name}")
            print("Note: * significant at 1%, ** at 5%, *** at 10% (p-value). Regression: constant and trend ('ct').")

    if "table4" in processed_tables:
        for panel_name, coint_df in processed_tables["table4"].items():
            display_table_for_console(coint_df, title=f"TABLE 4: ENGLE-GRANGER COINTEGRATION - {panel_name}")
            print("Note: * sig at 1%, ** at 5%, *** at 10% (p-value for coeff). Dep var: lower avg price.")

    if "table5" in processed_tables:
        for panel_name, adf_res_df in processed_tables["table5"].items():
            display_table_for_console(adf_res_df, title=f"TABLE 5: ADF TESTS ON RESIDUALS - {panel_name}")
            print("Note: * sig at 1%, ** at 5%, *** at 10% (p-value). Regression for residuals: none ('n'). Stochastic Diff. method implemented per paper's methodology.")

    if "table6" in processed_tables and processed_tables["table6"]:
        for panel_name, pivot_table6_df in processed_tables["table6"].items():
             display_table_for_console(pivot_table6_df, title=f"TABLE 6: PROFIT PAYOFFS - {panel_name}")
    elif "table6" in processed_tables:
         print("\n\n--- TABLE 6: PROFIT PAYOFFS (No data to display) ---")
    
    print("\nAnalysis script console output complete. All tables processed.")
    
    # Generate markdown report
    print("\nGenerating markdown report...")
    
    try:
        # Import here to avoid circular imports
        from markdown_to_pdf_generator import MarkdownPDFGenerator
        
        # Create reports directory if it doesn't exist
        reports_dir = "../reports"
        os.makedirs(reports_dir, exist_ok=True)
        
        # Generate markdown report
        generator = MarkdownPDFGenerator(output_dir=reports_dir)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"crypto_pairs_trading_report_{timestamp}_main_analysis.md"
        report_path = os.path.join(reports_dir, report_filename)
        
        # Generate the markdown content
        markdown_content = generator.generate_markdown_report(processed_tables)
        
        # Write to file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"Markdown report generated: {report_path}")
        
    except ImportError as e:
        print(f"Warning: Could not import markdown generator: {e}")
        print("Markdown report generation skipped.")
    except Exception as e:
        print(f"Error generating markdown report: {e}")