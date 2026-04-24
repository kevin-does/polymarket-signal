from scipy.stats import norm
import numpy as np, pandas as pd

def collapse_to_windows(df, minutes=5, risk_free_ann=0.035):
    window = f'{str(minutes)}Min'
    df['weighted_price_up'] = df['PRICE_UP'] * df['USDC']
    df['vol_bull'] = np.where(
        ((df['BUY_SELL'] == 'BUY') & (df['UP_DOWN'] == 'UP')) |
        ((df['BUY_SELL'] == 'SELL') & (df['UP_DOWN'] == 'DOWN')),
        df['USDC'], 0)

    df['vol_bear'] = np.where(
        ((df['BUY_SELL'] == 'SELL') & (df['UP_DOWN'] == 'UP')) |
        ((df['BUY_SELL'] == 'BUY') & (df['UP_DOWN'] == 'DOWN')),
        df['USDC'], 0)

    df = df.set_index('TIMESTAMP')
    suffix = f"_{minutes}m"
    agg_dict = {
        'TIME_TO_EXP': 'last',
        'PRICE_UP': ['first', 'last', 'max', 'min'],
        'weighted_price_up': 'sum',
        'USDC': 'sum',
        'SHARES': 'count',
        'vol_bull': 'sum',
        'vol_bear': 'sum',
        'stock_open_day': 'last',
        f'stock_close{suffix}': 'last',
        f'stock_avg{suffix}': 'mean',
        f'stock_vol{suffix}': 'last'}

    collapsed = df.groupby('KEY').resample(window).agg(agg_dict)
    collapsed.columns = ['time_to_exp',
        'open_bet', 'close_bet', 'high_bet', 'low_bet', 'sum_weighted_price',
        'total_volume', 'trade_count', 'bull_volume', 'bear_volume',
        'stock_open_day', 'stock_close', 'stock_avg_period', 'stock_vol']
    collapsed['avg_price_up'] = collapsed['sum_weighted_price'] / collapsed['total_volume']
    collapsed.drop(columns=['sum_weighted_price'], inplace=True)
    collapsed['poly_vol_imbalance'] = (collapsed['bull_volume']-collapsed['bear_volume'])/(collapsed['bull_volume']+collapsed['bear_volume'])
    collapsed.drop(columns=['bull_volume', 'bear_volume'], inplace=True)
    # compute theoretical value with BS cash-or-nothing neutral probability for call opt (N(d2), so it's a prob)
    S = collapsed['stock_close'].shift(1) #the current price is the price of the stock when the window ends
    K = collapsed['stock_open_day']
    sigma = collapsed['stock_vol'].shift(1) #lagged because the vol is based on close returns, so stock closes: you dont know the stock close of your current time window
    T = ((collapsed['time_to_exp'] / 24) / 252).clip(lower=1e-9) # time to exp in years

    # d2 formula for neutral probability
    d2 = (np.log(S / K) + (risk_free_ann - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T).replace(0, np.nan))
    collapsed['bs_neutral_prob'] = norm.cdf(d2.fillna(0))
    collapsed['true_sentiment'] = collapsed['avg_price_up'] - collapsed['bs_neutral_prob'] #difference between polymarket prices and fair price: the signal!
    return collapsed.dropna().reset_index()

def check_lead_lag(df): #add lagged daily volume on polymarket to account to liquidity of the bet in general
    df = df.sort_values(['KEY', 'TIMESTAMP'])
    df['next_stock_move'] = np.log(df.groupby('KEY')['stock_close'].shift(-2) / df.groupby('KEY')['stock_close'].shift(-1)) #percentage change from t+1 to t+2; sentiment comes from t
    df['curr_stock_move'] = np.log(df['stock_close'] / df.groupby('KEY')['stock_close'].shift(1))
    df['next_true_sent'] = df.groupby('KEY')['true_sentiment'].shift(-2) - df.groupby('KEY')['true_sentiment'].shift(-1) #here simple diff because the value can be negative
    df['abs_sentiment'] =df['true_sentiment'].abs() #abs sentiment is a measure of 'conviction' that price is going to go up/down, if ~ 0, then not very convincing
    df['avg_trade'] = df['total_volume'] / df['trade_count']
    valid = df[df['time_to_exp'] > 0.5].dropna()
    overall_lead = valid['true_sentiment'].corr(valid['next_stock_move'])
    asset_lead = valid.groupby('KEY').apply(lambda x: x['true_sentiment'].corr(x['next_stock_move']), include_groups=False)
    overall_lag = valid['curr_stock_move'].corr(valid['next_true_sent'])
    asset_lag = valid.groupby('KEY').apply(lambda x: x['curr_stock_move'].corr(x['next_true_sent']), include_groups=False)

    print(f"Overall lead correlation: {overall_lead:.4f}")
    print("Lead correlation by asset:")
    print(asset_lead, '\n')
    print("-"*30, '\n')
    print(f"Overall lag correlation: {overall_lag:.4f}")
    print("Lag correlation by asset:")
    print(asset_lag)
    return df

def lead_lag_ccf(df, max_lag=5): #this is cross-correlation for the cross-section starting from signal at t and return at t+2 (=lag 0)
    df = df.sort_values(['KEY', 'TIMESTAMP'])
    valid = df[df['time_to_exp'] > 0.5].dropna()
    lags = range(0, max_lag + 1)

    lead_ccf = {}
    for k in lags:
        shifted_move = valid.groupby('KEY')['next_stock_move'].shift(-k)
        lead_ccf[k] = valid['true_sentiment'].corr(shifted_move)
    lead_ccf = pd.Series(lead_ccf)

    lag_ccf = {}
    for k in lags:
        shifted_sent = valid.groupby('KEY')['next_true_sent'].shift(-k)
        lag_ccf[k] = valid['curr_stock_move'].corr(shifted_sent)
    lag_ccf = pd.Series(lag_ccf)

    print("X^poly → future price (lead):\n")
    print(lead_ccf, "\n")
    print("-" * 40, "\n")
    print("price → Future X^poly (lag):\n")
    print(lag_ccf)

def sum_stats(df):
    cols = ['KEY','true_sentiment', 'abs_sentiment', 'avg_trade', 'poly_vol_imbalance', 'next_stock_move', 'stock_vol', 'time_to_exp', 'total_volume', 'trade_count']
    #summary = df[cols].groupby('KEY').describe().T
    summary = df[cols].describe().T
    return round(summary,4)

import statsmodels.api as sm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, average_precision_score

def run_logit(train_df, test_df, X_var, rf):
    results = []
    train_df = train_df.assign(direction=(train_df['next_stock_move'] > rf).astype(int)) #define up/down
    test_df = test_df.assign(direction=(test_df['next_stock_move'] > rf).astype(int))
    scaler = StandardScaler()

    for asset, train_data in train_df.groupby('KEY'):
        if asset not in test_df['KEY'].values: continue
        test_data = test_df[test_df['KEY'] == asset]

        X_train_raw = train_data[X_var]
        X_test_raw = test_data[X_var]
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X_var, index=train_data.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test_raw), columns=X_var, index=test_data.index)
        X_train = sm.add_constant(X_train_scaled)
        try:
            model = sm.Logit(train_data['direction'], X_train).fit(disp=0)
            X_test = sm.add_constant(X_test_scaled)
            y_prob = model.predict(X_test)
            y_pred = (y_prob > 0.5).astype(int)

            acc = (y_pred == test_data['direction']).mean()
            auc = roc_auc_score(test_data['direction'], y_prob)
            prec = precision_score(test_data['direction'], y_pred, zero_division=0)
            rec = recall_score(test_data['direction'], y_pred, zero_division=0)
            auprc = average_precision_score(test_data['direction'], y_prob)
            baseline = test_data['direction'].mean()
            results.append({
                'Asset': asset,
                **model.params.to_dict(),
                **model.tvalues.add_suffix('_z-stat').to_dict(),
                'Accuracy%': acc * 100,
                'AUC': auc,
                'AUPRC': auprc,
                'AUPRC_Lift': auprc - baseline,
                'Precision': prec,
                'Recall': rec,
                'Train N': len(train_data),
                'Test N': len(X_test)})
        except: continue

    return pd.DataFrame(results).set_index('Asset').drop(columns='const_z-stat', errors='ignore')

from sklearn.linear_model import ElasticNetCV

def run_elastic_net(train_df, test_df, X_var, rf):
    results = []
    l1_space = np.arange(0.0000001,1,.2) #this is for ridge (0) or lasso (1)
    alphas = np.logspace(-5, 0, 20)

    for asset, train_data in train_df.groupby('KEY'):
        if asset not in test_df['KEY'].values: continue
        test_data = test_df[test_df['KEY'] == asset]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_data[X_var])
        X_test = scaler.transform(test_data[X_var])

        y_train = (train_data['next_stock_move'] - rf) * 100
        y_test = (test_data['next_stock_move'] - rf)* 100
        model = ElasticNetCV(l1_ratio=l1_space, alphas=alphas, cv=TimeSeriesSplit(n_splits=3), max_iter=1000).fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2_oos = (1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2))) * 100

        results.append({
            'Asset': asset,
            **dict(zip(X_var, model.coef_)),
            'Penalty': round(model.alpha_,4),
            'L1_ratio': round(model.l1_ratio_,4),
            'OOS_R2%': r2_oos,
            'Train N': len(train_data),
            'Test N': len(X_test)})

    return pd.DataFrame(results).set_index('Asset')

def analyse_sentiment_dynamics(df, risk_free_ann=0.04, train_size=0.75, intraday_window=5):
    df = df.sort_values('TIMESTAMP')
    rf_minute = (1+risk_free_ann)**(1/(365*24*60))-1; rf_window=(1+rf_minute)**(intraday_window)-1
    cols = ['time_to_exp', 'true_sentiment', 'next_stock_move', 'abs_sentiment', 'curr_stock_move', 'avg_trade', 'poly_vol_imbalance']
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)

    split_idx = int(len(df) * train_size) #define train/test; careful, it's time series data
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    upper_limit = train_df['abs_sentiment'].quantile(0.95) #avoid noisy data based on training set
    train_df = train_df[((train_df['abs_sentiment']) < upper_limit)].copy()
    test_df = test_df[((test_df['abs_sentiment']) < upper_limit)].copy()

    for label, d_slice in [("IN-SAMPLE (TRAIN)", train_df), ("OUT-OF-SAMPLE (TEST)", test_df)]: #successful directioning
        print(f"\n--- {label} ---")
        d_slice['is_hit'] = (np.sign(d_slice['true_sentiment']) == np.sign(d_slice['next_stock_move'])).astype(int)
        print(f"{'percentile':<15} | {'min abs conviction':<15} | {'hit rate (%)':<12} | {'N'}")
        print("-" * 60)
        for p in [0.0, 0.5, 0.75, 0.9, 0.95]:
            thresh = d_slice['abs_sentiment'].quantile(p)
            sub = d_slice[d_slice['abs_sentiment'] >= thresh]
            name = f"top {int((1-p)*100)}%" if p > 0 else "all signals"
            print(f"{name:<15} | {thresh:>20.4f} | {sub['is_hit'].mean()*100:>11.2f}% | {len(sub)}")

    X_var = ['true_sentiment', 'poly_vol_imbalance', 'avg_trade',]# 'stock_vol_lag'] #the ols
    print("\n"+"="*30 + "\nresults from polymarket sentiment on price change OLS (OUT-OF-SAMPLE):\n")
    ols_results = []
    scaler = StandardScaler()

    for asset, train_data in train_df.groupby('KEY'):
        if asset not in test_df['KEY'].values: continue
        test_data = test_df[test_df['KEY'] == asset]
        X_train_raw = train_data[X_var]
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X_var, index=train_data.index)
        X_train = sm.add_constant(X_train_scaled)
        y_train = (train_data['next_stock_move'] - rf_window) * 100
        model_ols = sm.OLS(y_train, X_train).fit(cov_type='HAC', cov_kwds={'maxlags': int(len(X_train)**(1/4))})

        X_test_raw = test_data[X_var]
        X_test_scaled = pd.DataFrame(scaler.transform(X_test_raw), columns=X_var, index=test_data.index)
        X_test = sm.add_constant(X_test_scaled)
        y_test = (test_data['next_stock_move'] - rf_window) * 100
        y_pred = model_ols.predict(X_test)
        mae_oos = np.mean(np.abs(y_test - y_pred))
        ss_res, ss_tot = np.sum((y_test - y_pred)**2), np.sum((y_test - np.mean(y_test))**2)
        r2_oos = (1 - (ss_res / ss_tot)) * 100 if ss_tot != 0 else 0

        ols_results.append({'Asset': asset, **model_ols.params.to_dict(), **model_ols.tvalues.add_suffix('_t-stat').to_dict(), 'OOS_R2%': r2_oos, 'OOS_MAE': mae_oos,
                            'Train N': len(train_data), 'Test N': len(X_test)})

    print(pd.DataFrame(ols_results).set_index('Asset').drop(columns='const_t-stat'))
    print('\nOLS coefficients are standardised') #are pp increases in stock price following a 1 unit increase in sentiment')

    print("\n" + "="*30 + "\nresults from OLS with ElasticNet (OUT-OF-SAMPLE):\n") #the ML (elastic net)
    ols_regularised = run_elastic_net(train_df, test_df, X_var, rf_window)
    print(ols_regularised)
    print('\nEN coefficients are standardised')

    print("\n" + "="*30 + "\nresults from LOGIT directional prediction (OUT-OF-SAMPLE):\n") #the logit
    logit_results = run_logit(train_df, test_df, X_var, rf_window)
    print(logit_results)