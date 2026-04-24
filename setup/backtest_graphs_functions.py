import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def pro_backtest(df, initial_capital=10000, leverage=1, min_confidence=0.8, max_bet_pct=0.5, bp_cost=1.5, min_cash=1000,
                 asset=None, print_results=True, strategy='long-short', risk_free_ann=0.035, intraday_window_min=5):

    df = df.dropna(subset=['next_stock_move', 'abs_sentiment', 'true_sentiment'])
    df.reset_index(inplace=True); df.set_index('TIMESTAMP', inplace=True)
    if asset: df = df[df['KEY'].isin(asset)]

    signals = df[(df['abs_sentiment'] >= min_confidence) & (df['abs_sentiment'] < .95)].copy()
    if strategy == 'long-only': signals = signals[signals['true_sentiment']>0]
    if strategy == 'short-only': signals = signals[signals['true_sentiment']<0]
    if len(signals) == 0: return None
    signals = signals.sort_index()

    periods_per_year = (252 * 6.5 * 60) / intraday_window_min
    rf_per_period = (1 + risk_free_ann)**(1 / periods_per_year) - 1
    all_ts = pd.date_range(start=df.index.min(), end=df.index.max(), freq=f'{intraday_window_min}min')

    commission = bp_cost / 10000
    current_cash = initial_capital
    peak = initial_capital
    mdd_tracker = 0
    total_commission = 0
    total_trades = 0
    active_positions = {}
    equity_history = []
    asset_pnl_tracker = {} #this is to make sure you can use cahs from gains only once you receive them at t+2
    all_trade_pnl = []
    hits = []

    signal_dict = {ts: group for ts, group in signals.groupby(level=0)}
    pending_pnl = {}

    for ts in all_ts:
        if ts in pending_pnl: current_cash += pending_pnl.pop(ts)
        if current_cash <= min_cash: break
        current_cash *= (1 + rf_per_period) #gain insterest on idle cash

        if ts in signal_dict:
            group = signal_dict[ts]
            if group['time_to_exp'].iloc[0] <= 0.1: #always exit if market closes
                for asset_key in list(active_positions.keys()):
                    pos = active_positions.pop(asset_key)
                    current_cash -= (pos['size'] * commission)
                    asset_pnl_tracker[asset_key] -= (pos['size'] * commission)
                    total_commission += (pos['size'] * commission)
                    total_trades +=1
                equity_history.append({'timestamp': ts, 'equity': current_cash, 'pnl': 0})
                continue

            current_keys = group['KEY'].tolist()
            for asset_key in list(active_positions.keys()):
                if asset_key not in current_keys: #exit if asset already held is no longer significant
                    pos = active_positions.pop(asset_key)
                    current_cash -= (pos['size'] * commission)
                    total_commission += (pos['size'] * commission)
                    asset_pnl_tracker[asset_key] -= (pos['size'] * commission)
                    total_trades +=1 #all positions closed at t+x, x>1

            num_trades = len(group)
            individual_bet = (current_cash * max_bet_pct * leverage) / num_trades

            ts_pnl = 0
            for _, trade in group.iterrows():
                asset_key = trade['KEY']
                pos_size = individual_bet * trade['abs_sentiment']
                hits.append(np.sign(trade['true_sentiment']) == np.sign(trade['next_stock_move']))

                fee = 0
                if asset_key not in active_positions:
                    fee = pos_size * (commission + 0.1*(leverage-1)/10000) #leverage is added to trade fee (0.01 bp)
                    active_positions[asset_key] = {'size': pos_size, 'sign': np.sign(trade['true_sentiment'])}
                elif active_positions[asset_key]['sign'] != np.sign(trade['true_sentiment']):
                    fee = pos_size * (2 * commission + 0.1*(leverage-1)/10000)
                    active_positions[asset_key] = {'size': pos_size, 'sign': np.sign(trade['true_sentiment'])}
                    asset_pnl_tracker[asset_key] -= fee
                    total_trades +=1 #all positions closed specifically at t+1

                pnl = (pos_size * np.sign(trade['true_sentiment']) * trade['next_stock_move']) - fee
                all_trade_pnl.append(pnl)
                asset_pnl_tracker[asset_key] = asset_pnl_tracker.get(asset_key, 0) + pnl
                current_cash -= fee #only the fee you need to pay instantly
                ts_pnl += pnl
                total_commission += fee
                arrival_ts = ts + pd.Timedelta(minutes=intraday_window_min * 2) #you get reward 2 time windows later because next_stock_move is between t+1 and t+2
                pending_pnl[arrival_ts] = pending_pnl.get(arrival_ts, 0) + pnl +fee #+fee because you already paid it at time t

        equity_history.append({'timestamp': ts, 'equity': current_cash, 'pnl': ts_pnl if ts in signals.index else 0})
        if current_cash > peak: peak = current_cash
        mdd_tracker = max(mdd_tracker, (peak - current_cash) / peak)

    for asset_key in list(active_positions.keys()):
        pos = active_positions.pop(asset_key)
        exit_fee = pos['size'] * commission
        current_cash -= exit_fee
        total_commission += exit_fee
        total_trades +=1 #exit last open position in tradebook

    avg_win = np.mean([p for p in all_trade_pnl if p > 0])
    avg_loss = np.mean([p for p in all_trade_pnl if p < 0])
    exp_value = avg_win-abs(avg_loss)
    win_rate = (np.array(all_trade_pnl) > 0).mean() if len(all_trade_pnl) > 0 else 0
    hit_rate = np.mean(hits) if len(hits) > 0 else 0
    res = pd.DataFrame(equity_history).set_index('timestamp')
    res['ret'] = res['equity'].pct_change().fillna(0)

    excess_returns = res['ret'] - rf_per_period
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year) if excess_returns.std() != 0 else 0
    total_ret = (current_cash - initial_capital) / initial_capital

    if print_results:
        print(f"=== pro backtest [{intraday_window_min}m] ===")
        print(f"Total return: {total_ret:.2%}")
        print(f"Sharpe ratio: {sharpe:.2f}")
        print(f"Max drawdown: {mdd_tracker:.2%}")
        print(f"Win rate:     {win_rate:.2%}")
        print(f"Exp return:   ${exp_value:.2f}")
        print(f"Hit rate:     {hit_rate:.2%}")
        print(f'Net gain:     ${(current_cash-initial_capital):.2f}')
        print(f"Total fees:   ${total_commission:.2f}")
        print(f'Total trades: {total_trades}')
        print("\n--- ASSET BREAKDOWN (Net PnL) ---")
        asset_series = pd.Series(asset_pnl_tracker).sort_values(ascending=False)
        breakdown_df = pd.DataFrame({'Net PnL ($)': asset_series, 'Contribution (%)': (asset_series / initial_capital) * 100})
        print(breakdown_df)
        profitable_assets = (asset_series > 0).sum()
        total_assets = len(asset_series)
        print(f"\nAsset breadth: {profitable_assets}/{total_assets} assets were profitable!")

    return res, {"sharpe": sharpe, "total_return": total_ret, "mdd": mdd_tracker, "win_rate": win_rate, "hit_rate": hit_rate, "exp_value": exp_value}

def rolling_backtest_plot(df, min_confidence=0.8, bp_cost=5, strategy='long-short', window_days=30, include_exp_value=True):
    
    df = df.copy()
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    start, end = df['TIMESTAMP'].min(), df['TIMESTAMP'].max()
    dates, sharpes, win_rates, hit_rates, exp_vals = [], [], [], [], []
    while start + pd.Timedelta(days=window_days) <= end:
        w_end = start + pd.Timedelta(days=window_days)
        w = df[(df['TIMESTAMP'] >= start) & (df['TIMESTAMP'] < w_end)]
        if not w.empty:
            _, s = pro_backtest(w, min_confidence=min_confidence, bp_cost=bp_cost,strategy=strategy, leverage=1, print_results=False)
            dates.append(w_end)
            sharpes.append(s['sharpe'])
            win_rates.append(s['win_rate'])
            hit_rates.append(s['hit_rate'])
            exp_vals.append(s['exp_value'])
        start += pd.Timedelta(days=1)

    fig, ax1 = plt.subplots(figsize=(15, 8))
    l1 = ax1.plot(dates, sharpes, label='Sharpe', lw=2)
    lines = l1
    if include_exp_value:
        l2 = ax1.plot(dates, exp_vals, label='Exp Return', alpha=0.7)
        lines += l2

    ax1.set(xlabel='Date', ylabel='Sharpe/Expected return ($)')
    ax2 = ax1.twinx()
    l3 = ax2.plot(dates, win_rates, '--', label='Win Rate', color='seagreen')
    l4 = ax2.plot(dates, hit_rates, ':', label='Hit Rate', color='orange')
    lines += l3 + l4
    ax2.set_ylabel('Rate')
    if sharpes:
        primary = sharpes + (exp_vals if include_exp_value else [])
        m = max(abs(min(primary)), abs(max(primary))) * 1.2
        ax1.set_ylim(-m, m)

    if win_rates and hit_rates:
        d = max(max(abs(pd.Series(win_rates)-0.5)), max(abs(pd.Series(hit_rates)-0.5))) * 1.2
        ax2.set_ylim(0.5-d, 0.5+d)

    ax1.axhline(0, color='red', alpha=0.6)
    ax1.legend(lines, [l.get_label() for l in lines], loc='upper left', ncol=2, frameon=True, shadow=True)
    ax1.grid(ls=':', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def rolling_backtest_grid(df, min_conf_list, bp_cost_list, strategy='long-short', window_days=30, include_exp_value=True):

    df = df.copy()
    start0, end = df['TIMESTAMP'].min(), df['TIMESTAMP'].max()
    min_conf_list = sorted(min_conf_list)
    bp_cost_list = sorted(bp_cost_list)
    fig, axes = plt.subplots(len(bp_cost_list), len(min_conf_list),
                             figsize=(5*len(min_conf_list), 4*len(bp_cost_list)),
                             sharex=True)

    for i, bp in enumerate(bp_cost_list):
        for j, mc in enumerate(min_conf_list):
            start = start0
            dates, sharpes, win_rates, hit_rates, exp_vals = [], [], [], [], []
            while start + pd.Timedelta(days=window_days) <= end:
                w_end = start + pd.Timedelta(days=window_days)
                w = df[(df['TIMESTAMP'] >= start) & (df['TIMESTAMP'] < w_end)]
                if not w.empty:
                    _, s = pro_backtest(w, min_confidence=mc, bp_cost=bp, strategy=strategy, leverage=1, print_results=False)
                    dates.append(w_end)
                    sharpes.append(s['sharpe'])
                    win_rates.append(s['win_rate'])
                    hit_rates.append(s['hit_rate'])
                    exp_vals.append(s['exp_value'])
                start += pd.Timedelta(days=1)

            ax1 = axes[i, j] if len(bp_cost_list) > 1 else axes[j]
            l1 = ax1.plot(dates, sharpes, lw=1.5, label='Sharpe')
            lines = l1
            if include_exp_value:
                l2 = ax1.plot(dates, exp_vals, alpha=0.7, label='Exp')
                lines += l2

            ax2 = ax1.twinx()
            l3 = ax2.plot(dates, win_rates, '--', color='seagreen', label='Win')
            l4 = ax2.plot(dates, hit_rates, ':', color='orange', label='Hit')
            lines += l3 + l4
            if sharpes:
                primary = sharpes + (exp_vals if include_exp_value else [])
                m = max(abs(min(primary)), abs(max(primary))) * 1.2
                ax1.set_ylim(-m, m)

            if win_rates and hit_rates:
                d = max(max(abs(pd.Series(win_rates)-0.5)),
                        max(abs(pd.Series(hit_rates)-0.5))) * 1.2
                ax2.set_ylim(0.5-d, 0.5+d)

            ax1.axhline(0, color='red', alpha=0.5, lw=1)
            ax1.set_title(f"X^poly>{mc}, bp={bp}", fontsize=9)
            ax1.grid(True, linestyle='--', alpha=0.6)
            if i == len(bp_cost_list) - 1:
                ax1.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

def rolling_backtest_grid_by_strategy(df, strategies=['long-short', 'long-only', 'short-only'], min_confidence=0.8, bp_cost=5, window_days=30, include_exp_value=False):

    df = df.copy()
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    start0, end = df['TIMESTAMP'].min(), df['TIMESTAMP'].max()
    fig, axes = plt.subplots(1, len(strategies),
                             figsize=(6 * len(strategies), 5),
                             sharex=True)

    for j, strategy in enumerate(strategies):
        start = start0
        dates, sharpes, win_rates, hit_rates, exp_vals = [], [], [], [], []
        while start + pd.Timedelta(days=window_days) <= end:
            w_end = start + pd.Timedelta(days=window_days)
            w = df[(df['TIMESTAMP'] >= start) & (df['TIMESTAMP'] < w_end)]
            if not w.empty:
                _, s = pro_backtest(w, min_confidence=min_confidence, bp_cost=bp_cost,strategy=strategy, leverage=1, print_results=False)
                dates.append(w_end)
                sharpes.append(s['sharpe'])
                win_rates.append(s['win_rate'])
                hit_rates.append(s['hit_rate'])
                exp_vals.append(s['exp_value'])
            start += pd.Timedelta(days=1)

        ax1 = axes[j]
        l1 = ax1.plot(dates, sharpes, lw=1.5, label='Sharpe', color='royalblue')
        lines = l1
        if include_exp_value:
            l2 = ax1.plot(dates, exp_vals, alpha=0.7, label='Exp', color='gray')
            lines += l2

        ax2 = ax1.twinx()
        l3 = ax2.plot(dates, win_rates, '--', color='seagreen', label='Win Rate')
        l4 = ax2.plot(dates, hit_rates, ':', color='orange', label='Hit Rate')
        lines += l3 + l4

        if sharpes: #this is to set graph size
            primary = sharpes + (exp_vals if include_exp_value else [])
            m = max(abs(min(primary)), abs(max(primary))) * 1.2
            ax1.set_ylim(-m, m)

        if win_rates and hit_rates: #this is to centre
            d = max(max(abs(pd.Series(win_rates)-0.5)),
                    max(abs(pd.Series(hit_rates)-0.5))) * 1.2
            ax2.set_ylim(0.5-d, 0.5+d)
        ax1.axhline(0, color='red', alpha=0.5, lw=1)

        ax1.set_title(f"Strategy: {strategy.upper()}", fontsize=11)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.tick_params(axis='x', rotation=45)
        if j == 0:
            labs = [l.get_label() for l in lines]
            ax1.legend(lines, labs, loc='upper left', fontsize='small', frameon=True, shadow=True)

    plt.tight_layout()
    plt.show()

def plot_hit_rate_heatmap(df):
    df['hour_utc'] = pd.to_datetime(df['TIMESTAMP']).dt.hour
    df = df[(df['hour_utc']>13)&(df['hour_utc'] < 21)].copy()
    df['hour_ny'] = df['hour_utc']-4 #dont need to do -5 too, because it's already normalised from Ordebook
    df = df[df['abs_sentiment']>0.5].copy()
    df['is_hit'] = (np.sign(df['true_sentiment']) == np.sign(df['next_stock_move'])).astype(int)
    heatmap_data = df.groupby(['KEY', 'hour_ny'])['is_hit'].mean().unstack()

    plt.figure(figsize=(16, 8))
    sns.heatmap(
        heatmap_data,
        cmap='YlGnBu',
        annot=True,
        fmt=".0%",
        cbar_kws={'label': 'Hit rate (%)'}
    )
    plt.xlabel('Hour of day (NY)')
    plt.ylabel('Stock')
    plt.show()