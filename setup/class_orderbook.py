import requests, json, time, pandas as pd, pytz, yfinance as yf, numpy as np
from datetime import datetime as dt, timedelta, timezone, date
from concurrent.futures import ThreadPoolExecutor

class Orderbook:
    def __init__(self, days_back=7, max_workers=10, start_day_from_now=0, intraday_minutes=15):
        self.days_back = days_back
        self.start_day_from_now = start_day_from_now
        self.intraday_minutes = intraday_minutes
        self.max_workers = max_workers
        self.gamma_api = "https://gamma-api.polymarket.com/events/slug"
        self.goldsky_url = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/orderbook-subgraph/0.0.1/gn"
        self.timezone_map = {
            'new york': 'America/New_York',
            'london': 'Europe/London',
            'tokyo': 'Asia/Tokyo',
            'europe': 'Europe/Berlin'}
        self.asset_keys = []
        self.df = pd.DataFrame()
        self.orderbook = pd.DataFrame()

    def get_data(self, keys):
        self._get_multiple_tokens(keys)
        self._process_all_orderfills()

        def get_outcome_date(row): #to get the right match
            base_date = row['TIMESTAMP'].astimezone(pytz.timezone(self.timezone_map.get(row['country'].lower(), 'UTC'))).date()
            return base_date + timedelta(days=1) if 0 <= row['REL_HOUR'] < 8 else base_date

        self.orderbook['market_date'] = self.orderbook.apply(get_outcome_date, axis=1)
        stock_df = self._pull_stock_close()
        self.orderbook = self.orderbook.merge(
            stock_df,
            left_on=['KEY', 'market_date'],
            right_on=['KEY', 'date'], how='left')
        self.orderbook['TIMESTAMP'] = pd.to_datetime(self.orderbook['TIMESTAMP'])
        self.orderbook = self.orderbook.sort_values('TIMESTAMP')
        time_window_stock_pull = self.intraday_minutes #add 30 min too for robustness
        stock_min = self._pull_stock_minutes(minutes=time_window_stock_pull, window_size=120//time_window_stock_pull) #the window size determines std for BS: use past 120mins
        stock_min['TIMESTAMP'] = pd.to_datetime(stock_min['TIMESTAMP'])
        stock_min = stock_min.sort_values('TIMESTAMP')
        self.orderbook = pd.merge_asof(
            self.orderbook,
            stock_min,
            on='TIMESTAMP',
            by='KEY',
            direction='backward',
            tolerance=pd.Timedelta(f'{str(time_window_stock_pull)}min'))
        self.orderbook = self.orderbook.drop(columns=['date', 'market_date']).reset_index(drop=True)
        self.orderbook = self.orderbook.sort_values(by=['KEY', 'TIMESTAMP'], ascending=[True, True])
        return self.orderbook

    def _get_market_tokens(self, key_tuple):
        key = key_tuple[0]; country = key_tuple[1].lower()
        local_timezone = pytz.timezone(self.timezone_map.get(country, 'UTC'))
        now_local = dt.now(local_timezone) - timedelta(days=self.start_day_from_now)
        data = []
        for i in range(self.days_back):
            target = now_local - timedelta(days=i)
            slug = f"{key}-up-or-down-on-{target.strftime('%B-%-d-%Y').lower()}"
            try:
                r = requests.get(f"{self.gamma_api}/{slug}")
                if r.status_code == 200:
                    res_json = r.json()
                    if res_json.get("markets"):
                        m = res_json["markets"][0]
                        condition_id = m.get('conditionId')
                        tks = json.loads(m.get('clobTokenIds', '[]'))
                        data.append([key.upper(), target.date(), tks[0], tks[1], condition_id, country])
            except Exception as e:
                print(f"error fetching {slug}: {e}")
        #print(condition_id, tks[0], tks[1], target) #to get token - date to manually test on graphiQL
        return pd.DataFrame(data, columns=['key', 'ts', 'up_token', 'down_token', 'condition_id', 'country'])

    def _get_multiple_tokens(self, keys):
        all_dfs = []
        for k in keys:
            asset_name = k[0] if isinstance(k, tuple) else k
            self.asset_keys.append(asset_name.upper())
            print(f"--- collecting & processing: {asset_name.upper()} ---")
            raw = self._get_market_tokens(k if isinstance(k, tuple) else (k, 'new york'))
            all_dfs.append(raw)

        if not all_dfs: return pd.DataFrame()
        self.df = pd.concat(all_dfs).sort_values(['ts', 'key']).reset_index(drop=True)
        return self.df

    def _get_single_orderfills(self, a, start_dt, end_dt, up_down_key):
        q = '''query($a: String!, $g: BigInt!, $l: BigInt!) {
            orderFilledEvents(
              where: {
                or: [
                  {
                    timestamp_gte: $g,
                    timestamp_lte: $l,
                    takerAssetId: $a,
                    makerAssetId: "0"
                  },
                  {
                    timestamp_gte: $g,
                    timestamp_lte: $l,
                    makerAssetId: $a,
                    takerAssetId: "0"
                  }
                ]
              }
              orderBy: timestamp
              orderDirection: desc
            ) {
              id
              timestamp
              maker
              taker
              takerAssetId
              makerAssetId
              makerAmountFilled
              takerAmountFilled
            }
        }'''
        ts_g = int(start_dt.timestamp())
        ts_l = int(end_dt.timestamp())
        try:
            r = requests.post(self.goldsky_url, json={'query': q, 'variables': {'a': a, 'g': str(ts_g), 'l': str(ts_l)}}, timeout=15)
            if r.status_code != 200 or 'data' not in r.json(): return pd.DataFrame()
            events = r.json()['data']['orderFilledEvents']
            if not events: return pd.DataFrame()
            data = []
            for e in events:
                buy_shares = 'BUY' if str(e['takerAssetId']) == "0" else "SELL"
                m, t = int(e['makerAmountFilled'])/1e6, int(e['takerAmountFilled'])/1e6
                price = (t/m if e['takerAssetId'] == '0' else m/t) if m != 0 else 0
                price_up = round(price, 4) if up_down_key == 'UP' else round(1 - price, 4)
                data.append({
                    'UP_DOWN': up_down_key,
                    'TIMESTAMP': dt.fromtimestamp(int(e['timestamp']), tz=timezone.utc),
                    'MAKER': e['maker'],
                    'TAKER': e['taker'],
                    'SHARES': round(m, 4),
                    'USDC': round(t, 4),
                    'PRICE': round(price, 4),
                    'PRICE_UP': price_up,
                    'BUY_SELL': buy_shares,
                    'id': e['id'],
                    'log_odds': round(np.log(price/(1-price)), 4)})
            return pd.DataFrame(data)
        except Exception:
            return pd.DataFrame()

    def _process_all_orderfills(self):
        tasks = []
        now_utc = dt.now(timezone.utc) - timedelta(days=self.start_day_from_now)

        for _, row in self.df.iterrows():
            tz_name = self.timezone_map.get(row['country'], 'UTC')
            local_tz = pytz.timezone(tz_name)
            target_midnight = local_tz.localize(dt.combine(row['ts'], dt.min.time()))
            prev_close_local = target_midnight - timedelta(hours=8)
            market_start_utc = prev_close_local.astimezone(pytz.UTC)
            hour_step = 0.2 #12min to avoid hitting the 1k limit in orders per api call
            for side in ['up_token', 'down_token']:
                h = 0.0
                while h < 24:
                    chunk_start = market_start_utc + timedelta(hours=h)
                    if chunk_start >= now_utc: break
                    chunk_end = chunk_start + timedelta(hours=hour_step)
                    tasks.append({
                        'a': row[side], 's_dt': chunk_start, 'e_dt': chunk_end,
                        'side': 'UP' if 'up' in side else 'DOWN',
                        'key': row['key'],
                        'country': row['country']})
                    h += hour_step

        def fetch(t):
            res = self._get_single_orderfills(t['a'], t['s_dt'], t['e_dt'], t['side'])
            time.sleep(0.1) #for api call limit too, makes it veery long, but at least ensures we get all data
            if not res.empty:
                res['KEY'] = t['key']
                res['country'] = t['country']
            return res

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(fetch, tasks))

        if not results: return pd.DataFrame()
        results = [r for r in results if not r.empty]
        expected_cols = ['KEY', 'UP_DOWN', 'TIMESTAMP', 'MAKER', 'TAKER', 'SHARES', 'USDC', 'PRICE', 'PRICE_UP','BUY_SELL','id']
        if not results:
            return pd.DataFrame(columns=expected_cols)
        df = pd.concat(results, ignore_index=True)
        if 'id' in df.columns:
            df.drop_duplicates(subset=['id'], keep='first', inplace=True)

        #here you get relative hours wrt the countries' markets, not UTC
        def get_relative_hr(row):
            tz = pytz.timezone(self.timezone_map.get(row['country'].lower(), 'UTC'))
            lt = tz.normalize(row['TIMESTAMP'].astimezone(tz))
            return (lt.hour + lt.minute/60 + lt.second/3600 + 8) % 24

        df['REL_HOUR'] = df.apply(get_relative_hr, axis=1)
        df['TIME_TO_EXP'] = (24.0 - df['REL_HOUR']).clip(lower=0)
        self.orderbook = df.sort_values(by=['KEY', 'TIMESTAMP'], ascending=[True, False]).reset_index(drop=True)
        cols = ['KEY', 'REL_HOUR', 'TIME_TO_EXP'] + [c for c in self.orderbook.columns if c not in ['id', 'KEY', 'REL_HOUR', 'TIME_TO_EXP']]
        self.orderbook = self.orderbook[cols]

        return self.orderbook

    def _pull_stock_close(self):
        tickers = self.orderbook['KEY'].unique().tolist()
        start, end = self.orderbook['TIMESTAMP'].min(), self.orderbook['TIMESTAMP'].max()
        df = yf.download(tickers, start=start.date(), end=(end + timedelta(days=1)).date(), progress=False, auto_adjust=True)
        df = df[['Open', 'Close']].stack(future_stack=True).reset_index()

        df.columns = ['date', 'KEY', 'close', 'open']
        df['date'] = df['date'].dt.date
        valid_days = self.orderbook.assign(date=self.orderbook['TIMESTAMP'].dt.date)[['KEY', 'date']].drop_duplicates()
        conds = [df['close'] > df['open'], df['close'] < df['open']]
        df['stock_up'] = np.select(conds, [1, 0], default=0.5)
        df = df.rename(columns={'close':'stock_close_day', 'open':'stock_open_day'})

        return df.merge(valid_days, on=['KEY', 'date'])[['KEY', 'date', 'stock_open_day', 'stock_close_day']]

    def _pull_stock_minutes(self, minutes=5, window_size=5):
        keys = self.orderbook['KEY'].unique().tolist()
        start = self.orderbook['TIMESTAMP'].min()
        end = self.orderbook['TIMESTAMP'].max()
        cutoff = dt.now(pytz.UTC) - timedelta(days=59)
        fetch_start = max(start, cutoff)

        if fetch_start >= end: return pd.DataFrame()
        df = yf.download(keys, start=fetch_start.date(), end=(end + timedelta(days=1)).date(),
                         interval=f"{minutes}m", progress=False, auto_adjust=True)
        suffix = f"_{minutes}m"
        if isinstance(df.columns, pd.MultiIndex):
            df = df[['High', 'Low', 'Close', 'Open', 'Volume']].stack(level=1, future_stack=True).reset_index()
            df.columns = ['TIMESTAMP', 'KEY', f'stock_close{suffix}', f'stock_high{suffix}', f'stock_low{suffix}', f'stock_open{suffix}', f'stock_volume{suffix}']
        else:
            df = df[['High', 'Low', 'Close', 'Open', 'Volume']].reset_index()
            df['KEY'] = keys[0]
            df.columns = ['TIMESTAMP', f'stock_close{suffix}', f'stock_high{suffix}', f'stock_low{suffix}', f'stock_open{suffix}', f'stock_volume{suffix}', 'KEY']

        #avg stock price from high-low and close price
        df[f'stock_avg{suffix}'] = (df[f'stock_high{suffix}'] + df[f'stock_low{suffix}'] + df[f'stock_close{suffix}']) / 3
        if df['TIMESTAMP'].dt.tz is None:
            df['TIMESTAMP'] = df['TIMESTAMP'].dt.tz_localize('UTC')
        else:
            df['TIMESTAMP'] = df['TIMESTAMP'].dt.tz_convert('UTC')
        df = df.sort_values(['KEY', 'TIMESTAMP'])
        df['returns'] = df.groupby('KEY')[f'stock_close{suffix}'].transform(lambda x: np.log(x / x.shift(1)))
        ann_factor = np.sqrt((60/minutes * 6.5) * 252)
        df[f'stock_vol{suffix}'] = df.groupby('KEY')['returns'].transform(
            lambda x: x.rolling(window=window_size).std() * ann_factor)
        return df.sort_values('TIMESTAMP')