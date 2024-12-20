# %% [markdown]
# # High-Frequency Grid Trading
# 
# <div class="alert alert-info">
#     
# **Note:** This example is for educational purposes only and demonstrates effective strategies for high-frequency market-making schemes. All backtests are based on a 0.005% rebate, the highest market maker rebate available on Binance Futures. See <a href="https://www.binance.com/en/support/announcement/binance-updates-usd%E2%93%A2-margined-futures-liquidity-provider-program-2024-06-03-fefc6aa25e0947e2bf745c1c56bea13e">Binance Upgrades USDⓢ-Margined Futures Liquidity Provider Program</a> for more details.
#     
# </div>

# %% [markdown]
# ## Plain High-Frequency Grid Trading
# 
# This is a high-frequency version of Grid Trading that keeps posting orders on grids centered around the mid-price, maintaining a fixed interval and a set number of grids.

# %%


DEBUG = False
#! fixed_spread_trading strat
import numpy as np
from datetime import datetime, timedelta
from numba import njit, uint64, float64
from numba.typed import Dict
import polars as pl
import pandas as pd
import time
from hftbacktest import BUY, SELL, GTX, LIMIT


out_dtype = np.dtype([
    ('cur_ts', '<i8'),
    ('exch_ts', '<i8'),
    ('local_ts', '<i8'),
    ('mid', 'f8'),
    ('half_spread_tick', 'f8'),
    ('bid_price', 'f8'),
    ('ask_price', 'f8'),
])

@njit
def fixed_spread_trading(hbt, recorder):
    asset_no = 0
    tick_size = hbt.depth(asset_no).tick_size
    half_spread = tick_size * 10
    # out = np.zeros(500_000_000, out_dtype)
    t = 0
    # Running interval in nanoseconds.progress = 0
    progress = 0
    print('Start')
    while hbt.elapse(100_000_000) == 0:
        progress += 100_000_000 / 1_000_000_000 / 60 / 60 
        print('Progress:', progress, 'hours')
        # Clears cancelled, filled or expired orders.        
        hbt.clear_inactive_orders(asset_no)
        
        cur_ts = hbt.current_timestamp
        exch_ts,local_ts = hbt.feed_latency(asset_no)

        depth = hbt.depth(asset_no)
        position = hbt.position(asset_no)
        orders = hbt.orders(asset_no)
        
        best_bid = depth.best_bid
        best_ask = depth.best_ask
        
        mid_price = (best_bid + best_ask) / 2.0

        order_qty = 0.1 # np.round(notional_order_qty / mid_price / hbt.depth(asset_no).lot_size) * hbt.depth(asset_no).lot_size
        
        # Aligns the prices to the grid.
        bid_price = mid_price - half_spread
        ask_price = mid_price + half_spread

        #--------------------------------------------------------
        # Updates quotes.
        
        # Creates a new grid for buy orders.
        new_bid_orders = Dict.empty(np.uint64, np.float64)
        bid_price_tick = round(bid_price / tick_size)
        # order price in tick is used as order id.
        new_bid_orders[uint64(bid_price_tick)] = bid_price
                

        # Creates a new grid for sell orders.
        new_ask_orders = Dict.empty(np.uint64, np.float64)
        ask_price_tick = round(ask_price / tick_size)
        # order price in tick is used as order id.
        new_ask_orders[uint64(ask_price_tick)] = ask_price
                
        order_values = orders.values()
        while order_values.has_next():
            order = order_values.get()
            # Cancels if a working order is not in the new grid.
            if order.cancellable:
                if (
                    (order.side == BUY and order.order_id not in new_bid_orders)
                    or (order.side == SELL and order.order_id not in new_ask_orders)
                ):
                    hbt.cancel(asset_no, order.order_id, False)
                    
        for order_id, order_price in new_bid_orders.items():
            # Posts a new buy order if there is no working order at the price on the new grid.
            if order_id not in orders:
                hbt.submit_buy_order(asset_no, order_id, order_price, order_qty, GTX, LIMIT, False)
                
        for order_id, order_price in new_ask_orders.items():
            # Posts a new sell order if there is no working order at the price on the new grid.
            if order_id not in orders:
                hbt.submit_sell_order(asset_no, order_id, order_price, order_qty, GTX, LIMIT, False)
        
        # out[t].cur_ts = cur_ts
        # out[t].exch_ts = exch_ts
        # out[t].local_ts = local_ts
        # out[t].mid = mid_price
        # out[t].half_spread_tick = half_spread / tick_size
        # out[t].bid_price = bid_price
        # out[t].ask_price = ask_price # 
        
        t += 1
        # Records the current state for stat calculation.
        recorder.record(hbt)
        
        if DEBUG and t >= 100:
            break
    # return out[:t]

# %% [markdown]
# For generating order latency from the feed data file, which uses feed latency as order latency, please see [Order Latency Data](https://hftbacktest.readthedocs.io/en/latest/tutorials/Order%20Latency%20Data.html).

# # %%
from hftbacktest import BacktestAsset, ROIVectorMarketDepthBacktest, Recorder,HashMapMarketDepthBacktest
   
from datetime import datetime, timedelta
import os

def backtest(args):
    asset_name, asset_info= args["asset_name"], args["asset_info"]
    fill_exchange = args["fill_exchange"]
    begin_date, end_date = args["begin_date"], args["end_date"]
    queue_model = args["queue_model"]
    queue_params = args["queue_params"]
    data_dir = args["data_dir"]
    out_dir = args["out_dir"]
    # Obtains the mid-price of the assset to determine the order quantity.
    start = datetime.strptime(begin_date, "%Y%m%d")
    prev_date = (start - timedelta(days=1)).strftime("%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    out_path = f'{out_dir}/fixed_spread_strat_{fill_exchange}_{asset_name}_{queue_model}_{queue_params}_{begin_date}_{end_date}.npz'
    if os.path.exists(out_path):
        print(f'{out_path} already exists')
        return
    # 生成日期列表
    date_list = []
    current_date = start
    while current_date <= end:
        # 将日期格式化为字符串并添加到列表中
        date_list.append(current_date.strftime("%Y%m%d"))
        # 增加一天
        current_date += timedelta(days=1)
    asset = (
        BacktestAsset()
            .data([f'{data_dir}/{asset_name}_{date}.npz' for date in date_list])
            .initial_snapshot(f'{data_dir}/{asset_name}_{prev_date}_eod.npz')
            .linear_asset(1.0) 
            .constant_latency(10_000_000,20_000_000) # constant_latency 
            .trading_value_fee_model(0, 0.0007)
            .tick_size(asset_info['tick_size'])
            .lot_size(asset_info['lot_size'])
            .roi_lb(0.0)    
            .roi_ub(7000)
    )
    if fill_exchange == 'no_partial':
        asset.no_partial_fill_exchange()
    elif fill_exchange == 'partial_fill':
        asset.partial_fill_exchange()

    if queue_model == 'PowerProbQueueModel1':
        asset.power_prob_queue_model(queue_params)
    elif queue_model == 'PowerProbQueueModel2':
        asset.power_prob_queue_model2(queue_params)
    elif queue_model == 'PowerProbQueueModel3':
        asset.power_prob_queue_model3(queue_params) # pay attention to the parameter
    elif queue_model == 'LogProbQueueModel':
        asset.log_prob_queue_model()
    elif queue_model == 'LogProbQueueModel2':
        asset.log_prob_queue_model2()
    elif queue_model == 'RiskAdverseQueueModel': #risk_adverse_queue_model
        asset.risk_adverse_queue_model()


    hbt = ROIVectorMarketDepthBacktest([asset])

    recorder = Recorder(1, 30_000_000)
    
    
    fixed_spread_trading(hbt, recorder.recorder) #TODO:add strat name for multiple strats

    hbt.close()

    recorder.to_npz(out_path)




n_jobs = 1
basic_args= {
    "asset_name": "ETHUSDT",
    "asset_info": {
        "tick_size": 0.01,
        "lot_size": 0.001
    },
    "fill_exchange": "no_partial",
    "begin_date": "20240320",
    "end_date": "20240420",
    "data_dir": "/mnt/data/hftbacktest_data/data",
    "out_dir": "report"
}

args_list = []

no_param_queue_models = ['LogProbQueueModel', 'LogProbQueueModel2', 'RiskAdverseQueueModel']

for queue_model in no_param_queue_models:
    args = basic_args.copy()
    args["queue_model"] = queue_model
    args["queue_params"] = None
    args_list.append(args)

has_param_queue_models = ['PowerProbQueueModel1', 'PowerProbQueueModel2', 'PowerProbQueueModel3']
param_list = [2,3,9,20]
for queue_model in has_param_queue_models:
    for queue_params in param_list:
        args = basic_args.copy()
        args["queue_model"] = queue_model
        args["queue_params"] = queue_params
        args_list.append(args)

from multiprocessing import Pool

with Pool(n_jobs) as p:
    p.map(backtest, args_list)

out_name_list = []
for arg in args_list:
    out_name_list.append(f'fixed_spread_strat_{arg["fill_exchange"]}_{arg["asset_name"]}_{arg["queue_model"]}_{arg["queue_params"]}_{arg["begin_date"]}_{arg["end_date"]}.npz')

import polars as pl
from matplotlib import pyplot as plt

# %%
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
def extract_time_range(arr, start_time_str, end_time_str):
    # 将字符串时间转换为datetime对象
    start_time = pd.to_datetime(start_time_str)
    end_time = pd.to_datetime(end_time_str)
    
    # 计算起始和结束时间对应的纳秒数
    start_ns = int(start_time.timestamp() * 1e9)
    end_ns = int(end_time.timestamp() * 1e9)
    
    # 提取指定时间范围内的数据
    filtered_arr = arr[(arr['timestamp'] >= start_ns) & (arr['timestamp'] <= end_ns)]
    
    return filtered_arr



import polars as pl
from matplotlib import pyplot as plt
from hftbacktest.stats import LinearAssetRecord

def compute_net_equity(out_path):
    equity_values = {}
    sr_values = {}
    data = np.load(out_path)['0']
    stats = (
        LinearAssetRecord(data)
            .resample('5m')
            .stats()
    )

    equity = stats.entire.with_columns(
        (pl.col('equity_wo_fee') - pl.col('fee')).alias('equity')
    ).select(['timestamp', 'equity'])

    return equity



np.seterr(divide='ignore', invalid='ignore')

fig = plt.figure()
fig.set_size_inches(10, 3)
legend = []
out_dir = basic_args['out_dir']
for out_name in out_name_list:
    out_path = f'{out_dir}/{out_name}'
    expt_name = out_name.split('.')[0]
    net_equity_ = compute_net_equity(out_path)

    pnl = net_equity_['equity'].diff()
    # Since the P&L is resampled at a 5-minute interval
    sr = pnl.mean() / pnl.std() * np.sqrt(24 * 60 / 5)
    legend.append('100 assets, Daily SR={:.2f}, {}'.format(sr, expt_name))
    plt.plot(net_equity_['timestamp'], net_equity_['equity'] * 100)
    
plt.legend(
    legend,
    loc='upper center', bbox_to_anchor=(0.5, -0.15),
    fancybox=True, shadow=True, ncol=3
)

plt.grid()
plt.ylabel('Cumulative Returns (%)')
plt.savefig(f'{out_dir}/fixed_spread_strat.png', bbox_inches='tight')