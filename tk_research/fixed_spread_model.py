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

# %%


DEBUG = True

import numpy as np

from numba import njit, uint64, float64
from numba.typed import Dict

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
    out = np.zeros(10_000_000, out_dtype)
    t = 0
    # Running interval in nanoseconds.
    while hbt.elapse(100_000_000) == 0:
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
                
        order_values = orders.values();
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
        
        out[t].cur_ts = cur_ts
        out[t].exch_ts = exch_ts
        out[t].local_ts = local_ts
        out[t].mid = mid_price
        out[t].half_spread_tick = half_spread / tick_size
        out[t].bid_price = bid_price
        out[t].ask_price = ask_price # 
        
        t += 1
        # Records the current state for stat calculation.
        recorder.record(hbt)
        
        if DEBUG and t >= 100:
            break
    return out[:t]

# %% [markdown]
# For generating order latency from the feed data file, which uses feed latency as order latency, please see [Order Latency Data](https://hftbacktest.readthedocs.io/en/latest/tutorials/Order%20Latency%20Data.html).

# # %%
from hftbacktest import BacktestAsset, ROIVectorMarketDepthBacktest, Recorder
# import hydra

# @hydra.main(config_path='conf', config_name='config')
# def asset_generate(cfg):
#     asset = (
#         BacktestAsset()
#             .data([
#                 'data/ETHUSDT_20221003.npz',
#             ])
#             .initial_snapshot('data/ETHUSDT_20221002_eod.npz')
#             .linear_asset(1.0) 
#             .constant_latency(10_000_000,20_000_000) # constant_latency 
#             .partial_fill_exchange()#! no partial fill
#             .trading_value_fee_model(0, 0.0007) #! 0 bps for maker, 7 bps for taker
#             .tick_size(0.01)
#             .lot_size(0.001)
#             .roi_lb(0.0)    
#             .roi_ub(3000.0)
#     )
#     if cfg["queue_model"] == "power_prob":
#         asset.power_prob_queue_model(2.0) #!
   
    

asset = (
    BacktestAsset()
        .data([
            'data/ETHUSDT_20221003.npz',
        ])
        .initial_snapshot('data/ETHUSDT_20221002_eod.npz')
        .linear_asset(1.0) 
        .constant_latency(10_000_000,20_000_000) # constant_latency 
        .power_prob_queue_model(2.0) #!
        .partial_fill_exchange()#! no partial fill
        .trading_value_fee_model(0, 0.0007) #! 0 bps for maker, 7 bps for taker
        .tick_size(0.01)
        .lot_size(0.001)
        .roi_lb(0.0)    
        .roi_ub(3000.0)
)
hbt = ROIVectorMarketDepthBacktest([asset])

recorder = Recorder(1, 5_000_000)

# %%

out = fixed_spread_trading(hbt, recorder.recorder)

hbt.close()

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

# %%
records = recorder.get(0)

merged_field_names = list(out.dtype.names) + list(records.dtype.names)
merged_field_dtypes = list(out.dtype.descr) + list(records.dtype.descr)

# 创建合并后的结构化数组
merged_array = np.zeros(len(out), dtype=merged_field_dtypes)

# 填充数据
for name in out.dtype.names:
    merged_array[name] = out[name]
for name in records.dtype.names:
    merged_array[name] = records[name]


# add records['timestamp'] and records['price'] to out use np array method 


selected_range = extract_time_range(merged_array, "20221003 07:00:00", "20221003 19:00:00")

# %%
from hftbacktest.stats import LinearAssetRecord
stats = LinearAssetRecord(selected_range).stats(book_size=10_000)

stats.summary()
# %%
stats.plot()

# save plot
import matplotlib.pyplot as plt
plt.savefig('stats/gridtrading_simple_hf_mm1_ETHUSDT.png')



