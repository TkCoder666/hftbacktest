import numpy as np

from numba import njit

from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest, BUY, SELL, GTX, LIMIT

import time

import numpy as np

from numba import njit, uint64,float64
from numba.typed import Dict

from hftbacktest import (
    BacktestAsset,
    ROIVectorMarketDepthBacktest,
    GTX,
    LIMIT,
    BUY,
    SELL,
    BUY_EVENT,
    SELL_EVENT,
    Recorder
)
from hftbacktest.stats import LinearAssetRecord

@njit(cache=True)
def compute_coeff_simplified(gamma, delta, A, k):
    inv_k = np.divide(1, k)
    c1 = inv_k
    c2 = np.sqrt(np.divide(gamma * np.exp(1), 2 * A * delta * k))
    return c1, c2

@njit(cache=True)
def measure_trading_intensity(order_arrival_depth, out):
    max_tick = 0
    for depth in order_arrival_depth:
        if not np.isfinite(depth):
            continue

        # Sets the tick index to 0 for the nearest possible best price
        # as the order arrival depth in ticks is measured from the mid-price
        tick = round(depth / .5) - 1

        # In a fast-moving market, buy trades can occur below the mid-price (and vice versa for sell trades)
        # since the mid-price is measured in a previous time-step;
        # however, to simplify the problem, we will exclude those cases.
        if tick < 0 or tick >= len(out):
            continue

        # All of our possible quotes within the order arrival depth,
        # excluding those at the same price, are considered executed.
        out[:tick] += 1

        max_tick = max(max_tick, tick)
    return out[:max_tick]

@njit(cache=True)
def linear_regression(x, y):
    sx = np.sum(x)
    sy = np.sum(y)
    sx2 = np.sum(x ** 2)
    sxy = np.sum(x * y)
    w = len(x)
    slope = np.divide(w * sxy - sx * sy, w * sx2 - sx**2)
    intercept = np.divide(sy - slope * sx, w)
    return slope, intercept



@njit
def gridtrading_glft_mm(hbt, recorder, gamma, order_qty):
    asset_no = 0
    tick_size = hbt.depth(asset_no).tick_size

    arrival_depth = np.full(buffer_size, np.nan, np.float64)
    mid_price_chg = np.full(buffer_size, np.nan, np.float64)

    t = 0
    prev_mid_price_tick = np.nan
    mid_price_tick = np.nan

    tmp = np.zeros(500, np.float64)
    ticks = np.arange(len(tmp)) + 0.5

    A = np.nan
    k = np.nan
    volatility = np.nan
    delta = 1

    grid_num = 20
    max_position = 50 * order_qty
    
    last_lvl1_bid = None
    last_lvl1_ask = None
    reset_cnt = 0
    # Checks every 100 milliseconds.
    while hbt.elapse(100_000_000) == 0:
        #--------------------------------------------------------
        # Records market order's arrival depth from the mid-price.
        if not np.isnan(mid_price_tick):
            depth = -np.inf
            for last_trade in hbt.last_trades(asset_no):
                trade_price_tick = last_trade.px / tick_size

                if last_trade.ev & BUY_EVENT == BUY_EVENT:
                    depth = max(trade_price_tick - mid_price_tick, depth)
                else:
                    depth = max(mid_price_tick - trade_price_tick, depth)
            arrival_depth[t] = depth

        hbt.clear_last_trades(asset_no)
        hbt.clear_inactive_orders(asset_no)

        depth = hbt.depth(asset_no)
        position = hbt.position(asset_no)
        orders = hbt.orders(asset_no)

        best_bid_tick = depth.best_bid_tick
        best_ask_tick = depth.best_ask_tick

        prev_mid_price_tick = mid_price_tick
        mid_price_tick = (best_bid_tick + best_ask_tick) / 2.0

        # Records the mid-price change for volatility calculation.
        mid_price_chg[t] = mid_price_tick - prev_mid_price_tick

        #--------------------------------------------------------
        # Calibrates A, k and calculates the market volatility.

        # Updates A, k, and the volatility every 5-sec.
        if t % 50 == 0:
            # Window size is 10-minute.
            if t >= 6_000 - 1:
                # Calibrates A, k
                tmp[:] = 0
                lambda_ = measure_trading_intensity(arrival_depth[t + 1 - 6_000:t + 1], tmp)
                if len(lambda_) > 2:
                    lambda_ = lambda_[:70] / 600
                    x = ticks[:len(lambda_)]
                    y = np.log(lambda_)
                    k_, logA = linear_regression(x, y)
                    A = np.exp(logA)
                    k = -k_

                # Updates the volatility.
                volatility = np.nanstd(mid_price_chg[t + 1 - 6_000:t + 1]) * np.sqrt(10)

        #--------------------------------------------------------
        # Computes bid price and ask price.

        c1, c2 = compute_coeff_simplified(gamma, delta, A, k)

        half_spread_tick = c1 + delta / 2 * c2 * volatility
        skew = c2 * volatility

        normalized_position = position / order_qty

        reservation_price_tick = mid_price_tick - skew * normalized_position

        bid_price_tick = min(np.round(reservation_price_tick - half_spread_tick), best_bid_tick)
        ask_price_tick = max(np.round(reservation_price_tick + half_spread_tick), best_ask_tick)

        bid_price = bid_price_tick * tick_size
        ask_price = ask_price_tick * tick_size

        grid_interval = max(np.round(half_spread_tick) * tick_size, tick_size)

        bid_price = np.floor(bid_price / grid_interval) * grid_interval
        ask_price = np.ceil(ask_price / grid_interval) * grid_interval

        #--------------------------------------------------------
        # Updates quotes.

        # Creates a new grid for buy orders.
        if (last_lvl1_bid is None or abs(last_lvl1_bid - bid_price) / last_lvl1_bid> reset_bps / 1e4) or (last_lvl1_ask is None or abs(last_lvl1_ask - ask_price) / last_lvl1_ask > reset_bps / 1e4):
            last_lvl1_bid = bid_price
            last_lvl1_ask = ask_price
            reset_cnt += 1
            order_values = orders.values()
            while order_values.has_next():
                order = order_values.get()
                # Cancels if a working order is not in the new grid.
                if order.cancellable:
                    hbt.cancel(asset_no, order.order_id, False)

            new_bid_orders = Dict.empty(uint64, float64)
            if position < max_position and np.isfinite(bid_price):
                for i in range(grid_num):
                    bid_price_tick = round(bid_price / tick_size)
                    # order price in tick is used as order id.
                    new_bid_orders[uint64(bid_price_tick)] = bid_price
                    bid_price -= grid_interval

            for order_id, order_price in new_bid_orders.items():
                # Posts a new buy order if there is no working order at the price on the new grid.
                if order_id not in orders:
                    hbt.submit_buy_order(asset_no, order_id, order_price, order_qty, GTX, LIMIT, False)        
                    
            
            new_ask_orders = Dict.empty(uint64, float64)
            if position > -max_position and np.isfinite(ask_price):
                for i in range(grid_num):
                    ask_price_tick = round(ask_price / tick_size)
                    # order price in tick is used as order id.
                    new_ask_orders[uint64(ask_price_tick)] = ask_price
                    ask_price += grid_interval
                    
            for order_id, order_price in new_ask_orders.items():
                # Posts a new sell order if there is no working order at the price on the new grid.
                if order_id not in orders:
                    hbt.submit_sell_order(asset_no, order_id, order_price, order_qty, GTX, LIMIT, False)
        #--------------------------------------------------------
        # Records variables and stats for analysis.

        t += 1

        if t >= len(arrival_depth) or t >= len(mid_price_chg):
            raise Exception

        # Records the current state for stat calculation.
        recorder.record(hbt)
    return reset_cnt
        

from hftbacktest.stats import LinearAssetRecord
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

if __name__ == '__main__':
    # This backtest assumes market maker rebates.
    # https://www.binance.com/en/support/announcement/binance-upgrades-usd%E2%93%A2-margined-futures-liquidity-provider-program-2023-04-04-01007356e6514df3811b0c80ab8c83bf
    asset_name = "ETHUSDT"
    begin_date =  "20240320"
    end_date =  "20240604"
    data_dir =  "/mnt/data/hftbacktest_data/data"
    start = datetime.strptime(begin_date, "%Y%m%d")
    prev_date = (start - timedelta(days=1)).strftime("%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
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
            .data(
                [f'{data_dir}/{asset_name}_{date}.npz' for date in date_list])
            .linear_asset(1.0)
            .constant_latency(10_000_000,20_000_000)
            .risk_adverse_queue_model()
            .no_partial_fill_exchange()
            .trading_value_fee_model(-0.00005, 0.0007)
            .tick_size(0.01)
            .lot_size(0.001)
            .roi_lb(0.0)    
            .roi_ub(7000.0)
            .last_trades_capacity(10000)
    )
    print("Start backtest.")
    gamma = 5
    order_qty = 1
    reset_bps_list = [1,5,10,25,50]
    for reset_bps in reset_bps_list:
        buffer_size = 500_000_000
        recorder = Recorder(1, buffer_size)
        hbt = ROIVectorMarketDepthBacktest([asset])
        t1 = time.time()
        reset_cnt = gridtrading_glft_mm(hbt, recorder.recorder, gamma, order_qty)
        avg_reset_cnt = reset_cnt / len(date_list)
        hbt.close()
        t2 = time.time()
        print(f"Elapsed time: {t2 - t1:.2f} seconds for the backtest.")
        recorder.to_npz(f'stats/reset_expt/gridtrading_simple_glft_mm1_{asset_name}_{gamma}_{order_qty}_{begin_date}_{end_date}_{reset_bps}_{avg_reset_cnt}.npz')
        stats = LinearAssetRecord(recorder.get(0)).stats()
        print(stats.summary())
        stats.plot()
        plt.savefig(f'stats/reset_expt/gridtrading_simple_glft_mm1_{asset_name}_{gamma}_{order_qty}_{begin_date}_{end_date}_{reset_bps}_{avg_reset_cnt}.png')
        
