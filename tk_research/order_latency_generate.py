import numpy as np
from numba import njit
import polars as pl
from hftbacktest import LOCAL_EVENT, EXCH_EVENT
from datetime import datetime, timedelta
from tqdm import tqdm

@njit
def generate_order_latency_nb(data, order_latency, mul_entry, offset_entry, mul_resp, offset_resp):
    for i in range(len(data)):
        exch_ts = data[i].exch_ts
        local_ts = data[i].local_ts
        feed_latency = local_ts - exch_ts
        order_entry_latency = mul_entry * feed_latency + offset_entry
        order_resp_latency = mul_resp * feed_latency + offset_resp

        req_ts = local_ts
        order_exch_ts = req_ts + order_entry_latency
        resp_ts = order_exch_ts + order_resp_latency

        order_latency[i].req_ts = req_ts
        order_latency[i].exch_ts = order_exch_ts
        order_latency[i].resp_ts = resp_ts

def generate_order_latency(feed_file, output_file = None, mul_entry = 1, offset_entry = 0, mul_resp = 1, offset_resp = 0):
    data = np.load(feed_file)['data']
    df = pl.DataFrame(data)

    df = df.filter(
        (pl.col('ev') & EXCH_EVENT == EXCH_EVENT) & (pl.col('ev') & LOCAL_EVENT == LOCAL_EVENT)
    ).with_columns(
        pl.col('local_ts').alias('ts')
    ).group_by_dynamic(
        'ts', every='1000000000i'
    ).agg(
        pl.col('exch_ts').last(),
        pl.col('local_ts').last()
    ).drop('ts')

    data = df.to_numpy(structured=True)

    order_latency = np.zeros(len(data), dtype=[('req_ts', 'i8'), ('exch_ts', 'i8'), ('resp_ts', 'i8'), ('_padding', 'i8')])
    generate_order_latency_nb(data, order_latency, mul_entry, offset_entry, mul_resp, offset_resp)

    if output_file is not None:
        np.savez_compressed(output_file, data=order_latency)

    return order_latency

sym = "ETHUSDT"
npz_dir = "data"
latency_dir= "data/latency"
from_date = "20221002"
end_date = "20221007"
start = datetime.strptime(from_date, "%Y%m%d")
end = datetime.strptime(end_date, "%Y%m%d")
current_date = start
# add tqdm to show progress bar
pbar = tqdm(total=(end-start).days+1)
while current_date <= end:  
    current_date_str = current_date.strftime("%Y%m%d")
    feed_file = f"{npz_dir}/{sym}_{current_date_str}.npz"
    output_file = f"{latency_dir}/{sym}_{current_date_str}_latency.npz"
    generate_order_latency(feed_file, output_file)
    current_date += timedelta(days=1)
    pbar.update(1)
     