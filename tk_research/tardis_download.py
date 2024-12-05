import json
import subprocess
from datetime import datetime, timedelta
# read json from asset/tardis_api.json
with open('/home/tangke/code/hftbacktest/asset/tardis_api.json') as f:
    data = json.load(f)
api_key = data['api_key']
authorization_str = f"Authorization: Bearer {api_key}"
# x = 1
# wget --header="Authorization: Bearer {}" "https://datasets.tardis.dev/v1/binance-futures/book_snapshot_25/2020/09/02/BTCUSDT.csv.gz --O BTCUSDT_trades_20200902.csv.gz"

def download_depth_data(sym, from_date, end_date,date_type,out_dir="."):
    # Convert string dates to datetime objects
    start = datetime.strptime(from_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    
    # Loop through each day from start to end
    current_date = start
    while current_date <= end:
        current_date_str = current_date.strftime("%Y%m%d")
        url = f"https://datasets.tardis.dev/v1/binance-futures/{date_type}/{current_date_str[:4]}/{current_date_str[4:6]}/{current_date_str[6:8]}/{sym}.csv.gz"
        output_file = f"{out_dir}/{sym}_{date_type}_{current_date_str}.csv.gz"
        command = ["wget", "--header", authorization_str, url, "-O", output_file]
        subprocess.run(command)
        
        # Move to the next day
        current_date += timedelta(days=1)
date_type_list = ["incremental_book_L2","trades"]
sym = "ETHUSDT"
from_date = "20221002"
end_date = "20221007"
for date_type in date_type_list:
    download_depth_data(sym, from_date, end_date,date_type,"/home/tangke/code/hftbacktest/data/tardis_data")