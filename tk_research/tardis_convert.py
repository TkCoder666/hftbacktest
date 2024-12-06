from hftbacktest.data.utils import tardis
from datetime import datetime, timedelta
# _ = tardis.convert(
#     ['BTCUSDT_trades.csv.gz', 'BTCUSDT_book.csv.gz'],
#     output_filename='btcusdt_20200201.npz',
#     buffer_size=200_000_000
# )
sym =  "ETHUSDT"
from_date = "20221002"
end_date = "20221007"

def convert_depth_data(sym, from_date, end_date, from_dir,out_dir="."):
    # Convert string dates to datetime objects
    start = datetime.strptime(from_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    current_date = start
    while current_date <= end:
        current_date_str = current_date.strftime("%Y%m%d")
        input_files = [f"{from_dir}/{sym}_trades_{current_date_str}.csv.gz", f"{from_dir}/{sym}_incremental_book_L2_{current_date_str}.csv.gz"]
        output_file = f"{out_dir}/{sym}_{current_date_str}.npz"
        _ = tardis.convert(input_files, output_filename=output_file, buffer_size=2_000_000_000)
        current_date += timedelta(days=1)

convert_depth_data(sym, from_date, end_date, "/home/tangke/code/hftbacktest/data/tardis_data","/home/tangke/code/hftbacktest/data")