from hftbacktest.data.utils.snapshot import create_last_snapshot

# Builds 20240808 End of Day snapshot. It will be used for the initial snapshot for 20240809.
data_dir = "/mnt/data/hftbacktest_data/data"
_ = create_last_snapshot(
    [f'{data_dir}/ETHUSDT_20240319.npz'],
    tick_size=0.1,
    lot_size=0.001,
    output_snapshot_filename=f'/mnt/data/hftbacktest_data/eod_data/ETHUSDT_20240319_eod.npz'
)