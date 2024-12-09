import numpy as np

# 假设已存在两个结构化数组 out 和 records
out = np.zeros(3, dtype=[('cur_ts', '<i8'), ('exch_ts', '<i8'), ('local_ts', '<i8'),
                         ('mid', '<f8'), ('half_spread_tick', '<f8'),
                         ('bid_price', '<f8'), ('ask_price', '<f8')])

records = np.zeros(3, dtype=[('timestamp', '<i8'), ('price', '<f8'), ('position', '<f8'),
                             ('balance', '<f8'), ('fee', '<f8'), ('num_trades', '<i8'),
                             ('trading_volume', '<f8'), ('trading_value', '<f8')])

# 自动生成合并后的字段名和 dtype
merged_field_names = list(out.dtype.names) + list(records.dtype.names)
merged_field_dtypes = list(out.dtype.descr) + list(records.dtype.descr)

# 创建合并后的结构化数组
merged_array = np.zeros(len(out), dtype=merged_field_dtypes)

# 填充数据
for name in out.dtype.names:
    merged_array[name] = out[name]
for name in records.dtype.names:
    merged_array[name] = records[name]

# 查看合并结果
print(merged_array.dtype)
print(merged_array)
