import polars as pl
import numpy as np
# Load the data


if __name__ == '__main__':
    # load data/btcusdt_20240730.npz use polars
    data  = np.load('data/btcusdt_20240730.npz')

    # 假设文件中有一个名为 'data' 的数组
    array_data = data['data']

    # 定义所需的 dtype
    dtype = np.dtype([
        ('ev', '<u8'),
        ('exch_ts', '<i8'),
        ('local_ts', '<i8'),
        ('px', '<f8'),
        ('qty', '<f8'),
        ('order_id', '<u8'),
        ('ival', '<i8'),
        ('fval', '<f8')
    ], align=True)

    # 创建一个新的 ndarray，并指定 dtype
    result_array = np.ndarray(shape=array_data.shape, dtype=dtype)

    # 将数据从原始数组复制到新数组
    for name in dtype.names:
        result_array[name] = array_data[name]

    # 打印结果
    df = pl.DataFrame(result_array)
    
    # check if px has val <= 0