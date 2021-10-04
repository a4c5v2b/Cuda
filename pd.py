from numba import jit
import pandas as pd

x = {'a': [1, 2, 3], 'b': [20, 30, 40]}

@jit
def use_pandas(a): # Numba对这个函数支持不好
    df = pd.DataFrame.from_dict(a) # Numba 不知道 pd.DataFrame 在做什么
    df += 1                        # Numba 也无法优化这个操作
    return df.cov()

print(use_pandas(x))