from sklearn.linear_model import LinearRegression

def adj_2r(x, y) -> dict:
    model = LinearRegression()
    model.fit(x, y)
    r  = model.score(x, y)
    n  = x.shape[0]
    p  = x.shape[1]
    r2 = 1 - ( 1 - r ) * ( n - 1 ) / ( n - p - 1 ) # 自由度修正済み決定係数
    return {
        'r_score': r,
        'data_row_cnt': n,
        'data_col_cnt': p,
        'adj_r_score': r2
    }
