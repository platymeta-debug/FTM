from .data import load_klines

# simple loader stub
def load_dataset(cfg):
    """load_dataset uses cfg.BT_SYMBOLS and returns dict of symbol->rows"""
    data = {}
    for sym in cfg.BT_SYMBOLS:
        path = f"./data/{sym}_{cfg.BT_TF}.csv"
        try:
            data[sym] = load_klines(sym, cfg.BT_TF, path)
        except FileNotFoundError:
            data[sym] = []
    return data
