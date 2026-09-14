
import io
import pandas as pd
import requests
import pandas as pd
from finvizfinance.screener.overview import Overview
from constants import *



def get_tickers(my_filters):
    match CONFIG["MARKET_SIZE"]:
        case 0:
            return get_SmallTickers(my_filters)
        case 1:
            return get_MediumTickers(my_filters)
        case 2:
            return get_LargeTickers(my_filters)
        case 3:
            return get_DebugTickers(my_filters)


def get_DebugTickers(my_filters):
    return fetch_tickers(my_filters)

    
def get_LargeTickers(my_filters):
    return fetch_tickers(my_filters)

def get_MediumTickers(my_filters):
    return fetch_tickers(my_filters)

def get_SmallTickers(my_filters):
    return fetch_tickers(my_filters)




def fetch_tickers(my_filters):
    """
    Fetch tickers from Finviz screener based on market cap size.

    Args:
        num (int): 0 for Small cap, 1 for Mid cap, 2 for Large cap.

    Returns:
        list: List of ticker symbols matching the filters.
    """
    

    
    # 1. Initialize the Overview screener
    screener = Overview()

    

    # 3. Apply filters
    screener.set_filter(filters_dict=my_filters)

    # 4. Fetch data
    df = screener.screener_view()
    if df is None or df.empty:
      print("No tickers matched the current filters.")
      return []
    tickers = df['Ticker'].tolist()

    print(f"Total Tickers ({len(tickers)}):")
    #print(tickers)

    return tickers
