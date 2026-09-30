




tickers = [

   "A", "AAOI", "AAPL", "ABBV", "ABNB", "ACMR", "ADI", "AER", "AIR", "ALAB",
    "AMAT", "AMD", "AME", "AMRX", "AMZN", "ANET", "APH", "ARMK", "ARQT", "ATI",
    "ATRC", "ATRO", "AU", "AVAH", "AVGO", "AVNT", "AVPT", "AXTA", "BDX", "BE",
    "BIIB", "BMRN", "BMY", "BTSG", "BULL", "BWA", "CAH", "CART", "CAT", "CDE",
    "CDNA", "CDNS", "CDW", "CGNX", "CHRD", "CIEN", "CNK", "COHR", "COP", "CORT",
    "CRBG", "CRDO", "CRM", "CRVW", "CRWD", "CSCO", "CTAS", "CTVA", "CVLT", "CVX",
    "CXW", "DAR", "DASH", "DDOG", "DE", "DELL", "DGX", "DHR", "DIS", "DK",
    "DOCN", "DT", "DVN", "DXCM", "ECL", "EL", "ELF", "ELV", "EMR", "ENTG",
    "EOG", "ESI", "ESTC", "ET", "ETSY", "EW", "EXEL", "EXLS", "EXPE", "FCX",
    "FIGS", "FIVE", "FIVN", "FLEX", "FLS", "FLYW", "FORM", "FRSH", "FTI", "GDDY",
    "GEV", "GLW", "GNRC", "GOOG", "GOOGL", "GPN", "GTES", "GTX", "HALO", "HPE",
    "HQY", "HSIC", "HTGC", "HUM", "IFF", "INCY", "INGM", "INOD", "INSW", "INTC",
    "IOT", "IQV", "IREN", "ITT", "JBL", "KDP", "KEYS", "KLAC", "KO", "LECO",
    "LITE", "LLY", "LNG", "LRCX", "MANH", "MCHP", "MDB", "META", "MGNI", "MMM",
    "MNST", "MPC", "MPLX", "MRK", "MRVL", "MSFT", "MTCH", "MTSI", "MU", "NBIS",
    "NEM", "NESR", "NOW", "NTAP", "NTNX", "NVDA", "NWS", "NWSA", "OKE", "OKTA",
    "ONTO", "ORCL", "OVV", "OXY", "P", "PAA", "PAGP", "PANW", "PARR", "PAY",
    "PCTY", "PDFS", "PG", "PH", "PLTR", "PR", "PSX", "Q", "QCOM", "REGN",
    "RELY", "RGEN", "RGLD", "RKLB", "ROK", "ROST", "SANM", "SCHW", "SHC", "SITM",
    "SKHY", "SLB", "SM", "SMTC", "SN", "SNDK", "SNX", "SOFI", "SPCX", "SSRM",
    "ST", "STT", "TER", "TKR", "TMO", "TOST", "TRGP", "TSLA", "TSM", "TTC",
    "TTEK", "TWLO", "TXN", "UBER", "UNH", "USFD", "VCYT", "VEEV", "VG", "VLO",
    "VSH", "VST", "WAT", "WAY", "WDAY", "WK", "WSM", "WT", "WTTR", "XOM",
    "XYZ", "ZBRA", "ZM"

]



from backtest_v2 import backtest


#if __name__ == "__main__":

"""
    res = backtest(
        tickers,
        entry="signal_close",       # or "both" to also see next_open
        exits="all",
        max_hold=20,                # fixed / breakeven keep the original 20-day time stop
        trail_max_hold=90,          # atr_trail / ema_trail may run up to 90 days
        trail_activate_r=2.0,       # trailing and break-even arm only after +2R
        trail_atr_mult=2.5,
        detail_exit="atr_trail",
        workers=4,
    )
    
    res = backtest(tickers, entry="signal_close", exits="all", max_hold=20, trail_max_hold=90,
               trail_activate_r=2.0, detail_exit="ema_trail", workers=4, control_k=10)
    print(res["summary"])
"""

if __name__ == "__main__":
    backtest(tickers, entry="signal_close", exits="all", max_hold=20, trail_max_hold=90,
             trail_activate_r=2.0, detail_exit="ema_trail", split_date="2025-07-15",
             tag="items", workers=4, conf_levels=[2, 3], variants="items")