import os, datetime, pytz

TZ = pytz.timezone(os.getenv("TZ", "Asia/Kolkata"))

def is_market_open_now() -> bool:
    now = datetime.datetime.now(TZ)
    if now.weekday() >= 5:
        return False
    start = now.replace(hour=9, minute=15, second=0, microsecond=0)
    end = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return start <= now <= end

def choose_order_mode(price_type: str):
    open_now = is_market_open_now()
    price_type = (price_type or "MARKET").upper()
    if open_now:
        return ("regular", "MARKET" if price_type == "MARKET" else "LIMIT")
    else:
        return ("amo", "LIMIT")
