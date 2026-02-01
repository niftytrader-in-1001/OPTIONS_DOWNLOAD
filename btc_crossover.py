import requests
import time
import json
import os
from datetime import datetime, timedelta, timezone

base_url = 'https://api.india.delta.exchange'

# =========================================================
# TELEGRAM CONFIG
# =========================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials not set. Skipping alert.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }

    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print("Telegram send error:", e)

# =========================================================
# TIME & EXPIRY LOGIC
# =========================================================

def get_three_expiry_dates_utc_noon_cutoff():
    now_utc = datetime.now(timezone.utc)

    if now_utc.hour >= 12:
        base_date = now_utc.date() + timedelta(days=1)
    else:
        base_date = now_utc.date()

    return [
        (base_date + timedelta(days=i)).strftime("%d-%m-%y")
        for i in range(3)
    ]

# =========================================================
# PRICE
# =========================================================

def get_btcusd_price():
    url = f'{base_url}/v2/tickers/BTCUSD'
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get('success'):
            return float(data['result']['spot_price'])
    except Exception as e:
        print("Price fetch error:", e)
    return None

# =========================================================
# STRIKE
# =========================================================

def calculate_strike_prices(price, interval=200):
    floor = int(price // interval) * interval
    return floor, floor + interval

# =========================================================
# OPTIONS
# =========================================================

def get_available_options(underlying, expiry_date):
    url = f'{base_url}/v2/tickers'
    params = {
        'contract_types': 'call_options,put_options',
        'underlying_asset_symbols': underlying,
        'expiry_date': expiry_date
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get('success'):
            return data['result']
    except Exception as e:
        print(f"Options fetch error ({expiry_date}):", e)

    return []

def find_closest_strikes(options, target_floor):
    calls, puts = {}, {}

    for opt in options:
        strike = float(opt.get('strike_price', 0))
        if opt['contract_type'] == 'call_options':
            calls[strike] = opt['symbol']
        elif opt['contract_type'] == 'put_options':
            puts[strike] = opt['symbol']

    strikes = sorted(calls.keys())
    floor = max([s for s in strikes if s <= target_floor], default=None)
    ceil = min([s for s in strikes if s > target_floor], default=None)

    return {
        'floor_ce': calls.get(floor),
        'floor_pe': puts.get(floor),
        'ceiling_ce': calls.get(ceil),
        'ceiling_pe': puts.get(ceil)
    }

# =========================================================
# OHLC
# =========================================================

def get_ohlc_data(symbol, resolution='30m', days_back=30):
    url = f'{base_url}/v2/history/candles'
    end = int(time.time())
    start = end - days_back * 86400

    params = {
        'symbol': symbol,
        'resolution': resolution,
        'start': start,
        'end': end
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get('success'):
            return data['result']
    except Exception as e:
        print(f"OHLC error ({symbol}):", e)

    return []

# =========================================================
# EMA INDICATOR (TradingView-style)
# =========================================================

def calculate_ema(values, period):
    k = 2 / (period + 1)
    ema = [values[0]]
    for price in values[1:]:
        ema.append(price * k + ema[-1] * (1 - k))
    return ema

def check_latest_ema_cross(candles):
    if len(candles) < 20:
        return None

    closes = [c['close'] for c in candles]
    ema9 = calculate_ema(closes, 9)
    ema15 = calculate_ema(closes, 15)

    prev9, curr9 = ema9[-2], ema9[-1]
    prev15, curr15 = ema15[-2], ema15[-1]

    if prev9 < prev15 and curr9 > curr15:
        return "BULLISH"
    elif prev9 > prev15 and curr9 < curr15:
        return "BEARISH"

    return None

# =========================================================
# MAIN
# =========================================================

def main():
    print("\nBTC OPTIONS EMA 9/15 SCANNER")
    print("=" * 60)

    price = get_btcusd_price()
    if not price:
        return
    
    print(f"\nBTC Spot Price: ${price:,.2f}")

    floor, _ = calculate_strike_prices(price)
    
    expiry_dates = get_three_expiry_dates_utc_noon_cutoff()
    print(f"\nExpiry Dates: {expiry_dates}")

    for expiry in expiry_dates:
        print(f"\n--- EXPIRY {expiry} ---")
        options = get_available_options('BTC', expiry)
        if not options:
            continue

        symbols = find_closest_strikes(options, floor)

        for symbol in symbols.values():
            if not symbol:
                continue

            candles = get_ohlc_data(symbol)
            cross = check_latest_ema_cross(candles)

            if cross == "BULLISH":
                msg = (
                    f"<b>EMA 9/15 Crossover Alert</b>\n"
                    f"Symbol: <b>{symbol}</b>\n"
                    f"Expiry: {expiry}\n"
                    f"Signal: <b>BULLISH</b>\n"
                    f"Time (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
                )
                print(f"{symbol} → ✅ Bullish EMA Cross")
                send_telegram_message(msg)

            elif cross == "BEARISH":
                msg = (
                    f"<b>EMA 9/15 Crossover Alert</b>\n"
                    f"Symbol: <b>{symbol}</b>\n"
                    f"Expiry: {expiry}\n"
                    f"Signal: <b>BEARISH</b>\n"
                    f"Time (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
                )
                print(f"{symbol} → 🔻 Bearish EMA Cross")
                send_telegram_message(msg)

            elif cross is None:
                # distinguish no-cross vs no-data
                if len(candles) < 20:
                    print(f"{symbol} → Not enough data")
                else:
                    print(f"{symbol} → ❌ No EMA Cross")

            time.sleep(0.4)


    print("\nScan completed.")
# =========================================================

if __name__ == "__main__":
    main()
