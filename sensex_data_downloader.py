import os
import time
import pandas as pd
from datetime import datetime, timezone, timedelta
import pyotp
import sys
import requests
import zipfile
import io
from dotenv import load_dotenv
import traceback
import logging
import concurrent.futures
from threading import Lock
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# ===============================
# CONFIG - GitHub Actions Compatible
# ===============================
API_SLEEP = 1.0
MAX_RETRIES = 3
MAX_WORKERS = 3
IST = timezone(timedelta(hours=5, minutes=30))
WEEKS_FOR_RANGE = 4
SENSEX_TOKEN = "99919000"
SENSEX_STRIKE_MULTIPLE = 100

# GitHub Secrets
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
# Angel One credentials from secrets
ANGEL_API_KEY = os.getenv("ANGEL_API_KEY")
ANGEL_CLIENT_ID = os.getenv("ANGEL_CLIENT_ID")
ANGEL_PIN = os.getenv("ANGEL_PIN")
ANGEL_TOTP_KEY = os.getenv("ANGEL_TOTP_KEY")

# Thread-safe counters and lists
success_list = []
failed_list = []
failed_details = []
zip_lock = Lock()
counter_lock = Lock()
processed_counter = 0
total_symbols = 0

# ===============================
# UTILITY FUNCTIONS
# ===============================
def round_to_nearest_100(price):
    """Round price to nearest 100"""
    return round(price / 100) * 100


def get_SENSEX_historical_data(smart_api, weeks=4):
    """
    Get SENSEX historical data for specified number of weeks
    Returns: DataFrame with OHLC data or None if fails
    """
    try:
        # Calculate date range
        to_date = datetime.now(IST)
        from_date = to_date - timedelta(weeks=weeks)
        
        params = {
            "exchange": "BSE",
            "symboltoken": SENSEX_TOKEN,
            "interval": "ONE_DAY",
            "fromdate": from_date.strftime("%Y-%m-%d 09:15"),
            "todate": to_date.strftime("%Y-%m-%d %H:%M")
        }

        logger.info(f"Fetching {weeks} weeks of SENSEX historical data...")
        resp = smart_api.getCandleData(params)
        
        if resp and resp.get("status") and resp.get("data"):
            data = resp["data"]
            df = pd.DataFrame(
                data,
                columns=["Date", "Open", "High", "Low", "Close", "Volume"]
            )
            df["Date"] = pd.to_datetime(df["Date"])
            
            # Calculate max high and min low
            max_high = df["High"].max()
            min_low = df["Low"].min()
            current_close = df["Close"].iloc[-1] if len(df) > 0 else None
            
            logger.info(f"SENSEX {weeks}-week range: Low={min_low:.2f}, High={max_high:.2f}, Current={current_close:.2f}")
            return {
                "df": df,
                "max_high": max_high,
                "min_low": min_low,
                "current_close": current_close
            }
        else:
            logger.error(f"No historical data returned: {resp}")
            
    except Exception as e:
        logger.error(f"SENSEX historical data error: {e}")
        traceback.print_exc()

    return None


def get_SENSEX_ltp(smart_api):
    """Get current SENSEX LTP"""
    try:
        params = {
            "exchange": "BSE",
            "symboltoken": SENSEX_TOKEN,
            "interval": "ONE_MINUTE",
            "fromdate": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d 09:15"),
            "todate": datetime.now().strftime("%Y-%m-%d %H:%M")
        }

        resp = smart_api.getCandleData(params)
        if resp and resp.get("status") and resp.get("data"):
            return resp["data"][-1][4]
    except Exception as e:
        logger.error(f"SENSEX LTP error: {e}")

    return None


def calculate_strike_range(smart_api, buffer=1000):
    """
    Calculate strike range based on 4-week high/low
    Returns: (start_strike, end_strike)
    """
    hist_data = get_SENSEX_historical_data(smart_api, weeks=WEEKS_FOR_RANGE)
    
    if hist_data:
        min_low = hist_data["min_low"]
        max_high = hist_data["max_high"]
        
        # Calculate range with buffer and round to nearest 100
        start_strike = round_to_nearest_100(min_low - buffer)
        end_strike = round_to_nearest_100(max_high + buffer)
        
        # Ensure start_strike is not negative
        start_strike = max(0, start_strike)
        
        logger.info(f"""
📊 Strike Range Calculation:
  4-week Low: {min_low:.2f}
  4-week High: {max_high:.2f}
  Buffer: ±{buffer}
  Start Strike: {start_strike}
  End Strike: {end_strike}
  Range Width: {end_strike - start_strike} points
  Strike Multiple: {SENSEX_STRIKE_MULTIPLE}
""")
        
        return start_strike, end_strike
    
    # Fallback: Use current LTP with fixed range if historical fails
    logger.warning("Using fallback LTP-based range calculation")
    SENSEX_ltp = get_SENSEX_ltp(smart_api)
    if SENSEX_ltp is None:
        logger.error("Cannot get SENSEX LTP, using default range")
        return 60000, 75000  # Default range
    
    rounded = round_to_nearest_100(SENSEX_ltp)
    start_strike = rounded - 3500
    end_strike = rounded + 3500
    
    logger.info(f"""
📊 Fallback Strike Range:
  Current LTP: {SENSEX_ltp:.2f}
  Rounded: {rounded}
  Start Strike: {start_strike}
  End Strike: {end_strike}
""")
    
    return start_strike, end_strike


def load_symbol_master():
    """Load BFO symbol master file"""
    BFO_URL = "https://api.shoonya.com/BFO_symbols.txt.zip"

    try:
        logger.info("Downloading symbol master...")
        r = requests.get(BFO_URL, timeout=60)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            with z.open(z.namelist()[0]) as f:
                content = f.read().decode("utf-8")
                content = "\n".join(line.rstrip(",") for line in content.splitlines())
                df = pd.read_csv(io.StringIO(content))
                logger.info(f"Symbol master loaded: {len(df)} records")
                return df
    except Exception as e:
        logger.error(f"Symbol master load failed: {e}")
        return None


def is_today_SENSEX_expiry(df_master):
    """Check if today is SENSEX expiry day"""
    today_ist = datetime.now(IST).date()

    df = df_master[
        (df_master["Symbol"] == "BSXOPT") &
        (df_master["Instrument"] == "OPTIDX")
    ].copy()

    if df.empty:
        return False, None

    df["ExpiryDate"] = pd.to_datetime(
        df["Expiry"], format="%d-%b-%Y", errors="coerce"
    ).dt.date

    expiry_dates = df["ExpiryDate"].dropna().unique()

    if today_ist in expiry_dates:
        return True, today_ist

    return False, None


def get_option_symbols(df_master, expiry_date, start_strike, end_strike):
    """
    Get ALL option symbols within strike range
    SENSEX strikes are in multiples of 100
    """
    expiry_str = expiry_date.strftime("%d-%b-%Y").upper()

    df = df_master[
        (df_master["Symbol"] == "BSXOPT") &
        (df_master["Instrument"] == "OPTIDX") &
        (df_master["Expiry"] == expiry_str)
    ].copy()

    if df.empty:
        return pd.DataFrame()

    df["StrikePrice"] = pd.to_numeric(df["StrikePrice"], errors="coerce")

    # Filter by strike range and ensure multiples of 100
    df = df[
        (df["StrikePrice"] >= start_strike) &
        (df["StrikePrice"] <= end_strike) &
        (df["StrikePrice"] % SENSEX_STRIKE_MULTIPLE == 0)
    ]

    logger.info(f"Found {len(df)} option symbols between {start_strike} and {end_strike}")
    logger.info(f"Strikes are in multiples of {SENSEX_STRIKE_MULTIPLE}")
    
    return df.sort_values(["StrikePrice", "OptionType"])


def send_zip_to_telegram(zip_bytes, zip_name, max_retries=5):
    """
    Send zip to Telegram
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials not set, skipping Telegram upload")
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument"
    
    try:
        files = {
            "document": (zip_name, zip_bytes, "application/zip")
        }
        data = {"chat_id": TELEGRAM_CHAT_ID}

        logger.info(f"Sending {zip_name} to Telegram...")
        r = requests.post(url, data=data, files=files, timeout=(30, 600))
        r.raise_for_status()
        
        logger.info("Telegram upload successful")
        return True

    except Exception as e:
        logger.error(f"Telegram upload failed: {e}")
        return False


def create_excel_in_memory(df):
    """Create Excel file in memory"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    output.seek(0)
    return output.read()


def get_candles_with_retry(smart, params, retries=MAX_RETRIES):
    """Get candle data with retry logic"""
    for attempt in range(retries):
        try:
            response = smart.getCandleData(params)
            
            if response and response.get("status"):
                return response
            elif response and response.get("errorcode") == "AB1004":
                wait_time = (attempt + 1) * 10
                logger.warning(f"Server error AB1004, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.warning(f"API returned error: {response}")
                break
                
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on attempt {attempt + 1}, retrying...")
            time.sleep((attempt + 1) * 5)
        except Exception as e:
            logger.error(f"Error on attempt {attempt + 1}: {e}")
            if attempt == retries - 1:
                raise
            time.sleep((attempt + 1) * 5)
    
    return None


def download_symbol_data(args):
    """Worker function to download data for a single symbol"""
    global processed_counter
    
    smart, row, from_date, to_date = args
    symbol = row["TradingSymbol"]
    token = str(row["Token"])
    
    with counter_lock:
        processed_counter += 1
        current_count = processed_counter
    
    logger.info(f"[{current_count}/{total_symbols}] Processing {symbol}...")
    
    try:
        params = {
            "exchange": "BFO",
            "symboltoken": token,
            "interval": "ONE_MINUTE",
            "fromdate": from_date,
            "todate": to_date
        }
        
        response = get_candles_with_retry(smart, params)
        
        if response and response.get("status") and response.get("data"):
            data = response["data"]
            
            if data:
                ohlc = pd.DataFrame(
                    data,
                    columns=["Date", "Open", "High", "Low", "Close", "Volume"]
                )
                ohlc["Date"] = pd.to_datetime(ohlc["Date"]).dt.tz_localize(None)
                ohlc.drop_duplicates(subset=["Date"], inplace=True)
                
                excel_data = create_excel_in_memory(ohlc)
                
                result = {
                    "symbol": symbol,
                    "data": excel_data,
                    "candle_count": len(ohlc),
                    "success": True
                }
                
                logger.info(f"[{current_count}/{total_symbols}] {symbol}: ✅ {len(ohlc)} candles")
                return result
            else:
                logger.warning(f"[{current_count}/{total_symbols}] {symbol}: ⚠️ No data")
                return {
                    "symbol": symbol,
                    "error": "No data",
                    "success": False
                }
        else:
            error_msg = response.get("message", "Unknown") if response else "No response"
            logger.error(f"[{current_count}/{total_symbols}] {symbol}: ❌ {error_msg}")
            return {
                "symbol": symbol,
                "error": error_msg,
                "success": False
            }
            
    except Exception as e:
        error_msg = str(e)[:100]
        logger.error(f"[{current_count}/{total_symbols}] {symbol}: ❌ Error: {error_msg}")
        return {
            "symbol": symbol,
            "error": error_msg,
            "success": False
        }


def download_data_concurrent(smart, df, from_date, to_date, zip_buffer):
    """Download data using concurrent workers"""
    global total_symbols, success_list, failed_list, failed_details
    
    total_symbols = len(df)
    
    # Prepare arguments for workers
    args_list = []
    for _, row in df.iterrows():
        args_list.append((smart, row, from_date, to_date))
    
    logger.info(f"Starting download with {MAX_WORKERS} concurrent workers...")
    logger.info(f"Total symbols to download: {total_symbols}")
    
    # Use ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_symbol = {
            executor.submit(download_symbol_data, args): args[1]["TradingSymbol"] 
            for args in args_list
        }
        
        # Process completed tasks as they finish
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            
            try:
                result = future.result()
                
                if result["success"]:
                    with zip_lock:
                        with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED) as zf:
                            zf.writestr(f"{result['symbol']}.xlsx", result["data"])
                    success_list.append(result["symbol"])
                else:
                    failed_list.append(result["symbol"])
                    failed_details.append((result["symbol"], result["error"]))
                    
            except Exception as e:
                error_msg = str(e)[:100]
                logger.error(f"Exception for {symbol}: {error_msg}")
                failed_list.append(symbol)
                failed_details.append((symbol, error_msg))


# ===============================
# GITHUB ACTIONS SPECIFIC FUNCTIONS
# ===============================
def setup_angel_api():
    """Dynamically import and setup Angel One API"""
    try:
        # Try to install smartapi if not available
        import subprocess
        import importlib
        
        try:
            from SmartApi.smartConnect import SmartConnect
            logger.info("SmartAPI already installed")
        except ImportError:
            logger.info("Installing smartapi...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "smartapi-python"])
            from SmartApi.smartConnect import SmartConnect
        
        return SmartConnect
    except Exception as e:
        logger.error(f"Failed to setup Angel API: {e}")
        return None


def save_to_github_artifacts(data, filename):
    """Save data as artifact for GitHub Actions"""
    try:
        os.makedirs('artifacts', exist_ok=True)
        filepath = f"artifacts/{filename}"
        
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=False)
        elif isinstance(data, bytes):
            with open(filepath, 'wb') as f:
                f.write(data)
        elif isinstance(data, dict):
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        logger.info(f"Saved artifact: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save artifact {filename}: {e}")
        return None


# ===============================
# MAIN EXECUTION
# ===============================
def main():
    global total_symbols
    
    print("="*60)
    print("SENSEX EXPIRY DATA DOWNLOADER - GITHUB ACTIONS")
    print(f"Date: {datetime.now(IST).strftime('%d-%b-%Y %H:%M')}")
    print("="*60)
    
    # Setup Angel One API
    logger.info("Setting up Angel One API...")
    SmartConnect = setup_angel_api()
    if SmartConnect is None:
        raise Exception("Failed to setup Angel One API")
    
    # Login
    logger.info("Logging in to Angel One...")
    
    if not all([ANGEL_API_KEY, ANGEL_CLIENT_ID, ANGEL_PIN, ANGEL_TOTP_KEY]):
        raise Exception("Angel One credentials not set in GitHub Secrets")
    
    smart = SmartConnect(api_key=ANGEL_API_KEY)
    totp = pyotp.TOTP(ANGEL_TOTP_KEY).now()
    login = smart.generateSession(ANGEL_CLIENT_ID, ANGEL_PIN, totp)

    if not login or not login.get("status"):
        raise Exception("Login failed")
    logger.info("Login successful")
    
    # Load symbol master
    logger.info("Loading BFO symbol master...")
    df_master = load_symbol_master()
    if df_master is None:
        raise Exception("Cannot proceed without symbol master")
    
    # Check expiry
    is_expiry, expiry_date = is_today_SENSEX_expiry(df_master)
    if not is_expiry:
        logger.info("Today is NOT SENSEX expiry day. Exiting.")
        save_to_github_artifacts(
            {"status": "no_expiry", "date": datetime.now(IST).strftime('%Y-%m-%d')},
            "no_expiry.json"
        )
        return
    
    logger.info(f"TODAY IS SENSEX EXPIRY: {expiry_date.strftime('%d-%b-%Y')}")
    
    # Calculate strike range
    logger.info("Calculating strike range...")
    start_strike, end_strike = calculate_strike_range(smart, buffer=1000)
    
    # Get option symbols
    logger.info("Finding option symbols...")
    df = get_option_symbols(df_master, expiry_date, start_strike, end_strike)
    if df.empty:
        raise Exception("No option symbols found")
    
    total_symbols = len(df)
    
    # Save symbol list
    save_to_github_artifacts(
        df[["TradingSymbol", "StrikePrice", "OptionType"]],
        f"symbols_{expiry_date.strftime('%d%m%y')}.csv"
    )
    
    # Set date range
    FROM_DATE = (expiry_date - timedelta(days=90)).strftime("%Y-%m-%d 09:15")
    TO_DATE = expiry_date.strftime("%Y-%m-%d 15:30")
    logger.info(f"Date range: {FROM_DATE} to {TO_DATE}")
    
    # Prepare zip buffer
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        pass
    
    # Download data
    logger.info(f"Downloading {total_symbols} option symbols...")
    start_time = time.time()
    
    download_data_concurrent(smart, df, FROM_DATE, TO_DATE, zip_buffer)
    
    download_time = time.time() - start_time
    
    # Save failed symbols if any
    if failed_details:
        failed_df = pd.DataFrame(failed_details, columns=["Symbol", "Error"])
        save_to_github_artifacts(failed_df, f"failed_{expiry_date.strftime('%d%m%y')}.csv")
    
    # Save zip file
    if success_list:
        zip_buffer.seek(0)
        zip_data = zip_buffer.read()

        # Get current time
        current_time = datetime.now()

        # Format it: %d (day), %m (month), %y (year), %H (hours), %M (minutes)
        time_str = current_time.strftime("%d%m%y_%H:%M")

        zip_name = f"SENSEX_expiry_{expiry_date.strftime('%d%m%y')}_1min_{time_str}.zip"
        
        # Save locally first
        zip_path = save_to_github_artifacts(zip_data, zip_name)
        
        # Send to Telegram if configured
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            telegram_sent = send_zip_to_telegram(zip_data, zip_name)
        else:
            telegram_sent = False
            logger.info("Telegram not configured, skipping upload")
    else:
        telegram_sent = False
        logger.warning("No data downloaded")
    
    # Save summary
    summary = {
        "expiry_date": expiry_date.strftime('%d-%b-%Y'),
        "strike_range_start": start_strike,
        "strike_range_end": end_strike,
        "total_symbols": total_symbols,
        "success_count": len(success_list),
        "failed_count": len(failed_list),
        "success_rate": (len(success_list)/total_symbols)*100 if total_symbols > 0 else 0,
        "download_time_seconds": download_time,
        "telegram_sent": telegram_sent,
        "workers_used": MAX_WORKERS,
        "timestamp": datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')
    }
    
    save_to_github_artifacts(summary, f"summary_{expiry_date.strftime('%d%m%y')}.json")
    
    # Print summary
    logger.info("="*60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("="*60)
    for key, value in summary.items():
        logger.info(f"{key}: {value}")
    logger.info("="*60)


if __name__ == "__main__":
    try:
        main()
        logger.info("Script completed successfully")
    except KeyboardInterrupt:
        logger.warning("Script interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)
