"""快速检查数据目录中的文件日期范围和内容"""
import os
import sys
import glob
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_DIR
data_dir = sys.argv[1] if len(sys.argv) > 1 else DATA_DIR

files = sorted(glob.glob(os.path.join(data_dir, "daily_summary_*.csv")))
print(f"Directory: {data_dir}")
print(f"Total files: {len(files)}")

if not files:
    print("No daily_summary_*.csv files found!")
    sys.exit(1)

# Date range
dates = []
for f in files:
    d = os.path.basename(f).replace("daily_summary_", "").replace(".csv", "")
    try:
        dates.append(pd.to_datetime(d))
    except Exception:
        pass

dates = sorted(dates)
print(f"Date range: {dates[0].date()} ~ {dates[-1].date()}")
print(f"Total dates: {len(dates)}")

# Year breakdown
by_year = {}
for d in dates:
    by_year.setdefault(d.year, []).append(d)
for y in sorted(by_year):
    print(f"  {y}: {len(by_year[y])} trading days ({by_year[y][0].date()} ~ {by_year[y][-1].date()})")

# Peek at first file
print(f"\nPeeking at: {os.path.basename(files[0])}")
df = pd.read_csv(files[0], nrows=10)
print(f"  Columns: {list(df.columns)}")
print(f"  StockID sample: {df['StockID'].unique()[:5] if 'StockID' in df.columns else 'N/A'}")

# Check total rows in first file
df_full = pd.read_csv(files[0])
stock_counts = df_full.groupby("StockID").size()
print(f"  Stocks in file: {len(stock_counts)}")
print(f"  Rows per stock: {stock_counts.value_counts().to_dict()}")

# Check ChiNext50 overlap
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.chinext50 import get_chinext50_constituents
chinext = set(get_chinext50_constituents())
file_stocks = set()
for sid in df_full["StockID"].unique():
    if isinstance(sid, float):
        sid = int(sid)
    normalized = f"{int(sid):06d}" if isinstance(sid, int) else str(sid).strip()
    file_stocks.add(normalized)

overlap = file_stocks & chinext
print(f"\n  ChiNext50 list size: {len(chinext)}")
print(f"  Stocks in file (normalized): {len(file_stocks)}")
print(f"  Overlap with ChiNext50: {len(overlap)}")
if overlap:
    print(f"  Sample overlap: {list(overlap)[:10]}")
else:
    print(f"  File stock samples: {list(file_stocks)[:10]}")
    print(f"  ChiNext50 samples: {list(chinext)[:10]}")
