# TWSE Stock Analysis Toolkit

A comprehensive toolkit for downloading Taiwan Stock Exchange (TWSE) historical data and applying advanced technical analysis strategies for stock selection.

## 🌟 Features

### 📊 Data Crawler (`stock_crawler.py`)
- Download historical daily stock prices from TWSE
- Support for multiple stocks and custom date ranges
- ROC (Taiwan calendar) to Gregorian date conversion
- Automatic rate limiting to respect API constraints
- Export to CSV and Parquet formats
- Professional progress tracking

### 📈 Stock Selectors

#### **Strategy V1** (`stock_selector_v1.py`) - 六脉神剑 (Six-Pulse Strategy)
A 6-indicator resonance-based strategy:
- **MACD** - Moving Average Convergence Divergence
- **KD** - Stochastic Oscillator
- **RSI** - Relative Strength Index
- **LWR** - Larry Williams %R
- **BBI** - Bull Bear Index
- **MMS/MMM** - Momentum Indicators

**Signal Generation**: Buy signal triggered when all 6 indicators align (resonance)

#### **Strategy V2** (`stock_selector_v2.py`) - Advanced 10-Indicator Strategy
An enhanced version based on 小鑫课程公式 with 10 indicators:
- All 6 indicators from V1
- **DBCD** - Bias Deviation Indicator
- **HOLD** - Custom Hold Signal
- **ZLGJ** - Momentum Strength
- **Capital Flow** - Volume-based Money Flow Analysis

**Signal Generation**: Operation signal triggered when guideline score > 9 (out of 10)

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Install required packages
pip install -r requirements.txt
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/twse-stock-toolkit.git
cd twse-stock-toolkit

# Install dependencies
pip install -r requirements.txt
```

---

## 📖 Usage Guide

### 1. Download Stock Data

#### Download All Stocks from CSV
```bash
python stock_crawler.py --start 2024-01-01 --end 2024-12-31
```

#### Download Specific Stocks
```bash
python stock_crawler.py --symbols 2330,2317,2454 --start 2024-06-01 --end 2024-12-31
```

#### Test with Limited Stocks
```bash
python stock_crawler.py --limit 10 --start 2024-10-01 --end 2024-10-31
```

#### Custom Output Directory
```bash
python stock_crawler.py --output-dir ./my_data --pause 0.5
```

**Available Options:**
- `--symbols-file`: CSV file containing stock symbols (default: `STOCK_DAY_ALL_20251003.csv`)
- `--symbols`: Comma-separated stock symbols (e.g., `2330,2317`)
- `--start`: Start date in YYYY-MM-DD format (default: `2025-04-01`)
- `--end`: End date in YYYY-MM-DD format (default: `2025-10-03`)
- `--output-dir`: Output directory (default: `./data`)
- `--limit`: Limit number of stocks for testing
- `--pause`: Seconds between API requests (default: `0.2`)

---

### 2. Stock Selection - Strategy V1 (6 Indicators)

```bash
# Analyze dataset and display results
python stock_selector_v1.py data/twse_all_20240101_20241231.csv

# Save buy signals to file
python stock_selector_v1.py data/twse_all_20240101_20241231.csv --output signals_v1.csv
```

**Output Columns:**
- `stock_id`: Stock symbol
- `date`: Signal date
- `close`: Closing price
- `resonance`: All indicators aligned (True/False)
- `buy_signal`: New buy signal (True/False)
- `tj1_macd` to `tj6_momentum`: Individual indicator status

---

### 3. Stock Selection - Strategy V2 (10 Indicators)

```bash
# Analyze with advanced strategy
python stock_selector_v2.py data/twse_all_20240101_20241231.csv

# Save selected stocks
python stock_selector_v2.py data/twse_all_20240101_20241231.csv --output picks_v2.csv
```

**Output Columns:**
- `stock_id`: Stock symbol
- `date`: Signal date
- `close`: Closing price
- `guideline`: Score (0-10, sum of all indicators)
- `operation_signal`: Triggered when guideline > 9
- `a1` to `a10`: Individual indicator status

---

## 📁 Project Structure

```
twse-stock-toolkit/
├── stock_crawler.py              # TWSE data crawler
├── stock_selector_v1.py          # 6-indicator strategy
├── stock_selector_v2.py          # 10-indicator strategy
├── STOCK_DAY_ALL_20251003.csv   # Stock symbols list
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── data/                         # Output directory (auto-created)
    ├── twse_all_YYYYMMDD_YYYYMMDD.csv
    └── twse_all_YYYYMMDD_YYYYMMDD.parquet
```

---

## 🔬 Technical Indicators Explained

### Strategy V1 - Six-Pulse Indicators

| Indicator | Description           | Buy Condition    |
| --------- | --------------------- | ---------------- |
| **MACD**  | Trend momentum        | DIFF > DEA       |
| **KD**    | Stochastic oscillator | K > D            |
| **RSI**   | Relative strength     | RSI(5) > RSI(13) |
| **LWR**   | Williams %R           | LWR1 > LWR2      |
| **BBI**   | Bull/Bear balance     | Close > BBI      |
| **MOM**   | Momentum comparison   | MMS > MMM        |

### Strategy V2 - Advanced 10 Indicators

Includes all V1 indicators plus:

| Indicator | Description        | Buy Condition    |
| --------- | ------------------ | ---------------- |
| **DBCD**  | Bias deviation     | DBCD > MM        |
| **HOLD**  | Custom hold signal | HOLD > Downtrend |
| **ZLGJ**  | Momentum strength  | ZLGJ > MAZL      |
| **FLOW**  | Capital flow       | Positive inflow  |

---

## 📊 Data Format

### Input CSV Format (Stock Symbols)
```csv
證券代號
0050
2330
2317
```

### Output CSV Format
```csv
stock_id,date,open,high,low,close,volume,turnover
2330,2024-01-02,590.0,595.0,588.0,593.0,45678900,27089345000.0
```

---

## ⚙️ Configuration

### Strategy Parameters (V2)
Located in `stock_selector_v2.py`:
```python
N1 = 3   # Short-term period
N2 = 5   # Medium-short period
N3 = 9   # Medium period
N4 = 13  # Medium-long period
N5 = 21  # Long period
N6 = 34  # Extra-long period
```

### Crawler Configuration
Located in `stock_crawler.py`:
```python
REQUEST_TIMEOUT = 10      # API request timeout (seconds)
DEFAULT_PAUSE = 0.2       # Pause between requests (seconds)
```

---

## 🛠️ Development

### Running Tests
```bash
# Test crawler with 5 stocks
python stock_crawler.py --limit 5 --start 2024-10-01 --end 2024-10-31

# Test selector V1
python stock_selector_v1.py data/twse_all_20241001_20241031.csv

# Test selector V2
python stock_selector_v2.py data/twse_all_20241001_20241031.csv
```

### Common Issues

**Issue**: Rate limiting errors from TWSE API
```bash
# Solution: Increase pause time
python stock_crawler.py --pause 0.5
```

**Issue**: Missing volume column error
```bash
# Solution: Ensure you're using data from stock_crawler.py
# The selectors require volume data for capital flow analysis
```

---

## 📝 Examples

### Complete Workflow

```bash
# Step 1: Download data for Q4 2024
python stock_crawler.py \
  --start 2024-10-01 \
  --end 2024-12-31 \
  --output-dir ./Q4_2024

# Step 2: Run 6-indicator analysis
python stock_selector_v1.py \
  Q4_2024/twse_all_20241001_20241231.csv \
  --output Q4_2024/signals_v1.csv

# Step 3: Run 10-indicator analysis
python stock_selector_v2.py \
  Q4_2024/twse_all_20241001_20241231.csv \
  --output Q4_2024/signals_v2.csv
```

### Analyze Specific Stocks Only

```bash
# Download data for tech giants
python stock_crawler.py \
  --symbols 2330,2317,2454,3008 \
  --start 2024-01-01 \
  --end 2024-12-31

# Analyze with V2 strategy
python stock_selector_v2.py \
  data/twse_all_20240101_20241231.csv \
  --output tech_picks.csv
```

---

## 📚 Dependencies

- **pandas** >= 2.0.0 - Data manipulation
- **numpy** >= 1.24.0 - Numerical computing
- **requests** >= 2.31.0 - HTTP library for API calls
- **pyarrow** >= 14.0.0 - Parquet file support (optional)

---

## ⚖️ License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## ⭐ Acknowledgments

- Taiwan Stock Exchange (TWSE) for providing the data API
- 同花顺六脉神剑 strategy concept

---

## 📈 Disclaimer

**Important**: This toolkit is for educational and research purposes only. It is NOT financial advice. Always do your own research and consult with financial professionals before making investment decisions. Past performance does not guarantee future results.

---

<div align="center">

**If you find this project helpful, please give it a ⭐!**

</div>
