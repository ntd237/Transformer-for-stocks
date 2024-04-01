import yfinance as yf

# Tải dữ liệu cổ phiếu của Apple từ 01/01/2020 đến 31/12/2020
stock_data = yf.download('AAPL', start='2020-01-01', end='2020-12-31')

# Lưu dữ liệu vào file CSV
stock_data.to_csv('AAPL_stock_data.csv')
