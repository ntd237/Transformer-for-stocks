import pandas as pd

# Đọc dữ liệu từ file CSV
data = pd.read_csv('AAPL_stock_data.csv')

# Xuất dữ liệu sang file Excel
data.to_excel('AAPL_stock_data.xlsx', index=False) # index=False để không bao gồm cột index trong file Excel
