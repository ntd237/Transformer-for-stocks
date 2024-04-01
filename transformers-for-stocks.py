#Cài đặt thư viện và import các module cần thiết:
from yahoofinancials import YahooFinancials
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Flatten
from tensorflow.keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import sqlite3
import pandas as pd
import math
import matplotlib.pyplot as plt
from datetime import datetime

#Cài đặt định dạng và thiết lập môi trường làm việc:
plt.style.use('default')


#Thiết lập ngẫu nhiên và tái sử dụng mô hình:
from numpy.random import seed
seed(1)
tf.random.set_seed(2)


#Định nghĩa các hàm hỗ trợ:
def plot_predictions(test,predicted,symbol):
    plt.plot(test, color='red',label=f'Real {symbol} Stock Price')
    plt.plot(predicted, color='blue',label=f'Predicted {symbol} Stock Price')
    plt.title(f'{symbol} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel(f'{symbol} Stock Price')
    plt.legend()
    plt.show()

def plot_return_predictions(test,predicted,symbol):
    plt.plot(test, color='red',label=f'Real {symbol} Stock Price Returns')
    plt.plot(predicted, color='blue',label=f'Predicted {symbol} Stock Price Return')
    plt.title(f'{symbol} Stock Return Prediction')
    plt.xlabel('Time')
    plt.ylabel(f'{symbol} Stock Price Returns')
    plt.legend()
    plt.show()
    
def return_rmse(test,predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {}.".format(rmse))
    return rmse

def get_ticker_data(ticker: str, param_start_date, param_end_date) -> dict:
    raw_data = YahooFinancials(ticker)
    return raw_data.get_historical_price_data(param_start_date, param_end_date, "daily").copy()

def fetch_ticker_data(ticker: str, start_date, end_date) -> pd.DataFrame:
    date_range = pd.bdate_range(start=start_date, end=end_date)
    values = pd.DataFrame({'Date': date_range})
    values['Date'] = pd.to_datetime(values['Date'])
    raw_data = get_ticker_data(ticker, start_date, end_date)
    return pd.DataFrame(raw_data[ticker]["prices"])[['date', 'open', 'high', 'low', 'adjclose', 'volume']]

def shift(xs, n):
    e = np.empty_like(xs)
    if n >= 0:
        e[:n] = np.nan
        e[n:] = xs[:-n]
    else:
        e[n:] = np.nan
        e[:n] = xs[-n:]
    return e

#Import thư viện và module cần thiết: Dòng đầu tiên import class datetime từ module datetime.
from datetime import datetime


#Chọn biểu tượng cổ phiếu và khoảng thời gian: 
symbol_to_fetch = 'IBM'
start_date = '2020-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')


# Lấy dữ liệu giá cổ phiếu: 
stock = fetch_ticker_data(symbol_to_fetch, start_date, end_date)


#Chỉnh sửa và xử lý dữ liệu:
stock.columns = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']
stock['DateTime'] = stock['DateTime'].apply(lambda x: datetime.fromtimestamp(x))
stock = stock.ffill(axis=0) 
stock = stock.bfill(axis=0) 
stock = stock.set_index('DateTime')
stock['Symbol'] = symbol_to_fetch
stock.tail()


#Lưu bản sao của DataFrame gốc: 
original_stock = stock
original_symbol = symbol_to_fetch


#In và hiển thị dữ liệu cuối cùng của cột giá đóng cửa (Close): 
#In ra các dòng cuối cùng của cột Close để kiểm tra dữ liệu cuối cùng.
stock['Close'].tail()

from datetime import datetime 


#Chọn biểu tượng cổ phiếu và khoảng thời gian: 
symbol_to_fetch = 'IBM'
start_date = str(datetime(2017, 1, 1).date())
end_date = str(datetime(2021, 2, 18).date())

#Chọn mục tiêu dự đoán: 
target = 'Close' 


#Chọn tập huấn luyện và tập kiểm tra:
train_start_date = start_date
train_end_date = '2021-10-31'
test_start_date = '2021-11-01'
training_set = stock[train_start_date:train_end_date].iloc[:,3:4].values 
test_set = stock[test_start_date:].iloc[:,3:4].values


#Tính toán tỷ suất lợi nhuận (return) của tập kiểm tra:
test_set_return = stock[test_start_date:].iloc[:,3:4].pct_change().values


#In kích thước của tập huấn luyện và tập kiểm tra:
print(training_set.shape)
print(test_set.shape)

#Vẽ biểu đồ giá cổ phiếu của công ty trong tập huấn luyện và tập kiểm tra.
stock[target][train_start_date:train_end_date].plot(figsize=(16,4),legend=True)
stock[target][test_start_date:].plot(figsize=(16,4),legend=True)
plt.legend([f'Training set (Before {train_end_date})',f'Test set ({test_start_date} and beyond)'])
plt.title(f'{symbol_to_fetch} stock price')
plt.show()

#Chia tỷ lệ dữ liệu huấn luyện để đảm bảo rằng các giá trị đều nằm trong khoảng từ 0 đến 1.
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

#Chuẩn bị dữ liệu huấn luyện để sử dụng cho mô hình dự đoán.
timesteps = 8
x_train = []
y_train = []
for i in range(timesteps,training_set.shape[0]):
    x_train.append(training_set_scaled[i-timesteps:i,0])
    y_train.append(training_set_scaled[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

print(x_train[0], y_train[0])
print(x_train[1], y_train[1])

#Chuyển đổi hình dạng của `x_train` để phù hợp với yêu cầu đầu vào của mạng nơ-ron.
print(x_train.shape, y_train.shape)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape, y_train.shape)

#Xáo trộn ngẫu nhiên các mẫu trong tập huấn luyện `x_train` và `y_train`.
print(x_train.shape, y_train.shape)
idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]

#hàm transformer_encoder thực hiện một phần của quá trình mã hóa trong mô hình Transformer, bao gồm chuẩn hóa, attention, và feed-forward neural network.
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs
    

    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation = "relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

#Hàm build_model trong đoạn mã trên được sử dụng để xây dựng một mô hình mạng nơ-ron sử dụng kiến trúc Transformer.
def build_model(
    input_shape,             
    head_size,               
    num_heads,               
    ff_dim,                  
    num_transformer_blocks,  
    mlp_units,               
                             
    dropout=0,               
    mlp_dropout=0,
):
    
    
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    for _ in range(num_transformer_blocks):  
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="elu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1, activation="linear")(x) 
    return keras.Model(inputs, outputs)

def lr_scheduler(epoch, lr, warmup_epochs=30, decay_epochs=100, initial_lr=1e-6, base_lr=1e-3, min_lr=5e-5):
    if epoch <= warmup_epochs:
        pct = epoch / warmup_epochs
        return ((base_lr - initial_lr) * pct) + initial_lr

    if epoch > warmup_epochs and epoch < warmup_epochs+decay_epochs:
        pct = 1 - ((epoch - warmup_epochs) / decay_epochs)
        return ((base_lr - min_lr) * pct) + min_lr

    return min_lr


callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.LearningRateScheduler(lr_scheduler)
            ]


input_shape = x_train.shape[1:]
print(input_shape)



model = build_model(
    input_shape,
    head_size=46, 
    num_heads=60, 
    ff_dim=55, 
    num_transformer_blocks=5,
    mlp_units=[256],
    mlp_dropout=0.4,
    dropout=0.14,
)

model.compile(
    loss="mean_squared_error",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["mean_squared_error"],
)

history = model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=20,
    callbacks=callbacks,
)



dataset_total = pd.concat((original_stock[target][:train_end_date],original_stock[target][test_start_date:]),axis=0)
inputs = dataset_total[len(dataset_total)-len(test_set) - timesteps:].values
inputs = inputs.reshape(-1,1)
inputs  = sc.fit_transform(inputs)

X_test = []
for i in range(timesteps,test_set.shape[0] + timesteps):
    X_test.append(inputs[i-timesteps:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


print(test_set[-3],test_set[-2], test_set[-1])
shifted_test_set = shift(test_set, 1) 
print(shifted_test_set[-3],shifted_test_set[-2], shifted_test_set[-1])

print(predicted_stock_price[-1])
prediction_error = test_set - predicted_stock_price 
predicted_return = (shifted_test_set - predicted_stock_price) / shifted_test_set

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'], loc='upper left')
plt.show()

plot_predictions(test_set,predicted_stock_price,original_symbol)
return_rmse(test_set,predicted_stock_price)

plot_return_predictions(test_set_return,predicted_return,original_symbol)
return_rmse(test_set_return[1:], predicted_return[1:])