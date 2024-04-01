# Dự đoán Giá Cổ Phiếu bằng Mô hình Transformer

## Tổng quan
Repository này chứa mã nguồn để dự đoán giá cổ phiếu bằng mô hình Transformer. 
Mô hình Transformer, ban đầu được giới thiệu trong lĩnh vực xử lý ngôn ngữ tự nhiên, đã cho thấy kết quả hứa hẹn trong các nhiệm vụ dựa trên chuỗi do cơ chế chú ý của nó. 
Trong dự án này, chúng ta khám phá ứng dụng của nó trong dự báo giá cổ phiếu.

## Cài đặt
Để chạy mã nguồn trong repository này, bạn cần có Python cài đặt trên hệ thống của mình cùng với các thư viện phụ thuộc sau:
- yahoofinancials
- numpy
- tensorflow
- sklearn
- pandas
- matplotlib
Bạn có thể cài đặt các thư viện này bằng pip:
```bash
pip install -r requirements.txt
```

## Sử dụng
1. Lấy Dữ liệu:
   - Hàm `fetch_ticker_data` lấy dữ liệu lịch sử giá cổ phiếu từ Yahoo Finance bằng thư viện `yahoofinancials`.
2. Tiền xử lý dữ liệu:
   - Dữ liệu được lấy được tiền xử lý để xử lý các giá trị thiếu và chuẩn hóa giá cổ phiếu bằng cách sử dụng MinMaxScaler.
3. Xây dựng Mô hình:
   - Chúng tôi triển khai một mô hình Transformer tùy chỉnh cho dự đoán giá cổ phiếu bằng TensorFlow và Keras. 
     Kiến trúc mô hình bao gồm các lớp chú ý tự hồi quy đa đầu cuối cùng là các mạng nơ-ron tiếp theo.
4. Huấn luyện:
   - Mô hình được huấn luyện trên dữ liệu đã được tiền xử lý bằng cách sử dụng tập huấn luyện. 
     Chúng tôi sử dụng một lịch trình tốc độ học tập tùy chỉnh và gọi lại dừng sớm trong quá trình huấn luyện.
5. Đánh giá:
   - Sau quá trình huấn luyện, mô hình được đánh giá trên tập kiểm tra và các dự đoán được so sánh với giá cổ phiếu thực tế. 
     Các chỉ số đánh giá như sai số trung bình bình phương gốc (RMSE) được tính để đánh giá hiệu suất của mô hình.
6. Trực quan:
   - Matplotlib được sử dụng để trực quan hóa mất mát huấn luyện và giá cổ phiếu dự đoán so với giá thực tế.

## Cấu trúc Mã Nguồn
- `stock_price_prediction.ipynb`: Sổ ghi chú Jupyter chứa mã nguồn đầy đủ cho việc lấy dữ liệu, tiền xử lý, xây dựng mô hình, huấn luyện và đánh giá.
- `README.txt`: Tập tin này cung cấp một cái nhìn tổng quan về dự án, hướng dẫn cài đặt, hướng dẫn sử dụng và cấu trúc mã nguồn.
- `requirements.txt`: Tập tin văn bản liệt kê tất cả các phụ thuộc cần thiết cho dự án.

## Ví dụ Sử dụng
Để chạy mô hình dự đoán giá cổ phiếu:
1. Sao chép repository này vào máy cục bộ của bạn.
2. Di chuyển đến thư mục repository.
3. Cài đặt các phụ thuộc bằng `pip install -r requirements.txt`.
4. Mở và thực thi sổ ghi chú `stock_price_prediction.ipynb` b