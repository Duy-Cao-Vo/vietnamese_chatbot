# Vietnamese Clothing Store Chatbot

Chatbot thông minh cho cửa hàng quần áo tại Việt Nam, hỗ trợ tư vấn mua hàng, chính sách, thông tin sản phẩm, kiểm tra tồn kho và vị trí cửa hàng.

## Tính năng

- Tư vấn về chính sách đổi trả, bảo hành
- Tư vấn về sản phẩm (cách mặc, màu sắc, kích cỡ)
- Kiểm tra tồn kho sản phẩm
- Thông tin về các cửa hàng
- Hỗ trợ đặt hàng và xuất hóa đơn

## Kiến trúc hệ thống

- **Mô hình ngôn ngữ**: DeepSeek R1 chạy cục bộ trên MacBook
- **Vector Database**: Chroma DB để lưu trữ dữ liệu vector cho RAG
- **Backend**: FastAPI (Python)
- **Frontend**: Streamlit cho giao diện chat đơn giản

## Cài đặt

1. Cài đặt các thư viện Python:

```bash
pip install -r requirements.txt
```

2. Tải mô hình DeepSeek R1:

```bash
python scripts/download_model.py
```

3. Khởi động ứng dụng:

```bash
python app.py
```


# Option 2
Chạy 2 app độc lập 2 terminal
Terminal 1
```
uvicorn app:app --host 0.0.0.0 --port 8000
```

Terminal 2
```
streamlit run ui/streamlit_app.py
```

## Cấu trúc dự án

```
vietnamese-clothing-chatbot/
├── app.py                     # Ứng dụng chính
├── config.py                  # Cấu hình
├── README.md                  # Tài liệu dự án
├── requirements.txt           # Các thư viện cần thiết
├── data/                      # Dữ liệu của hệ thống
│   ├── policies/              # Dữ liệu chính sách
│   ├── products/              # Dữ liệu sản phẩm
│   ├── inventory/             # Dữ liệu tồn kho
│   └── stores/                # Dữ liệu cửa hàng
├── scripts/                   # Scripts hỗ trợ
│   ├── download_model.py      # Tải mô hình LLM
│   ├── ingest_data.py         # Nhập dữ liệu vào vector DB
│   └── test_chatbot.py        # Kiểm thử chatbot
├── src/                       # Mã nguồn
│   ├── agent/                 # Logic cho agent
│   │   ├── __init__.py
│   │   ├── intent_detector.py # Phát hiện ý định người dùng
│   │   └── response_generator.py # Tạo phản hồi
│   ├── database/              # Xử lý cơ sở dữ liệu
│   │   ├── __init__.py
│   │   ├── vector_store.py    # Cài đặt Chroma DB
│   │   └── inventory_db.py    # Truy vấn cơ sở dữ liệu tồn kho
│   ├── llm/                   # Tích hợp mô hình ngôn ngữ
│   │   ├── __init__.py
│   │   └── model.py          # Kết nối với DeepSeek R1
│   └── utils/                 # Các tiện ích
│       ├── __init__.py
│       └── text_processing.py # Xử lý văn bản tiếng Việt
└── ui/                       # Giao diện người dùng
    ├── __init__.py
    └── streamlit_app.py      # Giao diện Streamlit
``` 