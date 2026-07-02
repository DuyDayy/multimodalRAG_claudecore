#!/bin/bash

# Script khởi chạy Giao Diện Ứng Dụng (Streamlit UI)
echo "======================================================="
echo "🚀 Khởi động Hệ Thống Multimodal RAG (Giao Diện)"
echo "======================================================="

# Kiểm tra xem môi trường ảo đã tồn tại chưa
if [ ! -d ".venv" ]; then
    echo "❌ Không tìm thấy môi trường ảo .venv. Vui lòng cài đặt trước."
    exit 1
fi

# Kích hoạt môi trường ảo
source .venv/bin/activate


# Khởi chạy giao diện Streamlit
echo "Đang khởi chạy Streamlit..."
.venv/bin/python -m streamlit run app.py
