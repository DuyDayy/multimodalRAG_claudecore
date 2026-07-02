#!/bin/bash

# Script khởi chạy Luồng Mã Hóa Dữ Liệu Offline (Offline Encoder)
echo "======================================================="
echo "🎬 Khởi động Luồng Mã Hóa Offline (Offline Encoder)"
echo "======================================================="

# Kiểm tra môi trường ảo
if [ ! -d ".venv" ]; then
    echo "❌ Không tìm thấy môi trường ảo .venv. Vui lòng cài đặt trước."
    exit 1
fi

source .venv/bin/activate

echo "Đang quét và xử lý video trong data/raw_videos..."
.venv/bin/python -m src.ingestion.offline_encoder
