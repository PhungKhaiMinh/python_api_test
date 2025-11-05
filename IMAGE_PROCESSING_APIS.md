# Giải thích chi tiết các API trong mục "Image Processing"

## Tổng quan

Mục **"Image Processing"** trong Swagger UI chứa **8 API endpoints** liên quan đến việc xử lý ảnh, upload ảnh, và thực hiện OCR (Optical Character Recognition) trên các ảnh HMI (Human Machine Interface).

---

## 1. POST `/api/images` - Upload và thực hiện OCR

### Chức năng chính
**API quan trọng nhất** - Upload ảnh HMI và thực hiện toàn bộ quy trình OCR tự động.

### Quy trình xử lý (7 bước):

#### Bước 1: Phát hiện và tách HMI Screen
- Tự động phát hiện vùng màn hình HMI trong ảnh gốc
- Tách riêng phần HMI screen để xử lý
- **Lý do**: ROI coordinates được định nghĩa trên ảnh HMI đã tách, không phải ảnh gốc

#### Bước 2: Tự động phát hiện Machine và Screen
- Sử dụng template matching để xác định:
  - `machine_type`: Loại máy (F1, F41, F42)
  - `screen_id`: Loại màn hình (Production_Data, Reject_Summary, v.v.)
- So sánh với các template reference images

#### Bước 3: Phát hiện Sub-page (nếu là Reject_Summary)
- Đối với màn hình "Reject_Summary", có thể có nhiều sub-page (1, 2)
- Đọc nội dung từ vùng "special_region" ROI
- Sử dụng fuzzy matching để xác định sub-page nào

#### Bước 4: Lấy ROI coordinates
- Lấy tọa độ các vùng ROI cần OCR
- Hỗ trợ sub-page specific ROIs cho Reject_Summary

#### Bước 5: Alignment (căn chỉnh) ảnh
- So sánh ảnh với template reference
- Căn chỉnh ảnh để khớp với template
- Đảm bảo vị trí ROI chính xác

#### Bước 6: Lưu ảnh debug
- Lưu từng vùng ROI đã cắt vào `D:\python_WREMBLY_test-main\anhHMI`
- Giúp kiểm tra và debug xem ROI có được cắt đúng không

#### Bước 7: Thực hiện OCR
- Sử dụng EasyOCR để đọc text từ từng ROI
- Áp dụng decimal places formatting
- Trả về kết quả OCR cho từng ROI

### Parameters (Form Data):
- `file` (required): File ảnh (jpg, png, bmp)
- `area` (required): Mã khu vực (ví dụ: "F1", "F4")
- `machine_code` (required): Mã máy (ví dụ: "IE-F1-CWA01")

### Response:
```json
{
  "success": true,
  "filename": "1234567890_test.jpg",
  "machine_code": "IE-F1-CWA01",
  "machine_type": "F1",
  "screen_id": "Production_Data",
  "area": "F1",
  "sub_page": "1",  // Nếu là Reject_Summary
  "hmi_detection": {
    "hmi_extracted": true,
    "hmi_size": "1920x1080",
    "extraction_status": "HMI screen extracted successfully"
  },
  "ocr_results": [
    {
      "roi_name": "temperature",
      "value": "25.5",
      "confidence": 0.95
    }
  ],
  "roi_count": 5
}
```

### Use Cases:
- Upload ảnh HMI để đọc dữ liệu tự động
- Kiểm tra giá trị trên màn hình máy
- Thu thập dữ liệu sản xuất từ HMI screens

---

## 2. GET `/api/images` - Lấy danh sách ảnh đã upload

### Chức năng
Lấy danh sách tất cả các file ảnh đã được upload vào hệ thống.

### Response:
```json
{
  "images": [
    "1234567890_test.jpg",
    "1234567891_screenshot.png",
    "1234567892_hmi.bmp"
  ]
}
```

### Use Cases:
- Xem danh sách ảnh đã xử lý
- Kiểm tra lịch sử upload
- Quản lý storage

---

## 3. GET `/api/images/<filename>` - Lấy ảnh cụ thể

### Chức năng
Tải về một file ảnh cụ thể đã được upload.

### Parameters:
- `filename` (path): Tên file ảnh cần lấy

### Response:
- Trả về file ảnh (binary)
- Content-Type: image/jpeg, image/png, hoặc image/bmp

### Use Cases:
- Xem lại ảnh đã upload
- Download ảnh để phân tích
- Hiển thị ảnh trong frontend

---

## 4. DELETE `/api/images/<filename>` - Xóa ảnh

### Chức năng
Xóa một file ảnh khỏi hệ thống.

### Parameters:
- `filename` (path): Tên file ảnh cần xóa

### Response:
```json
{
  "message": "Deleted 1234567890_test.jpg"
}
```

### Use Cases:
- Xóa ảnh không cần thiết
- Giải phóng storage
- Quản lý dữ liệu

---

## 5. GET `/api/images/processed_roi/<filename>` - Lấy ảnh ROI đã xử lý

### Chức năng
Lấy ảnh các vùng ROI đã được xử lý (preprocessed) cho OCR.

### Parameters:
- `filename` (path): Tên file ảnh ROI

### Response:
- Trả về file ảnh ROI đã xử lý (binary)
- Ảnh đã được preprocessing (grayscale, resize, threshold, v.v.)

### Use Cases:
- Kiểm tra ảnh ROI trước khi OCR
- Debug preprocessing pipeline
- Xem ảnh đã được xử lý như thế nào

### Lưu ý:
- Ảnh được lưu trong thư mục `uploads/processed_roi/`
- Mỗi ROI được lưu riêng với tên file riêng

---

## 6. GET `/api/images/hmi_refined/<filename>` - Lấy ảnh HMI đã refine

### Chức năng
Lấy ảnh HMI đã được tinh chỉnh (refined) sau khi phát hiện và tách từ ảnh gốc.

### Parameters:
- `filename` (path): Tên file ảnh HMI refined

### Response:
- Trả về file ảnh HMI đã refine (binary)
- Ảnh chỉ chứa phần màn hình HMI, đã loại bỏ background

### Use Cases:
- Kiểm tra kết quả HMI detection
- Xem ảnh HMI đã được tách như thế nào
- Debug HMI extraction algorithm

### Lưu ý:
- Ảnh được lưu trong thư mục `reference_images/hmi_refined/`
- Đây là ảnh sau khi đã tách HMI screen từ ảnh gốc

---

## 7. GET `/api/images/aligned/<filename>` - Lấy ảnh đã align

### Chức năng
Lấy ảnh đã được căn chỉnh (aligned) với template reference.

### Parameters:
- `filename` (path): Tên file ảnh đã align

### Response:
- Trả về file ảnh đã align (binary)
- Ảnh đã được transform để khớp với template

### Use Cases:
- Kiểm tra kết quả alignment
- Xem ảnh đã được căn chỉnh như thế nào
- Debug alignment algorithm

### Lưu ý:
- Ảnh được lưu trong thư mục `uploads/aligned/`
- Alignment giúp đảm bảo ROI coordinates chính xác

---

## 8. GET `/api/images/hmi_detection/<filename>` - Lấy ảnh visualization HMI detection

### Chức năng
Lấy ảnh visualization của quá trình phát hiện HMI (có thể có vẽ bounding box).

### Parameters:
- `filename` (path): Tên file ảnh detection visualization

### Response:
- Trả về file ảnh visualization (binary)
- Ảnh có thể chứa các annotation về vùng HMI được phát hiện

### Use Cases:
- Debug HMI detection algorithm
- Xem vùng HMI được phát hiện ở đâu
- Kiểm tra độ chính xác của detection

### Lưu ý:
- Ảnh được lưu trong thư mục `uploads/`
- Đây là ảnh visualization, không phải ảnh thực tế được xử lý

---

## Luồng xử lý ảnh điển hình

```
1. Upload ảnh (POST /api/images)
   ↓
2. Phát hiện và tách HMI screen
   ↓
3. Tự động detect machine và screen type
   ↓
4. Detect sub-page (nếu là Reject_Summary)
   ↓
5. Lấy ROI coordinates
   ↓
6. Alignment với template
   ↓
7. Lưu debug images (các ROI)
   ↓
8. Thực hiện OCR trên từng ROI
   ↓
9. Áp dụng decimal formatting
   ↓
10. Trả về kết quả OCR
```

## Mối quan hệ giữa các API

- **POST `/api/images`**: API chính, thực hiện toàn bộ quy trình
- **GET `/api/images`**: Xem danh sách ảnh đã upload
- **GET `/api/images/<filename>`**: Xem lại ảnh gốc
- **GET `/api/images/processed_roi/<filename>`**: Xem ROI đã preprocess
- **GET `/api/images/hmi_refined/<filename>`**: Xem HMI đã tách
- **GET `/api/images/aligned/<filename>`**: Xem ảnh đã align
- **GET `/api/images/hmi_detection/<filename>`**: Xem visualization detection
- **DELETE `/api/images/<filename>`**: Xóa ảnh

## Các thư mục lưu trữ

- `uploads/`: Ảnh gốc đã upload
- `uploads/processed_roi/`: Ảnh ROI đã xử lý
- `uploads/aligned/`: Ảnh đã align
- `reference_images/hmi_refined/`: Ảnh HMI refined
- `D:\python_WREMBLY_test-main\anhHMI/`: Debug ROI images (từ code)

## Lưu ý quan trọng

1. **HMI Detection là bước quan trọng nhất**: Tất cả ROI coordinates được định nghĩa trên ảnh HMI đã tách, không phải ảnh gốc.

2. **Sub-page detection**: Chỉ áp dụng cho màn hình "Reject_Summary", sử dụng fuzzy matching để xác định sub-page.

3. **Alignment**: Giúp đảm bảo ROI coordinates chính xác, đặc biệt khi ảnh bị lệch hoặc zoom khác với template.

4. **Debug images**: Tất cả ROI images được lưu tự động để kiểm tra và debug.

5. **OCR với EasyOCR**: Sử dụng GPU nếu có, preprocessing tự động để tối ưu độ chính xác.

6. **Decimal formatting**: Kết quả OCR được format theo cấu hình decimal places dựa trên machine_code, screen_id, và roi_name.

## Ví dụ sử dụng

### Upload và OCR ảnh HMI:
```bash
curl -X POST "http://localhost:5000/api/images" \
  -F "file=@hmi_screenshot.jpg" \
  -F "area=F1" \
  -F "machine_code=IE-F1-CWA01"
```

### Xem danh sách ảnh:
```bash
curl "http://localhost:5000/api/images"
```

### Xem ảnh cụ thể:
```bash
curl "http://localhost:5000/api/images/1234567890_test.jpg" --output image.jpg
```

### Xóa ảnh:
```bash
curl -X DELETE "http://localhost:5000/api/images/1234567890_test.jpg"
```

