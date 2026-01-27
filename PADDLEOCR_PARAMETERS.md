# Các Tham Số PaddleOCR Hiện Tại

## Vị Trí Cấu Hình

File: `utils/paddleocr_engine.py`  
Function: `get_paddleocr_instance()` (lines 81-124)

## Các Tham Số Đang Sử Dụng

### 1. **lang** = `'en'`
- **Mô tả:** Ngôn ngữ OCR
- **Giá trị:** `'en'` (English)
- **Mục đích:** Nhận dạng text tiếng Anh và số
- **Có thể thay đổi:** Có (các giá trị: 'ch', 'korean', 'japan', v.v.)

### 2. **use_doc_orientation_classify** = `False`
- **Mô tả:** Có sử dụng classification hướng tài liệu không
- **Giá trị:** `False` (tắt)
- **Mục đích:** Tắt tính năng tự động xoay ảnh theo hướng tài liệu để tăng tốc độ
- **Ảnh hưởng:** Tăng tốc độ xử lý, giảm độ chính xác nếu ảnh bị xoay

### 3. **use_doc_unwarping** = `False`
- **Mô tả:** Có sử dụng unwarping (làm phẳng) tài liệu không
- **Giá trị:** `False` (tắt)
- **Mục đích:** Tắt tính năng làm phẳng ảnh cong để tăng tốc độ
- **Ảnh hưởng:** Tăng tốc độ xử lý, giảm độ chính xác nếu ảnh bị cong

### 4. **use_textline_orientation** = `False`
- **Mô tả:** Có sử dụng classification hướng textline không
- **Giá trị:** `False` (tắt)
- **Mục đích:** Tắt tính năng tự động xoay textline để tăng tốc độ
- **Ảnh hưởng:** Tăng tốc độ xử lý

### 5. **text_det_thresh** = `0.15`
- **Mô tả:** Ngưỡng confidence cho text detection
- **Giá trị:** `0.15` (15%)
- **Phạm vi:** 0.0 - 1.0
- **Mục đích:** Chỉ phát hiện text box có confidence >= 15%
- **Ảnh hưởng:**
  - **Giá trị thấp (0.1-0.2):** Phát hiện nhiều text hơn, có thể có nhiều false positives
  - **Giá trị cao (0.3-0.5):** Chỉ phát hiện text rõ ràng, có thể bỏ sót text mờ
- **Khuyến nghị:** 0.15 là giá trị cân bằng tốt cho HMI screens

### 6. **text_det_box_thresh** = `0.25`
- **Mô tả:** Ngưỡng confidence cho bounding box detection
- **Giá trị:** `0.25` (25%)
- **Phạm vi:** 0.0 - 1.0
- **Mục đích:** Chỉ giữ lại bounding box có confidence >= 25%
- **Ảnh hưởng:**
  - **Giá trị thấp:** Giữ nhiều bounding boxes hơn
  - **Giá trị cao:** Chỉ giữ bounding boxes chắc chắn
- **Khuyến nghị:** 0.25 phù hợp với HMI screens có text rõ ràng

### 7. **text_det_unclip_ratio** = `2.2`
- **Mô tả:** Tỷ lệ mở rộng bounding box
- **Giá trị:** `2.2`
- **Phạm vi:** Thường 1.5 - 3.0
- **Mục đích:** Mở rộng bounding box để bao phủ đầy đủ text
- **Ảnh hưởng:**
  - **Giá trị thấp (1.5-2.0):** Bounding box chặt hơn, có thể cắt mất một phần text
  - **Giá trị cao (2.5-3.0):** Bounding box rộng hơn, có thể bao gồm text khác
- **Khuyến nghị:** 2.2 là giá trị tốt cho HMI screens

### 8. **text_rec_score_thresh** = `0.0`
- **Mô tả:** Ngưỡng confidence cho text recognition
- **Giá trị:** `0.0` (0% - không lọc)
- **Phạm vi:** 0.0 - 1.0
- **Mục đích:** Giữ lại tất cả kết quả recognition, không lọc theo confidence
- **Ảnh hưởng:**
  - **Giá trị 0.0:** Giữ tất cả kết quả, kể cả text có confidence thấp
  - **Giá trị cao (0.5-0.7):** Chỉ giữ text có confidence cao
- **Lý do:** Để hệ thống tự xử lý và filter sau (qua IoU matching)

### 9. **text_det_limit_side_len** = `512`
- **Mô tả:** Kích thước tối đa của cạnh ảnh khi resize cho detection
- **Giá trị:** `512` pixels
- **Phạm vi:** Thường 320, 512, 640, 960
- **Mục đích:** Resize ảnh về kích thước này trước khi detection để tăng tốc
- **Ảnh hưởng:**
  - **Giá trị thấp (320-512):** Xử lý nhanh hơn, có thể giảm độ chính xác với text nhỏ
  - **Giá trị cao (640-960):** Xử lý chậm hơn, nhưng chính xác hơn với text nhỏ
- **Khuyến nghị:** 512 là giá trị cân bằng tốt

### 10. **text_det_limit_type** = `'max'`
- **Mô tả:** Cách resize ảnh khi vượt quá limit_side_len
- **Giá trị:** `'max'` (resize theo cạnh dài nhất)
- **Các giá trị:** `'max'`, `'min'`
- **Mục đích:** Resize theo cạnh dài nhất để giữ tỷ lệ khung hình
- **Ảnh hưởng:**
  - **'max':** Resize theo cạnh dài nhất, giữ tỷ lệ
  - **'min':** Resize theo cạnh ngắn nhất

## Các Tham Số Không Được Cấu Hình (Sử Dụng Mặc Định)

### **use_angle_cls** = `False` (mặc định)
- Classification góc xoay của text

### **cls_thresh** = `0.9` (mặc định)
- Ngưỡng confidence cho angle classification

### **det_model_dir** = `None` (mặc định)
- Thư mục chứa detection model (sử dụng model mặc định)

### **rec_model_dir** = `None` (mặc định)
- Thư mục chứa recognition model (sử dụng model mặc định)

### **use_gpu** = `True` (mặc định nếu có GPU)
- Sử dụng GPU nếu có

## Environment Variables Được Set

Trước khi khởi tạo PaddleOCR, hệ thống set các biến môi trường:

```python
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'
os.environ['GLOG_minloglevel'] = '3'
os.environ['FLAGS_call_stack_level'] = '0'
os.environ['PADDLE_PDX_SILENT_MODE'] = '1'
```

**Mục đích:** Giảm output log không cần thiết từ PaddleOCR

## Tối Ưu Hóa Hiện Tại

### 1. **Tốc Độ:**
- Tắt các tính năng không cần thiết (orientation, unwarping)
- Resize ảnh về 512px
- Sử dụng singleton pattern (khởi tạo 1 lần duy nhất)

### 2. **Độ Chính Xác:**
- `text_det_thresh=0.15`: Phát hiện text mờ
- `text_rec_score_thresh=0.0`: Giữ tất cả kết quả để filter sau
- `text_det_unclip_ratio=2.2`: Bao phủ đầy đủ text

### 3. **Memory:**
- Suppress output để giảm memory overhead
- Singleton pattern giảm memory footprint

## Khuyến Nghị Điều Chỉnh

### Nếu Cần Tăng Độ Chính Xác:
```python
text_det_thresh=0.2,          # Tăng từ 0.15 → 0.2
text_det_box_thresh=0.3,       # Tăng từ 0.25 → 0.3
text_det_limit_side_len=640,   # Tăng từ 512 → 640
use_doc_orientation_classify=True,  # Bật nếu ảnh có thể bị xoay
```

### Nếu Cần Tăng Tốc Độ:
```python
text_det_thresh=0.2,           # Tăng để giảm số lượng detection
text_det_limit_side_len=320,   # Giảm từ 512 → 320
text_rec_score_thresh=0.5,     # Lọc ngay từ recognition
```

### Nếu Ảnh Có Text Nhỏ:
```python
text_det_limit_side_len=960,   # Tăng để phát hiện text nhỏ tốt hơn
text_det_unclip_ratio=2.5,     # Tăng để bao phủ text nhỏ đầy đủ
```

## Code Khởi Tạo Hiện Tại

```python
_paddle_ocr_instance = PaddleOCR(
    lang='en',
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    text_det_thresh=0.15,
    text_det_box_thresh=0.25,
    text_det_unclip_ratio=2.2,
    text_rec_score_thresh=0.0,
    text_det_limit_side_len=512,
    text_det_limit_type='max',
)
```

## Tham Khảo

- PaddleOCR Documentation: https://github.com/PaddlePaddle/PaddleOCR
- File cấu hình: `utils/paddleocr_engine.py` (lines 95-106)
