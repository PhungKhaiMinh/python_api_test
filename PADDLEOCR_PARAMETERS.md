# Các Tham Số PaddleOCR Hiện Tại

**Cập nhật:** January 2026  
**Version:** 3.3 - Optimized for GPU (NumPy Compatibility Fix)

## Vị Trí Cấu Hình

File: `utils/paddleocr_engine.py`  
Function: `get_paddleocr_instance()` (lines 188-284)

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

### 5. **text_det_thresh** = `0.2` ⚡ OPTIMIZED
- **Mô tả:** Ngưỡng confidence cho text detection
- **Giá trị:** `0.2` (20%)
- **Phạm vi:** 0.0 - 1.0
- **Mục đích:** Slightly higher threshold to reduce noise while keeping most text
- **Ảnh hưởng:**
  - **Giá trị thấp (0.1-0.15):** Phát hiện nhiều text hơn, có thể có nhiều false positives
  - **Giá trị cao (0.3-0.5):** Chỉ phát hiện text rõ ràng, có thể bỏ sót text mờ
- **Khuyến nghị:** 0.2 là giá trị cân bằng tốt cho HMI screens với GPU

### 6. **text_det_box_thresh** = `0.3` ⚡ OPTIMIZED
- **Mô tả:** Ngưỡng confidence cho bounding box detection
- **Giá trị:** `0.3` (30%)
- **Phạm vi:** 0.0 - 1.0
- **Mục đích:** Filter weak boxes to reduce processing time
- **Ảnh hưởng:**
  - **Giá trị thấp:** Giữ nhiều bounding boxes hơn
  - **Giá trị cao:** Chỉ giữ bounding boxes chắc chắn
- **Khuyến nghị:** 0.3 phù hợp với HMI screens có text rõ ràng

### 7. **text_det_unclip_ratio** = `1.6` ⚡ OPTIMIZED
- **Mô tả:** Tỷ lệ mở rộng bounding box
- **Giá trị:** `1.6`
- **Phạm vi:** Thường 1.5 - 3.0
- **Mục đích:** Standard ratio để bao phủ text vừa đủ
- **Ảnh hưởng:**
  - **Giá trị thấp (1.5-1.8):** Bounding box chặt hơn, xử lý nhanh hơn
  - **Giá trị cao (2.2-3.0):** Bounding box rộng hơn, có thể bao gồm text khác
- **Khuyến nghị:** 1.6 là giá trị tối ưu cho tốc độ với GPU

### 8. **text_rec_score_thresh** = `0.3` ⚡ OPTIMIZED
- **Mô tả:** Ngưỡng confidence cho text recognition
- **Giá trị:** `0.3` (30%)
- **Phạm vi:** 0.0 - 1.0
- **Mục đích:** Filter low-confidence results early for faster processing
- **Ảnh hưởng:**
  - **Giá trị 0.0:** Giữ tất cả kết quả, kể cả text có confidence thấp
  - **Giá trị 0.3:** Chỉ giữ text có confidence >= 30%
- **Lý do:** Kết hợp với IoU filtering để đảm bảo kết quả chính xác

### 9. **text_det_limit_side_len** = `960` ⚡ OPTIMIZED
- **Mô tả:** Kích thước tối đa của cạnh ảnh khi resize cho detection
- **Giá trị:** `960` pixels
- **Phạm vi:** Thường 320, 512, 640, 960
- **Mục đích:** Larger for better accuracy on HMI screens (800x600 to 1920x1080)
- **Ảnh hưởng:**
  - **Giá trị thấp (320-512):** Xử lý nhanh hơn, có thể giảm độ chính xác với text nhỏ
  - **Giá trị cao (640-960):** Xử lý chậm hơn, nhưng chính xác hơn với text nhỏ
- **Khuyến nghị:** 960 với GPU để đảm bảo accuracy tốt

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
- Thư mục chứa detection model (sử dụng PP-OCRv5_server_det)

### **rec_model_dir** = `None` (mặc định)
- Thư mục chứa recognition model (sử dụng en_PP-OCRv5_mobile_rec)

### **use_gpu** = `True` (tự động nếu có PaddlePaddle GPU)
- Sử dụng GPU nếu PaddlePaddle được compile với CUDA

## Environment Variables Được Set

Trước khi khởi tạo PaddleOCR, hệ thống set các biến môi trường:

```python
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'
os.environ['GLOG_minloglevel'] = '3'
os.environ['FLAGS_call_stack_level'] = '0'
os.environ['PADDLE_PDX_SILENT_MODE'] = '1'

# GPU Memory optimization
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.5'  # Use 50% GPU memory
os.environ['FLAGS_eager_delete_tensor_gb'] = '0.0'  # Immediate tensor cleanup
```

**Mục đích:** 
- Giảm output log không cần thiết từ PaddleOCR
- Tối ưu sử dụng GPU memory

## Tối Ưu Hóa v3.1 (GPU Edition)

### 1. **Tốc Độ:**
- Tắt các tính năng không cần thiết (orientation, unwarping)
- Resize ảnh về 960px để tận dụng GPU
- Sử dụng singleton pattern (khởi tạo 1 lần duy nhất)
- Filter threshold cao hơn để giảm xử lý không cần thiết

### 2. **Độ Chính Xác:**
- `text_det_thresh=0.2`: Balance between speed and accuracy
- `text_rec_score_thresh=0.3`: Filter low-confidence early
- `text_det_limit_side_len=960`: Larger for better accuracy with GPU

### 3. **GPU Memory:**
- Environment variables để tối ưu GPU memory
- Suppress output để giảm memory overhead
- Singleton pattern giảm memory footprint

### 4. **CUDA Setup:**
- PaddlePaddle GPU 3.0.0 với CUDA 11.8
- NVIDIA libraries (cusparse, cublas, cudnn, etc.) cu11 version
- **Lưu ý**: Không cài PyTorch cùng lúc để tránh xung đột CUDA DLLs

## Khuyến Nghị Điều Chỉnh

### Nếu Cần Tăng Độ Chính Xác (chấp nhận chậm hơn):
```python
text_det_thresh=0.15,          # Giảm để phát hiện text mờ hơn
text_det_box_thresh=0.2,       # Giảm để giữ nhiều boxes hơn
text_rec_score_thresh=0.0,     # Giữ tất cả kết quả
use_doc_orientation_classify=True,  # Bật nếu ảnh có thể bị xoay
```

### Nếu Cần Tăng Tốc Độ:
```python
text_det_thresh=0.3,           # Tăng để giảm số lượng detection
text_det_box_thresh=0.4,       # Tăng để lọc weak boxes
text_det_limit_side_len=512,   # Giảm để xử lý nhanh hơn
text_rec_score_thresh=0.5,     # Lọc ngay từ recognition
```

### Nếu Ảnh Có Text Nhỏ:
```python
text_det_limit_side_len=1280,  # Tăng để phát hiện text nhỏ tốt hơn
text_det_unclip_ratio=2.0,     # Tăng để bao phủ text nhỏ đầy đủ
text_det_thresh=0.15,          # Giảm để phát hiện text mờ
```

## Code Khởi Tạo Hiện Tại (v3.2)

```python
_paddle_ocr_instance = PaddleOCR(
    lang='en',
    # Disable unnecessary preprocessing for speed
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    # Detection parameters - OPTIMIZED FOR GPU
    text_det_thresh=0.2,           # Higher to reduce noise
    text_det_box_thresh=0.3,       # Filter weak boxes
    text_det_unclip_ratio=1.6,     # Standard ratio
    text_det_limit_side_len=960,   # Larger for HMI screens
    text_det_limit_type='max',
    # Recognition parameters
    text_rec_score_thresh=0.3,     # Filter low-confidence results
)
```

## OCR Method Support (v3.3)

Hệ thống hỗ trợ cả hai phương thức gọi PaddleOCR:

### `ocr()` method (PaddleOCR 2.7.3 - hiện tại):
```python
results = ocr.ocr(image_path, cls=False)
# Returns: [[box, (text, score)], ...]
# Format: [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], (text, score)], ...]
# Direct list format
```

### `predict()` method (PaddleOCR 3.x - tương lai):
```python
results = ocr.predict(image_path)
# Returns: List of PredictResult objects
# Access data via: result.json['res']['rec_texts']
```

### Auto-Detection:
Code tự động phát hiện method nào available:
```python
if hasattr(ocr, 'predict'):
    results = ocr.predict(temp_path)  # PaddleOCR 3.x
else:
    results = ocr.ocr(temp_path, cls=False)  # PaddleOCR 2.x
```

**Lưu ý**: Hiện tại sử dụng PaddleOCR 2.7.3 với method `ocr()`, nhưng code đã sẵn sàng cho PaddleOCR 3.x.

## GPU Setup Requirements

### PaddlePaddle GPU Installation
```bash
# Bước 1: Cài đặt NumPy < 2.0 (QUAN TRỌNG!)
pip install "numpy<2.0" --force-reinstall

# Bước 2: Cài đặt PaddlePaddle GPU với CUDA 11.8
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# Bước 3: Cài đặt PaddleOCR 2.7.3 (tương thích tốt với PaddlePaddle 3.0.0)
pip install paddleocr==2.7.3
```

### Kiểm tra GPU
```python
import paddle
print('GPU available:', paddle.device.is_compiled_with_cuda())
print('GPU count:', paddle.device.cuda.device_count())
print('GPU name:', paddle.device.cuda.get_device_name(0))
```

### Lưu ý quan trọng
- **KHÔNG** cài PyTorch cùng lúc với PaddlePaddle GPU vì xung đột CUDA DLLs
- **PHẢI** cài NumPy < 2.0 để tương thích với scipy, scikit-image
- Nếu cần PyTorch, sử dụng virtual environment riêng
- CuPy có thể cài thêm để tăng tốc xử lý ảnh

## Tham Khảo

- PaddleOCR Documentation: https://github.com/PaddlePaddle/PaddleOCR
- PaddlePaddle GPU: https://www.paddlepaddle.org.cn/install/quick
- File cấu hình: `utils/paddleocr_engine.py` (lines 188-245)
