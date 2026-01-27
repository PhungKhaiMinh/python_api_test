# Tài Liệu Chi Tiết: Logic Xử Lý POST /api/images

## Tổng Quan

API endpoint `POST /api/images` là endpoint chính để xử lý ảnh và thực hiện OCR sử dụng PaddleOCR. Endpoint này hỗ trợ cả máy khu F1 và F4 với logic xử lý thống nhất.

## Cấu Trúc Request

```
POST /api/images
Content-Type: multipart/form-data

Parameters:
- file: Image file (required) - Hỗ trợ: png, jpg, jpeg, gif, bmp
- area: Mã khu vực (required) - Ví dụ: "F1", "F4"
- machine_code: Mã máy (required) - Ví dụ: "IE-F1-CWA01", "IE-F4-WBI01"
```

## Quy Trình Xử Lý Chi Tiết

### BƯỚC 1: Phát Hiện và Trích Xuất Màn Hình HMI

**Mục đích:** Tách phần màn hình HMI khỏi ảnh gốc để tăng độ chính xác OCR.

**Quy trình:**
1. **Enhance Image (Cải thiện ảnh):**
   - Chuyển đổi BGR → RGB → PIL Image
   - Tăng contrast lên 2x
   - Áp dụng CLAHE (Contrast Limited Adaptive Histogram Equalization)
   - Gaussian Blur để làm mịn

2. **Edge Detection (Phát hiện cạnh):**
   - Sử dụng Canny Edge Detection với threshold động
   - Kết hợp với Sobel filter (5x5 kernel)
   - Dilation và Erosion để nối các cạnh bị đứt

3. **Line Detection (Phát hiện đường thẳng):**
   - Sử dụng HoughLinesP để phát hiện đường thẳng
   - Phân loại thành horizontal lines và vertical lines
   - Lọc các đường thẳng quá ngắn (< 2% kích thước ảnh)

4. **Rectangle Detection (Phát hiện hình chữ nhật):**
   - Tìm hình chữ nhật từ các đường thẳng đã phát hiện
   - Validate: aspect ratio < 5, diện tích 1-90% ảnh gốc
   - Fine-tune và warp perspective để chỉnh góc nghiêng

5. **Kết quả:**
   - Nếu phát hiện được HMI screen → sử dụng phần đã trích xuất
   - Nếu không → sử dụng ảnh gốc

**File xử lý:** 
- `utils/image_processor.py` - `detect_hmi_screen()` (wrapper)
- `utils/paddleocr_engine.py` - `detect_hmi_screen_paddle()` (implementation)

---

### BƯỚC 2: Full Image OCR với PaddleOCR

**Mục đích:** Thực hiện OCR toàn bộ ảnh để phát hiện tất cả text.

**Quy trình:**
1. **Khởi tạo PaddleOCR (v3.2 GPU Optimized):**
   - Sử dụng singleton pattern để tối ưu hiệu suất
   - Cấu hình GPU optimized:
     - `lang='en'`
     - `text_det_thresh=0.2` (higher to reduce noise)
     - `text_det_box_thresh=0.3` (filter weak boxes)
     - `text_det_limit_side_len=960` (larger for HMI screens)
     - `text_rec_score_thresh=0.3` (filter low-confidence)

2. **OCR Processing (supports PaddleOCR 2.x và 3.x):**
   - **PaddleOCR 2.7.3** (hiện tại, sử dụng `ocr()` method):
     - Trả về list of lists: `[[box, (text, score)], ...]`
     - Mỗi item là `[polygon, (text_string, confidence)]`
     - Format: `[[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], (text, score)], ...]`
   
   - **PaddleOCR 3.x** (tương lai, sử dụng `predict()` method):
     - Trả về list of `PredictResult` objects
     - Mỗi object có `.json` property với structure:
       ```json
       {
         "input_path": "...",
         "res": {
           "rec_texts": ["text1", "text2", ...],
           "rec_scores": [0.99, 0.95, ...],
           "dt_polys": [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], ...]
         }
       }
       ```

3. **Extract OCR Data:**
   - Hàm `extract_ocr_data()` auto-detect format và chuyển đổi thành format chuẩn:
     ```python
     {
         'text': str,           # Text đã nhận dạng
         'confidence': float,   # Confidence score (0-1)
         'bbox': [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]  # Polygon coordinates
     }
     ```

4. **Debug Logging (v3.2):**
   - Log HMI image shape và dtype
   - Log PaddleOCR results type và structure
   - Log sample OCR data (first 5 items)
   ```
   [DEBUG] HMI image shape: (2293, 1743, 3), dtype: uint8
   [DEBUG] PaddleOCR results type: <class 'list'>
   [DEBUG] results[0].json keys: ['input_path', 'res']
   [DEBUG] Sample texts (first 5): ['text1', 'text2', ...]
   ```

**File xử lý:** 
- `utils/paddleocr_engine.py` - `read_image_with_paddleocr()` (supports 2.x và 3.x)
- `utils/paddleocr_engine.py` - `extract_ocr_data()` (multi-format extraction)

---

### BƯỚC 3: Screen Matching dựa trên Special_rois

**Mục đích:** Xác định màn hình và sub-page dựa trên các Special_rois được phát hiện.

**Quy trình:**
1. **Load ROI Info:**
   - Đọc từ `roi_data/roi_info.json`
   - Cấu trúc: `machines > machine_type > machine_code > screens > screen_name > sub_pages > sub_page`

2. **Text Normalization:**
   - Chuẩn hóa text OCR và Special_rois:
     - Uppercase
     - Loại bỏ dấu `:`
     - Normalize `"ST14 - LEAK"` → `"ST14-LEAK"`
     - Loại bỏ khoảng trắng thừa

3. **Fuzzy Matching:**
   - Sử dụng Levenshtein distance để so sánh
   - Threshold: 75% similarity
   - Kiểm tra substring match (nếu một chuỗi chứa chuỗi kia và length ratio >= 70%)

4. **Screen Matching Logic:**
   - Duyệt qua tất cả machine_type, machine_code, screen, sub_page
   - Filter theo `area` và `machine_code` nếu được cung cấp
   - Đếm số Special_rois khớp với OCR text
   - Chọn screen/sub_page có match_count cao nhất
   - Nếu bằng nhau, chọn theo match_percentage cao hơn

5. **Kết quả:**
   - Trả về: `(machine_type, machine_code, screen_name, sub_page, sub_page_data, match_count, match_percentage)`
   - Nếu không tìm thấy → `(None, None, None, None, None, 0, 0)`

**File xử lý:** `utils/paddleocr_engine.py` - `find_matching_screen()`

**Ví dụ:**
- F1 machines: `Production_Data`, `Reject_Summary` (sub_pages: 1, 2)
- F4 machines: 
  - F41: `Injection`, `Temp`, `Production`
  - F42: `Setting`, `Overview`, `Tracking`

---

### BƯỚC 4: Filter OCR Results bằng IoU với ROIs

**Mục đích:** Lọc và gán các text OCR vào đúng ROI tương ứng.

**Quy trình:**
1. **Convert Polygon to Bounding Box:**
   - Chuyển đổi polygon (4 điểm) → normalized bbox `[x1, y1, x2, y2]`
   - Normalize theo kích thước ảnh (0-1 range)

2. **Calculate IoU (Intersection over Union):**
   - Với mỗi OCR text box, tính IoU với tất cả ROIs
   - IoU = Intersection Area / Union Area
   - Threshold: `IOU_THRESHOLD = 0.01` (1%)

3. **Match OCR to ROI:**
   - Với mỗi OCR item, tìm ROI có IoU cao nhất
   - Chỉ giữ lại nếu IoU >= threshold
   - Gán `matched_roi` = tên ROI

4. **Kết quả:**
   - List các OCR items đã được match với ROI:
     ```python
     {
         'text': str,
         'confidence': float,
         'matched_roi': str,      # Tên ROI (roi_index)
         'iou': float,
         'roi_coords': [x1, y1, x2, y2]
     }
     ```

**File xử lý:** `utils/paddleocr_engine.py` - `filter_ocr_by_roi()`

---

### BƯỚC 5: Post-Processing và Formatting

**Mục đích:** Xử lý text và áp dụng format theo cấu hình.

**Quy trình:**
1. **Post-Process OCR Text:**
   - Chuyển đổi lỗi OCR phổ biến:
     - `'O'` (chữ O) → `'0'` (số 0) nếu độ dài = 1
     - Nếu nhiều ký tự nghi ngờ (O, U, I, L, C) → thử chuyển thành số
     - Ví dụ: `"O1O"` → `"010"`

2. **Apply Decimal Places Format:**
   - Đọc cấu hình từ `roi_data/decimal_places.json`
   - Cấu trúc:
     - **Standard:** `machine_type > screen_id > roi_name`
     - **Reject_Summary:** `machine_type > screen_id > machine_code > sub_page > roi_name`
   - Format số theo số chữ số thập phân:
     - `decimal_places = 0`: Loại bỏ dấu chấm (ví dụ: `"5.63"` → `"6"`)
     - `decimal_places = 2`: Giữ 2 chữ số (ví dụ: `"563"` → `"5.63"`)

3. **Kết quả:**
   - Text đã được format đúng theo cấu hình

**File xử lý:**
- `utils/paddleocr_engine.py` - `post_process_ocr_text()`
- `utils/ocr_processor.py` - `apply_decimal_places_format()`

---

### BƯỚC 6: Deduplication - Loại Bỏ Trùng Lặp

**Mục đích:** Đảm bảo mỗi `roi_index` chỉ có 1 kết quả duy nhất với IOU cao nhất.

**Vấn đề:**
- Nhiều OCR text boxes có thể match với cùng 1 ROI
- Ví dụ: "ST06 TESTED" xuất hiện 2 lần với IOU khác nhau

**Giải pháp:**
1. **Group by roi_index:**
   - Tạo dictionary `roi_index_map` để lưu kết quả theo `roi_index`

2. **Keep Highest IOU:**
   - Với mỗi kết quả, kiểm tra:
     - Nếu `roi_index` chưa có trong map → thêm vào
     - Nếu đã có → so sánh IOU:
       - Giữ lại kết quả có IOU cao hơn
       - Loại bỏ kết quả có IOU thấp hơn

3. **Kết quả:**
   - Mỗi `roi_index` chỉ có 1 kết quả duy nhất
   - Kết quả đó có IOU cao nhất trong các matches

**Ví dụ:**
```python
# Trước deduplication:
[
    {"roi_index": "ST06 TESTED", "text": "14934", "iou": 0.449},
    {"roi_index": "ST06 TESTED", "text": "", "iou": 0.015}
]

# Sau deduplication:
[
    {"roi_index": "ST06 TESTED", "text": "14934", "iou": 0.449}
]
```

**File xử lý:** 
- `routes/image_routes.py` - lines 202-218 (deduplication logic)
- `utils/paddleocr_engine.py` - `filter_ocr_by_roi()` (IoU filtering)

---

## Cấu Trúc Response

```json
{
    "success": true,
    "filename": "1234567890_image.jpg",
    "machine_code": "IE-F1-CWA01",
    "machine_type": "F1",
    "screen_id": "Reject_Summary",
    "area": "F1",
    "sub_page": "1",
    "hmi_detection": {
        "hmi_extracted": true,
        "hmi_size": "1797x2362",
        "extraction_time": 0.15
    },
    "screen_matching": {
        "matched": true,
        "match_count": 7,
        "match_percentage": 100.0
    },
    "ocr_results": [
        {
            "roi_index": "ST06 TESTED",
            "text": "14934",
            "confidence": 0.9975733757019043,
            "has_text": true,
            "original_value": "14934",
            "iou": 0.44900161899622143
        },
        {
            "roi_index": "ST06 REJECTS",
            "text": "842",
            "confidence": 0.9752138257026672,
            "has_text": true,
            "original_value": "842",
            "iou": 0.5255918337797731
        }
    ],
    "roi_count": 2,
    "ocr_engine": "PaddleOCR",
    "processing_time": {
        "hmi_detection": 0.15,
        "ocr": 1.23,
        "matching": 0.05,
        "filtering": 0.02,
        "total": 1.45
    }
}
```

## Hỗ Trợ Máy Khu F1 và F4

### F1 Machines
- **Machine Types:** `F1`
- **Screens:**
  - `Production_Data` (sub_page: 1)
  - `Reject_Summary` (sub_pages: 1, 2)
- **ROI Examples:** `ST02 TESTED`, `ST06 REJECTS`, `ST06 %`, `TIME EFFICIENCY`, `YIELD`

### F4 Machines
- **Machine Types:** `F41`, `F42`
- **F41 Screens:**
  - `Injection` (sub_page: 1)
  - `Temp` (sub_page: 1)
  - `Production` (sub_page: 1)
- **F42 Screens:**
  - `Setting` (sub_page: 1)
  - `Overview` (sub_page: 1)
  - `Tracking` (sub_page: 1)
- **ROI Examples (F41):** `Injection speed`, `Charge torque`, `Holding pressure 1`
- **ROI Examples (F42):** `Vung1 temp_current`, `Thu1 Ap suat`, `Kep Vtri`

### Logic Chung
- **Cùng một quy trình xử lý** cho tất cả machine types
- **Cấu trúc dữ liệu thống nhất** trong `roi_info.json` và `decimal_places.json`
- **Deduplication logic** hoạt động cho tất cả các máy

## Các File Liên Quan

1. **Routes:**
   - `routes/image_routes.py` - Endpoint handler (POST /api/images)

2. **OCR Processing:**
   - `utils/paddleocr_engine.py` - PaddleOCR engine, HMI detection, screen matching, ROI filtering
   - `utils/ocr_processor.py` - OCR processing logic, decimal formatting
   - `utils/image_processor.py` - Image preprocessing, HMI detection wrapper

3. **Data Files:**
   - `roi_data/roi_info.json` - ROI definitions (bao gồm Special_rois)
   - `roi_data/decimal_places.json` - Decimal places configuration
   - `roi_data/machine_screens.json` - Machine và screen metadata

4. **Utilities:**
   - `utils/cache_manager.py` - Cache management cho ROI data
   - `utils/config_manager.py` - Machine type configuration

## Lưu Ý Quan Trọng

1. **Deduplication:** Mỗi `roi_index` chỉ trả về 1 kết quả với IOU cao nhất
2. **IoU Threshold:** Chỉ giữ OCR items có IoU >= 1% với ROI
3. **Screen Matching:** Phải match được ít nhất 1 Special_roi để xác định screen
4. **Decimal Places:** Format số theo cấu hình trong `decimal_places.json`
5. **HMI Detection:** Nếu không phát hiện được HMI screen, sử dụng ảnh gốc

## Performance (v3.2 GPU Edition)

### Với GPU (NVIDIA GTX 1050 Ti):
- **HMI Detection:** ~0.1-0.2s
- **OCR Processing:** ~0.5-1.5s (GPU accelerated)
- **PaddleOCR Warm-up:** ~0.3s (chỉ lần đầu, với GPU)
- **Screen Matching:** ~0.05s
- **Filtering & Deduplication:** ~0.02s
- **Total:** ~1-2s per image (sau warm-up)

### So sánh CPU vs GPU:

| Metric | CPU | GPU | Improvement |
|--------|-----|-----|-------------|
| OCR Time | ~12s | ~3-5s | **-60%** |
| HMI Detection | ~0.5s | ~0.2s | **-60%** |
| Warm-up | ~5s | ~0.3s | **-94%** |
| Total | ~13.5s | ~4-6s | **-55%** |

### Debug Logging Overhead:
- v3.2 có thêm debug logging nhưng không ảnh hưởng đáng kể đến performance
- Debug logs chỉ xuất hiện trong terminal output, không ảnh hưởng response time
