# HMI OCR API Server - Hướng Dẫn Đầy Đủ

**Phiên bản:** 3.3 - PaddleOCR GPU Edition (NumPy Compatibility Fix)  
**Cập nhật:** January 2026  
**Trạng thái:** ✅ Sẵn sàng Production (GPU Accelerated)

---

## 📋 Mục Lục

1. [Giới Thiệu](#giới-thiệu)
2. [Yêu Cầu Hệ Thống](#yêu-cầu-hệ-thống)
3. [Cài Đặt](#cài-đặt)
4. [Cấu Trúc Dự Án](#cấu-trúc-dự-án)
5. [Chạy Server](#chạy-server)
6. [API Endpoints](#api-endpoints)
7. [Cấu Hình](#cấu-hình)
8. [Khắc Phục Sự Cố](#khắc-phục-sự-cố)
9. [Bảo Trì](#bảo-trì)

---

## 🎯 Giới Thiệu

**HMI OCR API Server** là hệ thống API dựa trên Flask để:
- Tự động nhận diện loại màn hình HMI (Human-Machine Interface)
- Trích xuất thông số từ ảnh màn hình bằng OCR (Optical Character Recognition)
- Xử lý song song với GPU acceleration
- Quản lý cấu hình máy móc và ROI (Region of Interest)

### Tính Năng Chính

✅ **Tự động phát hiện màn hình**: PaddleOCR-based HMI detection algorithm  
✅ **OCR GPU-accelerated**: PaddleOCR với CUDA (exclusively PaddleOCR)  
✅ **Screen Matching**: Fuzzy matching dựa trên Special_rois  
✅ **ROI Filtering**: IoU-based filtering để match OCR với ROIs  
✅ **Xử lý song song**: Multi-threading với 24 workers  
✅ **Cache thông minh**: ROI info và configuration caching  
✅ **RESTful API**: 27 API endpoints đầy đủ tính năng (unified decimal places API)  
✅ **Swagger UI**: Tài liệu API tương tác tại `/apidocs`  
✅ **Sub-page Detection**: Hỗ trợ Reject_Summary với nhiều sub-pages  
✅ **History Filtering**: Lọc lịch sử OCR theo machine_code, area, screen_id, time range  
✅ **Deduplication**: Tự động loại bỏ trùng lặp, giữ kết quả có IOU cao nhất

---

## 💻 Yêu Cầu Hệ Thống

### Phần Cứng Tối Thiểu

| Thành phần | Yêu cầu tối thiểu | Khuyến nghị |
|------------|-------------------|-------------|
| **CPU** | Intel Core i5 gen 8 hoặc tương đương | Intel Core i7 gen 10+ |
| **RAM** | 8 GB | 16 GB trở lên |
| **GPU** | Không bắt buộc | NVIDIA GPU với 4GB VRAM+ |
| **Ổ cứng** | 10 GB trống | SSD 20 GB+ |
| **Hệ điều hành** | Windows 10/11 64-bit | Windows 11 64-bit |

### Phần Cứng Đã Test

```
Tên máy: MSI
CPU: 12 cores
RAM: 16 GB
GPU: NVIDIA GeForce GTX 1050 Ti (4GB VRAM)
GPU Driver: 580.97
OS: Windows 11 64-bit (Build 2009)
Python: 3.12.10
CUDA: 12.1
```

### Phần Mềm Cần Thiết

1. **Python 3.8 - 3.12** (Đã test với Python 3.12.10)
2. **CUDA Toolkit 12.x** (nếu dùng GPU)
3. **Visual C++ Redistributable** (cho một số thư viện)
4. **Git** (để clone repository - tùy chọn)

---

## 🔧 Cài Đặt

### Bước 1: Cài Đặt Python

1. **Download Python 3.12.x** từ https://www.python.org/downloads/
2. **Chạy installer** với các tùy chọn:
   - ✅ **Add Python to PATH** (Quan trọng!)
   - ✅ Install for all users
   - ✅ Install pip
3. **Kiểm tra cài đặt**:
   ```bash
   python --version
   # Kết quả: Python 3.12.10
   
   pip --version
   # Kết quả: pip 24.x.x
   ```

### Bước 2: Cài Đặt CUDA (Cho GPU - Tùy Chọn)

**Lưu ý**: Nếu không có NVIDIA GPU, bỏ qua bước này. Hệ thống vẫn chạy được nhưng chậm hơn.

1. **Kiểm tra GPU**:
   ```bash
   nvidia-smi
   ```
   Xem phiên bản CUDA Compatible (ví dụ: 12.1)

2. **Download CUDA Toolkit** từ:
   https://developer.nvidia.com/cuda-downloads
   
   Chọn phiên bản phù hợp với driver (ví dụ: CUDA 12.1)

3. **Cài đặt CUDA Toolkit** theo hướng dẫn của NVIDIA

4. **Kiểm tra**:
   ```bash
   nvcc --version
   ```

### Bước 3: Giải Nén/Clone Dự Án

```bash
# Nếu có file zip
Unzip python_WREMBLY_test-main.zip

# Hoặc clone từ git
git clone [repository-url] python_WREMBLY_test-main
```

### Bước 4: Cài Đặt Dependencies

1. **Mở Terminal/PowerShell** tại thư mục dự án:
   ```bash
   cd D:\python_WREMBLY_test-main\python_api_test
   ```

2. **Tạo Virtual Environment** (Khuyến nghị):
   ```bash
   python -m venv venv
   
   # Kích hoạt virtual environment
   # Windows PowerShell:
   .\venv\Scripts\Activate.ps1
   
   # Windows CMD:
   .\venv\Scripts\activate.bat
   ```

3. **Nâng cấp pip**:
   ```bash
   python -m pip install --upgrade pip
   ```

4. **Cài đặt các packages**:
   ```bash
   pip install -r requirements.txt
   ```
   
   Quá trình này sẽ mất 5-15 phút tùy vào tốc độ mạng.

5. **Cài đặt CuPy (cho GPU)**:
   ```bash
   # Cho CUDA 12.x
   pip install cupy-cuda12x
   
   # Cho CUDA 11.x (nếu dùng CUDA 11)
   pip install cupy-cuda11x
   ```

### Bước 5: Cài Đặt NumPy (Quan Trọng!)

**⚠️ QUAN TRỌNG**: Hệ thống yêu cầu NumPy < 2.0 để tương thích với các thư viện khác.

```bash
# Downgrade NumPy về version 1.x (nếu đã cài NumPy 2.x)
pip install "numpy<2.0" --force-reinstall

# Kiểm tra version
python -c "import numpy; print('NumPy version:', numpy.__version__)"
# Kết quả mong đợi: NumPy version: 1.26.4
```

**Lý do**: NumPy 2.x không tương thích với scipy, scikit-image, và các thư viện được compile với NumPy 1.x.

### Bước 6: Cài Đặt PaddleOCR

**Quan trọng**: PaddleOCR cần cài đặt với GPU support để đạt hiệu năng tối ưu.

```bash
# Cài đặt PaddlePaddle GPU với CUDA 11.8 (KHUYẾN NGHỊ)
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# Cài đặt PaddleOCR 2.7.3 (tương thích tốt với PaddlePaddle 3.0.0)
pip install paddleocr==2.7.3

# (Tùy chọn) CuPy cho image processing acceleration
pip install cupy-cuda11x
```

**⚠️ QUAN TRỌNG**: 
- **KHÔNG** cài PyTorch cùng lúc với PaddlePaddle GPU vì xung đột CUDA DLLs
- Nếu đã cài PyTorch, gỡ bỏ trước khi cài PaddlePaddle GPU:
  ```bash
  pip uninstall torch torchvision -y
  pip uninstall nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 -y
  ```

**Kiểm tra PaddleOCR**:
```bash
python -c "import paddle; print('GPU:', paddle.device.is_compiled_with_cuda())"
python -c "from paddleocr import PaddleOCR; print('PaddleOCR installed successfully')"
```

**Kết quả mong đợi**:
```
GPU: True
PaddleOCR installed successfully
```

### Bước 7: Kiểm Tra Cài Đặt

```bash
python -c "from app import app; print('[OK] All imports successful')"
```

Nếu thành công, bạn sẽ thấy:
```
[OK] GPU Accelerator and Parallel Processor modules loaded
[OK] PaddleOCR initialized successfully
[OK] Enhanced thread pools initialized
[OK] GPU Accelerator ready
[OK] All imports successful
```

---

## 📁 Cấu Trúc Dự Án

```
python_api_test/
├── app.py                          # File chính - Main Flask application
├── app_original.py                 # Backup file gốc
│
├── utils/                          # Các modules tiện ích
│   ├── __init__.py                # Export functions
│   ├── cache_manager.py           # Quản lý cache (179 dòng)
│   ├── config_manager.py          # Quản lý cấu hình (362 dòng)
│   ├── image_processor.py         # Xử lý ảnh (300+ dòng)
│   ├── paddleocr_engine.py        # PaddleOCR engine (1400+ dòng) ⭐ NEW
│   ├── ocr_processor.py           # Xử lý OCR (550+ dòng)
│   ├── swagger_config.py          # Swagger configuration
│   ├── swagger_specs.py           # Swagger specifications
│   └── swagger_helper.py          # Swagger helper
│
├── routes/                         # API route blueprints
│   ├── __init__.py                # Export routes
│   ├── image_routes.py            # Routes xử lý ảnh (427 dòng)
│   ├── machine_routes.py          # Routes quản lý máy (290 dòng)
│   ├── decimal_routes.py          # Routes cấu hình số thập phân (447 dòng)
│   └── reference_routes.py        # Routes ảnh tham chiếu (200+ dòng)
│
├── Core modules (Optional)
│   ├── gpu_accelerator.py             # GPU acceleration
│   ├── parallel_processor.py          # Xử lý song song
│   ├── smart_detection_functions.py   # Thuật toán phát hiện màn hình (optional)
│   ├── ensemble_hog_orb_classifier.py # ML classifier (optional)
│   └── hog_svm_classifier.py          # ML classifier backup (optional)
│
├── Deployment
│   ├── wsgi.py                    # WSGI server cho production
│   ├── start_server.bat           # Script khởi động nhanh
│   └── web.config                 # Cấu hình IIS (nếu dùng)
│
├── Data folders
│   ├── roi_data/                  # Cấu hình ROI và máy móc
│   │   ├── machine_screens.json   # Cấu hình máy và màn hình
│   │   ├── roi_info.json          # Tọa độ ROI
│   │   ├── decimal_places.json    # Cấu hình số thập phân
│   │   ├── reference_images/      # Ảnh mẫu template
│   │   └── parameter_order_value.txt
│   │
│   ├── uploads/                   # Ảnh được upload
│   │   ├── aligned/               # Ảnh đã căn chỉnh
│   │   ├── hmi_refined/           # Ảnh đã tinh chỉnh
│   │   └── processed_roi/         # ROI đã xử lý
│   │
│   ├── ocr_results/               # Kết quả OCR lịch sử
│   │
│   └── Training data (Không xóa!)
│       ├── augmented_training_data/    # Dữ liệu huấn luyện
│       ├── advanced_augmented_data/    # Dữ liệu mở rộng
│       └── focused_training_data/      # Dữ liệu tập trung
│
└── Documentation
    ├── README.md                  # File này
    ├── TECHNICAL_DOCS.md          # Tài liệu kỹ thuật
    └── requirements.txt           # Dependencies
```

---

## 🚀 Chạy Server

### Chế Độ Development (Khuyến nghị cho testing)

```bash
cd D:\python_WREMBLY_test-main\python_api_test
python app.py
```

Server sẽ khởi động tại: `http://0.0.0.0:5000`

**Output mong đợi**:
```
[OK] CuPy GPU acceleration available
   - GPU Device: 0
   - GPU Memory: 4.00 GB
[WARNING] OpenCV CUDA not available - Using CPU for OpenCV
[INFO] PyTorch CUDA check skipped (using PaddlePaddle GPU instead)
[*] Parallel Processor Configuration:
   - CPU Cores: 12
   - Thread Pool Workers: 24
   - Process Pool Workers: 12
   - Default Batch Size: 8
[OK] GPU Accelerator and Parallel Processor modules loaded
[*] Initializing PaddleOCR...
[*] Initializing PaddleOCR reader (OPTIMIZED)...
[OK] PaddlePaddle GPU detected: NVIDIA GeForce GTX 1050 Ti
[OK] PaddleOCR initialized successfully (GPU mode)
[*] Warming up PaddleOCR instance...
[OK] PaddleOCR warm-up completed in 0.27s
[OK] PaddleOCR initialized successfully
[OK] OCR-ThreadPool initialized: 4-24 workers
[OK] Image-ThreadPool initialized: 4-24 workers
[OK] Enhanced thread pools initialized
[*] GPU Accelerator initialized - Using GPU
[OK] GPU Accelerator ready
[OK] Swagger UI initialized - Available at /apidocs
[OK] OCR Processor initialized with PaddleOCR
[OK] Swagger docstrings injected
[OK] Swagger docstrings injected for System routes

[*] Initializing all caches at startup...
[OK] ROI info cached successfully
[OK] ROI info cached: 2 items
[OK] Decimal places config cached successfully
[OK] Decimal places config cached: 3 items
[OK] Machine info cached successfully
[OK] Machine info cached: {'machine_code': 'F41', 'screen_id': 'Main'}
[OK] Cache initialization completed!

======================================================================
HMI OCR API SERVER - v3.1 PaddleOCR GPU Edition (OPTIMIZED)
======================================================================
Upload folder: D:\python_api_test-paddleOCR\uploads
ROI data folder: D:\python_api_test-paddleOCR\roi_data
GPU available: True
PaddleOCR available: True
OCR Engine: PaddleOCR (GPU mode)
======================================================================
OPTIMIZATIONS ENABLED:
  - GPU acceleration (if available)
  - Parallel ROI filtering
  - Parallel post-processing
  - Optimized image preprocessing
  - Performance tracking
======================================================================
Starting server on http://0.0.0.0:5000
======================================================================

 * Serving Flask app 'app'
 * Debug mode: off
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://192.168.x.x:5000
[2026-01-27 xx:xx:xx] [INFO] Press CTRL+C to quit
```

**Lưu ý**: 
- Debug mode được tắt (`debug=False`) để tránh lỗi circular import với PaddleOCR.
- OCR warm-up thường mất ~0.3s với GPU, ~5s với CPU.

### Chế Độ Production (Với Waitress)

```bash
python wsgi.py
```

Hoặc dùng batch file:
```bash
start_server.bat
```

**Waitress** an toàn hơn và hiệu năng tốt hơn cho môi trường production.

### Test Server

Mở browser và truy cập:
- http://localhost:5000/ - Trang chủ
- http://localhost:5000/debug - Thông tin debug
- http://localhost:5000/api/performance - Thông tin hiệu năng
- **http://localhost:5000/apidocs** - **Swagger UI** (Tài liệu API tương tác)

Hoặc dùng curl:
```bash
curl http://localhost:5000/
```

### Swagger UI - Tài liệu API

Hệ thống tích hợp **Swagger UI** để xem và test API trực tiếp trên trình duyệt.

**Truy cập**: http://localhost:5000/apidocs

**Tính năng:**
- ✅ Xem tất cả 27 API endpoints được phân loại theo tags
- ✅ Xem chi tiết parameters, request/response format
- ✅ Test API trực tiếp trong trình duyệt
- ✅ Tải OpenAPI spec (JSON) tại `/apispec.json`

**Cấu trúc Swagger:**
- `utils/swagger_config.py` - Cấu hình Swagger UI
- `utils/swagger_specs.py` - Tất cả Swagger specifications
- Tài liệu chi tiết: Xem `SWAGGER_DOCUMENTATION.md`

---

## 📡 API Endpoints

### 1. System Endpoints (4 endpoints)

**Base URL**: `http://localhost:5000`

#### `GET /`
Kiểm tra trạng thái server.

**Response:**
  ```json
  {
      "status": "Server is running",
      "version": "3.0 - PaddleOCR Edition",
      "ocr_engine": "PaddleOCR",
      "endpoints": [...]
  }
```

#### `GET /debug`
Thông tin debug chi tiết về routes và cấu hình.

#### `GET /api/performance`
Thông tin hiệu năng và GPU.

**Response:**
  ```json
  {
  "timestamp": "2025-10-03 10:30:00",
  "gpu_available": true,
  "gpu_info": {
    "gpu_device_id": 0,
    "gpu_name": "NVIDIA GeForce GTX 1050 Ti",
    "gpu_memory_total_gb": 4.0
  },
        "ocr": {
            "paddleocr_available": true,
            "engine": "PaddleOCR"
      }
  }
  ```

#### `GET /api/history`
Lấy lịch sử OCR với filtering.

**Query Parameters (Required):**
- `start_time`: Thời gian bắt đầu (YYYY-MM-DD hoặc YYYY-MM-DD HH:MM:SS)
- `end_time`: Thời gian kết thúc (YYYY-MM-DD hoặc YYYY-MM-DD HH:MM:SS)

**Query Parameters (Optional):**
- `machine_code`: Lọc theo mã máy (ví dụ: "IE-F1-CWA01")
- `area`: Lọc theo khu vực (ví dụ: "F1", "F4")
- `screen_id`: Lọc theo screen ID (ví dụ: "Production_Data")
- `limit`: Số lượng kết quả tối đa (mặc định: 100)

**Example:**
```bash
curl "http://localhost:5000/api/history?start_time=2025-11-01&end_time=2025-11-05&machine_code=IE-F1-CWA01&limit=50"
```

**Response:**
```json
{
  "history": [...],
  "count": 25,
  "limit": 50,
  "filters_applied": {
    "start_time": "2025-11-01",
    "end_time": "2025-11-05",
    "machine_code": "IE-F1-CWA01"
  }
}
```

---

### 2. Image Processing Endpoints (6 endpoints)

#### `POST /api/images`
Upload và xử lý ảnh HMI với PaddleOCR algorithm.

**Request (Form-data):**
- `file`: File ảnh (jpg, png, bmp) - **Required**
- `area`: Mã khu vực (ví dụ: "F1", "F4") - **Required**
- `machine_code`: Mã máy (ví dụ: "IE-F1-CWA01", "IE-F4-WBI01") - **Required**

**Processing Steps:**
1. Detect và extract HMI screen
2. Full image OCR với PaddleOCR
3. Match screen dựa trên Special_rois
4. Filter OCR results bằng IoU với ROIs
5. Post-process và format text
6. Deduplication (keep highest IOU)

**Example:**
```bash
curl -X POST http://localhost:5000/api/images \
  -F "file=@test_image.jpg" \
  -F "area=AREA1" \
  -F "machine_code=F41"
```

**Response:**
  ```json
  {
    "success": true,
    "filename": "1696320000_test_image.jpg",
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
        "confidence": 0.997,
        "has_text": true,
        "original_value": "14934",
        "iou": 0.449
      }
    ],
    "roi_count": 12,
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

#### `GET /api/images`
Lấy danh sách tất cả ảnh đã upload.

#### `GET /api/images/<filename>`
Lấy file ảnh cụ thể.

#### `DELETE /api/images/<filename>`
Xóa file ảnh.

#### `GET /api/images/hmi_detection/<filename>`
Lấy ảnh visualization kết quả phát hiện HMI.

#### `POST /api/images/ocr`
Alternative endpoint để thực hiện OCR với automatic screen detection và ROI matching.

**Request (Form-data):**
- `file`: File ảnh (jpg, png, bmp) - **Required**
- `area`: Mã khu vực (optional)
- `machine_code`: Mã máy (optional)

**Response:** Tương tự như `POST /api/images` nhưng với cấu trúc response đơn giản hơn.

---

### 3. Machine Management Endpoints (7 endpoints)

#### `GET /api/machines`
Lấy thông tin tất cả máy và khu vực.

#### `GET /api/machines/<area_code>`
Lấy danh sách máy theo khu vực.

**Example:**
```bash
curl http://localhost:5000/api/machines/AREA1
```

#### `GET /api/machine_screens/<machine_code>`
Lấy danh sách màn hình của một máy.

**Example:**
```bash
curl http://localhost:5000/api/machine_screens/F41
```

**Response:**
  ```json
  {
  "machine_code": "F41",
  "machine_type": "F41",
  "machine_name": "Máy ép F41",
  "screens": [
    {"id": 1, "screen_id": "Production", "description": "Màn hình sản xuất"},
    {"id": 2, "screen_id": "Temp", "description": "Màn hình nhiệt độ"}
  ]
}
```

#### `POST /api/set_machine_screen`
Đặt máy và màn hình hiện tại.

**Request (JSON):**
  ```json
  {
  "machine_code": "F41",
  "screen_id": "Production"
}
```

#### `POST /api/update_machine_screen`
Cập nhật máy và màn hình với parameter_order_value.txt.

**Request (Form-data):**
- `machine_code`: Mã máy
- `screen_id`: Tên màn hình
- `area`: Mã khu vực (optional)

#### `GET /api/current_machine_screen`
Lấy máy và màn hình hiện tại.

#### `GET /api/machine_screen_status`
Kiểm tra trạng thái cấu hình máy/màn hình.

---

### 4. Decimal Configuration Endpoints (6 endpoints)

#### `GET /api/decimal_places`
Lấy tất cả cấu hình số thập phân.

**Response:**
```json
{
  "F1": {
    "Production_Data": { "ROI_name": 0 },
    "Reject_Summary": {
      "ROI_0": 0,
      "IE-F1-CWA01": {
        "1": { "ST02_TESTED": 0 },
        "2": { "ST14_1_TESTED": 0 }
      }
    }
  }
}
```

#### `POST /api/decimal_places`
Cập nhật cấu hình số thập phân.

**Request (JSON):**
  ```json
  {
  "machine_code": "F41",
  "screen_id": "Production",
  "roi_config": {
    "Temperature": 1,
    "Pressure": 2,
    "Speed": 0
      }
  }
  ```

#### `GET /api/decimal_places/<machine_type>/<screen_name>` ⭐ UNIFIED API
Lấy cấu hình decimal places theo machine type và screen name.

**Path Parameters (BẮT BUỘC):**
- `machine_type`: Loại máy (F1, F41, F42)
- `screen_name`: Tên màn hình (Production_Data, Reject_Summary, Injection, etc.)

**Query Parameters (TÙY CHỌN):**
- `machine_code`: Mã máy (e.g., IE-F1-CWA01) - chỉ dùng cho Reject_Summary
- `sub_page`: Số trang con (1, 2) - chỉ dùng cho Reject_Summary với machine_code

**Examples:**
```bash
# 1. Lấy config cho screen thông thường
curl "http://localhost:5000/api/decimal_places/F41/Injection"

# 2. Lấy toàn bộ Reject_Summary
curl "http://localhost:5000/api/decimal_places/F1/Reject_Summary"

# 3. Lấy Reject_Summary cho máy cụ thể
curl "http://localhost:5000/api/decimal_places/F1/Reject_Summary?machine_code=IE-F1-CWA01"

# 4. Lấy Reject_Summary cho máy + sub-page
curl "http://localhost:5000/api/decimal_places/F1/Reject_Summary?machine_code=IE-F1-CWA01&sub_page=1"
```

**Response Example (Standard screen):**
```json
{
  "machine_type": "F41",
  "screen_name": "Injection",
  "decimal_config": {
    "Injection speed": 1,
    "Charge speed": 1
  }
}
```

**Response Example (Reject_Summary với machine_code và sub_page):**
```json
{
  "machine_type": "F1",
  "screen_name": "Reject_Summary",
  "machine_code": "IE-F1-CWA01",
  "sub_page": "1",
  "decimal_config": {
    "ST02_TESTED": 0,
    "ST02_REJECTS": 0,
    "ST02_PERCENT": 2
  }
}
```

#### `POST /api/decimal_places/<machine_type>/<screen_name>` ⭐ UNIFIED API
Cập nhật cấu hình decimal places.

**Path Parameters (BẮT BUỘC):**
- `machine_type`: Loại máy (F1, F41, F42)
- `screen_name`: Tên màn hình

**Query Parameters (TÙY CHỌN):**
- `machine_code`: Mã máy - chỉ dùng cho Reject_Summary
- `sub_page`: Số trang con - chỉ dùng cho Reject_Summary với machine_code

**Examples:**
```bash
# 1. Update screen thông thường
curl -X POST "http://localhost:5000/api/decimal_places/F41/Injection" \
  -H "Content-Type: application/json" \
  -d '{"Injection speed": 1, "Charge speed": 1}'

# 2. Update Reject_Summary cho máy (tất cả sub-pages)
curl -X POST "http://localhost:5000/api/decimal_places/F1/Reject_Summary?machine_code=IE-F1-CWA01" \
  -H "Content-Type: application/json" \
  -d '{"1": {"ST02_TESTED": 0}, "2": {"ST14_1_TESTED": 0}}'

# 3. Update Reject_Summary cho máy + sub-page
curl -X POST "http://localhost:5000/api/decimal_places/F1/Reject_Summary?machine_code=IE-F1-CWA01&sub_page=1" \
  -H "Content-Type: application/json" \
  -d '{"ST02_TESTED": 0, "ST02_REJECTS": 0}'
```

#### `POST /api/set_decimal_value`
Đặt giá trị số thập phân cho ROI đơn lẻ.

**Request (JSON):**
```json
{
  "machine_code": "F41",
  "screen_id": "Production",
  "roi_index": "Temperature",
  "decimal_places": 1
}
```

#### `POST /api/set_all_decimal_values`
Đặt tất cả giá trị số thập phân cho màn hình.

**Request (JSON):**
```json
{
  "machine_code": "F41",
  "screen_id": "Production",
  "decimal_config": {
    "Temperature": 1,
    "Pressure": 2
  }
}
```

---

### 5. Reference Images Endpoints (4 endpoints)

#### `POST /api/reference_images`
Upload ảnh template tham chiếu.

**Request (Form-data):**
- `file`: File ảnh template
- `machine_type`: Loại máy (F1, F41, F42)
- `screen_id`: ID màn hình

#### `GET /api/reference_images`
Lấy danh sách ảnh template.

#### `GET /api/reference_images/<filename>`
Lấy file ảnh template cụ thể.

#### `DELETE /api/reference_images/<filename>`
Xóa ảnh template.

---

## ⚙️ Cấu Hình

### File Cấu Hình Quan Trọng

#### 1. `roi_data/machine_screens.json`
Cấu hình máy móc và màn hình.

**Cấu trúc:**
  ```json
  {
  "areas": {
    "AREA1": {
      "name": "Khu vực 1",
      "machines": {
        "F41": {
          "type": "F41",
          "name": "Máy ép F41",
          "description": "..."
        }
      }
    }
  },
  "machine_types": {
    "F41": {
      "screens": [
        {
          "id": 1,
          "screen_id": "Production",
          "description": "Màn hình sản xuất"
        }
      ]
    }
      }
  }
  ```

#### 2. `roi_data/roi_info.json`
Tọa độ ROI cho từng màn hình.

**Cấu trúc:**
  ```json
  {
  "machines": {
    "F41": {
      "screens": {
        "Production": [
          {
            "name": "Temperature",
            "coordinates": [100, 200, 300, 250],
            "allowed_values": []
          }
        ]
      }
    }
  }
}
```

**Lưu ý**: Tọa độ có thể là:
- **Pixel tuyệt đối**: [x1, y1, x2, y2] (số nguyên)
- **Normalized**: [0.1, 0.2, 0.3, 0.4] (số thập phân 0-1)

#### 3. `roi_data/decimal_places.json`
Cấu hình số chữ số thập phân.

  ```json
  {
  "F41": {
    "Production": {
      "Temperature": 1,
      "Pressure": 2,
      "Speed": 0
    }
      }
  }
  ```

### Thay Đổi Port

Mặc định server chạy trên port 5000. Để thay đổi:

**File `app.py`** (dòng 248):
```python
app.run(host='0.0.0.0', port=5001, debug=True)  # Đổi 5000 thành 5001
```

**File `wsgi.py`**:
```python
httpd = make_server('0.0.0.0', 5001, app)  # Đổi 5000 thành 5001
```

---

## 🔥 Khắc Phục Sự Cố

### Vấn Đề 1: Import Error

**Lỗi:**
```
ModuleNotFoundError: No module named 'flask'
```

**Giải pháp:**
```bash
pip install -r requirements.txt
```

### Vấn Đề 2: GPU Không Phát Hiện

**Lỗi:**
```
[WARNING] GPU not available
```

**Kiểm tra:**
```bash
nvidia-smi
python -c "import paddle; print('GPU:', paddle.device.is_compiled_with_cuda())"
```

**Giải pháp:**
1. Kiểm tra NVIDIA driver:
   ```bash
   nvidia-smi
   ```
2. Cài đặt PaddlePaddle GPU với CUDA 11.8:
   ```bash
   pip uninstall paddlepaddle paddlepaddle-gpu -y
   pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
   ```

### Vấn Đề 3: CUDA DLL Conflicts (PyTorch vs PaddlePaddle)

**Lỗi:**
```
Error loading "cusparse64_12.dll" or one of its dependencies
```

**Nguyên nhân:** PyTorch với CUDA 12 xung đột với PaddlePaddle CUDA 11.8

**Giải pháp:**
```bash
# Gỡ bỏ PyTorch và CUDA 12 packages
pip uninstall torch torchvision -y
pip uninstall nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-nvjitlink-cu12 -y

# Cài lại PaddlePaddle GPU
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```

### Vấn Đề 4: PaddleOCR Circular Import Error

**Lỗi:**
```
AttributeError: partially initialized module 'paddle' has no attribute 'tensor' (most likely due to a circular import)
```

**Nguyên nhân:** Flask debug mode với reloader gây ra circular import

**Giải pháp:**
Đảm bảo `app.py` chạy với `debug=False`:
```python
# Trong app.py
app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
```

### Vấn Đề 5: PaddleOCR Lỗi Khởi Tạo

**Lỗi:**
```
RuntimeError: PDX has already been initialized. Reinitialization is not supported.
```

**Giải pháp:**
1. Restart server hoàn toàn
2. Đảm bảo không gọi `get_paddleocr_instance()` nhiều lần trong cùng process
3. Nếu vẫn lỗi:
   ```bash
   # Khởi động lại PowerShell/Terminal
   # Rồi chạy lại
   python app.py
   ```

### Vấn Đề 6: NumPy Compatibility Error

**Lỗi:**
```
ImportError: A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.4.1 as it may crash.
```

**Nguyên nhân:** NumPy 2.x không tương thích với scipy, scikit-image, và các thư viện được compile với NumPy 1.x.

**Giải pháp:**
```bash
# Downgrade NumPy về version 1.x
pip install "numpy<2.0" --force-reinstall

# Kiểm tra version
python -c "import numpy; print(numpy.__version__)"
# Phải là: 1.26.4 hoặc tương tự (< 2.0)
```

### Vấn Đề 7: OCR Trả Về 0 Kết Quả

**Hiện tượng:**
```json
{
  "ocr_results": [],
  "roi_count": 0,
  "screen_id": null
}
```

**Debug:** Xem terminal logs khi xử lý request:
```
[DEBUG] PaddleOCR results type: <class 'list'>
[DEBUG] extract_ocr_data: Detected PaddleOCR 2.x format
[DEBUG] OCR items with bbox: 0/0
```

**Giải pháp:**
1. **Kiểm tra PaddleOCR result format:**
   - PaddleOCR 2.7.3 sử dụng list format `[[box, (text, score)], ...]`
   - Code tự động detect format và parse đúng
   
2. **Kiểm tra HMI detection:**
   - Xem log `[OK] HMI extracted: WxH`
   - Nếu không extract được → ảnh không phát hiện được màn hình
   
3. **Kiểm tra image quality:**
   - Ảnh bị mờ, quá tối, hoặc quá sáng
   - Thử với ảnh khác chất lượng cao hơn

4. **Kiểm tra NumPy version:**
   ```bash
   python -c "import numpy; print(numpy.__version__)"
   # Phải là < 2.0
   ```

5. **Restart server nếu warm-up failed:**
   ```bash
   # Kill process cũ và restart
   python app.py
   ```

### Vấn Đề 8: Screen Matching Thất Bại

**Hiện tượng:**
```
[WARNING] No screen matched. OCR data count: 0
```

**Giải pháp:**
1. Kiểm tra `roi_info.json` có đúng cấu trúc cho machine_type/machine_code
2. Kiểm tra `machine_screens.json` có mapping đúng machine_code → machine_type
3. Đảm bảo Special_rois trong roi_info.json khớp với text trên màn hình HMI

### Vấn Đề 9: Port Đang Được Sử Dụng

**Lỗi:**
```
OSError: [WinError 10048] Only one usage of each socket address
```

**Giải pháp:**
1. Tìm process đang dùng port 5000:
   ```bash
   netstat -ano | findstr :5000
   ```
2. Kill process:
   ```bash
   taskkill /PID <process_id> /F
   ```
3. Hoặc đổi port trong `app.py`

### Vấn Đề 10: Out of Memory (GPU)

**Lỗi:**
```
RuntimeError: CUDA out of memory
```

**Giải pháp:**
1. Giảm batch size trong code
2. Xử lý ít ảnh hơn cùng lúc
3. Restart server để clear GPU memory

### Vấn Đề 11: Slow Performance

**Hiện tượng:** Server chạy chậm

**Kiểm tra:**
```bash
curl http://localhost:5000/api/performance
```

**Giải pháp:**
1. Kiểm tra GPU có đang hoạt động không
2. Kiểm tra thread pool: Nên thấy "24 workers"
3. Clear cache:
   ```bash
   # Xóa cache trong code
   # Hoặc restart server
   ```

### Vấn Đề 12: Template Not Found

**Lỗi:**
```
Template not found for machine X screen Y
```

**Giải pháp:**
1. Kiểm tra file template trong `roi_data/reference_images/`
2. Tên file phải đúng format: `template_{machine_type}_{screen_id}.jpg`
3. Upload template mới qua API:
   ```bash
   curl -X POST http://localhost:5000/api/reference_images \
     -F "file=@template.jpg" \
     -F "machine_type=F41" \
     -F "screen_id=Production"
   ```

---

## 🛠️ Bảo Trì

### Backup Dữ Liệu

**Các thư mục cần backup định kỳ:**
```
roi_data/                    # Cấu hình
uploads/                     # Ảnh đã xử lý (tùy chọn)
ocr_results/                 # Kết quả OCR (tùy chọn)
augmented_training_data/     # Dữ liệu training (quan trọng!)
```

**Script backup tự động:**
```bash
# Tạo file backup_data.bat
@echo off
set BACKUP_DIR=D:\Backups\HMI_OCR_%date:~-4,4%%date:~-7,2%%date:~-10,2%
mkdir "%BACKUP_DIR%"
xcopy /E /I /Y "roi_data" "%BACKUP_DIR%\roi_data"
xcopy /E /I /Y "augmented_training_data" "%BACKUP_DIR%\training_data"
echo Backup completed: %BACKUP_DIR%
```

### Update Dependencies

```bash
# Xem packages outdated
pip list --outdated

# Update một package cụ thể
pip install --upgrade flask

# Hoặc update tất cả (cẩn thận!)
pip install --upgrade -r requirements.txt
```

### Logs và Monitoring

**Xem logs:**
- Server logs: Output terminal
- OCR results: `ocr_results/` folder
- Performance: `GET /api/performance`

**Monitoring checklist:**
- [ ] GPU memory usage
- [ ] CPU usage
- [ ] Disk space
- [ ] Response time
- [ ] Error rate

### Clear Cache

**Xóa uploaded images cũ:**
```bash
cd uploads
del /Q *.jpg *.png
cd aligned
del /Q *.*
```

**Xóa OCR results cũ:**
```bash
cd ocr_results
del /Q *.json
```

**Lưu ý:** Không xóa các thư mục training data!

---


