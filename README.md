# HMI OCR API Server - Hướng Dẫn Đầy Đủ

**Phiên bản:** 2.0 - Refactored  
**Cập nhật:** November 5, 2025  
**Trạng thái:** ✅ Sẵn sàng Production

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

✅ **Tự động phát hiện màn hình**: Thuật toán Two-Stage Enhanced Detection v1.0  
✅ **OCR GPU-accelerated**: EasyOCR với CUDA  
✅ **Căn chỉnh ảnh tự động**: SIFT-based perspective correction  
✅ **Xử lý song song**: Multi-threading với 24 workers  
✅ **Cache thông minh**: Template và configuration caching  
✅ **RESTful API**: 30 API endpoints đầy đủ tính năng  
✅ **Swagger UI**: Tài liệu API tương tác tại `/apidocs`  
✅ **Sub-page Detection**: Hỗ trợ Reject_Summary với nhiều sub-pages  
✅ **History Filtering**: Lọc lịch sử OCR theo machine_code, area, screen_id, time range

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

### Bước 5: Cài Đặt PyTorch với CUDA

**Quan trọng**: PyTorch cần cài đặt đúng phiên bản CUDA.

```bash
# Gỡ cài đặt cũ (nếu có)
pip uninstall torch torchvision -y

# Cài đặt với CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Hoặc cho CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Kiểm tra PyTorch CUDA**:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
```

Kết quả mong đợi:
```
CUDA available: True
CUDA version: 12.1
```

### Bước 6: Kiểm Tra Cài Đặt

```bash
python -c "from app import app; print('[OK] All imports successful')"
```

Nếu thành công, bạn sẽ thấy:
```
[OK] CuPy GPU acceleration available
[OK] GPU Accelerator loaded
[OK] EasyOCR initialized with GPU
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
│   ├── image_processor.py         # Xử lý ảnh (529 dòng)
│   └── ocr_processor.py           # Xử lý OCR (413 dòng)
│
├── routes/                         # API route blueprints
│   ├── __init__.py                # Export routes
│   ├── image_routes.py            # Routes xử lý ảnh (202 dòng)
│   ├── machine_routes.py          # Routes quản lý máy (254 dòng)
│   ├── decimal_routes.py          # Routes cấu hình số thập phân (227 dòng)
│   └── reference_routes.py        # Routes ảnh tham chiếu (144 dòng)
│
├── Core modules (Không sửa)
│   ├── smart_detection_functions.py   # Thuật toán phát hiện màn hình
│   ├── gpu_accelerator.py             # GPU acceleration
│   ├── parallel_processor.py          # Xử lý song song
│   ├── ensemble_hog_orb_classifier.py # ML classifier
│   └── hog_svm_classifier.py          # ML classifier backup
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
[OK] GPU Accelerator và Parallel Processor modules loaded
[OK] EasyOCR initialized with GPU
[OK] SIFT detector initialized
[OK] Enhanced thread pools initialized
[OK] GPU Accelerator ready

======================================================================
HMI OCR API SERVER - REFACTORED v2.0
======================================================================
Upload folder: D:\python_WREMBLY_test-main\python_api_test\uploads
ROI data folder: D:\python_WREMBLY_test-main\python_api_test\roi_data
GPU available: True
EasyOCR available: True
======================================================================
Starting server on http://0.0.0.0:5000
======================================================================

 * Serving Flask app 'app'
 * Debug mode: on
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
```

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
- ✅ Xem tất cả 30 API endpoints được phân loại theo tags
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

#### `GET /`
Kiểm tra trạng thái server.

**Response:**
  ```json
  {
      "status": "Server is running",
  "version": "2.0 - Refactored",
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
    "easyocr_available": true,
    "gpu_enabled": true
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

### 2. Image Processing Endpoints (5 endpoints)

#### `POST /api/images`
Upload và xử lý ảnh HMI.

**Request (Form-data):**
- `file`: File ảnh (jpg, png, bmp)
- `area`: Mã khu vực (ví dụ: "AREA1")
- `machine_code`: Mã máy (ví dụ: "F41")

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
  "machine_code": "F41",
  "machine_type": "F41",
  "screen_id": "Production",
  "detection_method": "two_stage_enhanced",
  "similarity_score": 0.95,
  "ocr_results": [
    {
      "roi_index": "Temperature",
      "text": "245.5",
      "confidence": 0.98,
      "has_text": true
    }
  ],
  "roi_count": 12
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

### 4. Decimal Configuration Endpoints (10 endpoints)

#### `GET /api/decimal_places`
Lấy tất cả cấu hình số thập phân.

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

#### `GET /api/decimal_places/<machine_code>`
Lấy cấu hình theo máy.

#### `GET /api/decimal_places/<machine_code>/<screen_name>`
Lấy cấu hình theo màn hình.

#### `POST /api/decimal_places/<machine_code>/<screen_name>`
Cập nhật cấu hình cho màn hình cụ thể.

#### `POST /api/set_decimal_value`
Đặt giá trị số thập phân cho ROI đơn lẻ.

#### `POST /api/set_all_decimal_values`
Đặt tất cả giá trị số thập phân cho màn hình.

#### `GET /api/decimal_places/<machine_type>/Reject_Summary/<machine_code>`
Lấy decimal places cho tất cả sub-pages của một machine_code trong Reject_Summary.

**Example:**
```bash
curl "http://localhost:5000/api/decimal_places/F1/Reject_Summary/IE-F1-CWA01"
```

#### `GET /api/decimal_places/<machine_type>/Reject_Summary/<machine_code>/<sub_page>`
Lấy decimal places cho một sub-page cụ thể của Reject_Summary.

**Example:**
```bash
curl "http://localhost:5000/api/decimal_places/F1/Reject_Summary/IE-F1-CWA01/1"
```

#### `POST /api/decimal_places/<machine_type>/Reject_Summary/<machine_code>/<sub_page>`
Cập nhật decimal places cho một sub-page cụ thể của Reject_Summary.

**Request (JSON):**
```json
{
  "tested": 0,
  "reject": 0,
  "phantram": 0
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
python -c "import torch; print(torch.cuda.is_available())"
```

**Giải pháp:**
1. Cài đặt lại CUDA Toolkit
2. Cài đặt lại CuPy với đúng phiên bản CUDA:
   ```bash
   pip uninstall cupy -y
   pip install cupy-cuda12x  # Cho CUDA 12.x
   ```
3. Cài đặt lại PyTorch với CUDA:
   ```bash
   pip uninstall torch torchvision -y
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

### Vấn Đề 3: EasyOCR Lỗi

**Lỗi:**
```
Exception: EasyOCR failed to initialize
```

**Giải pháp:**
```bash
pip uninstall easyocr -y
pip install easyocr==1.7.2
```

Nếu vẫn lỗi, xóa cache:
```bash
# Windows
rmdir /s /q %USERPROFILE%\.EasyOCR
```

### Vấn Đề 4: Port Đang Được Sử Dụng

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

### Vấn Đề 5: Out of Memory (GPU)

**Lỗi:**
```
RuntimeError: CUDA out of memory
```

**Giải pháp:**
1. Giảm batch size trong code
2. Xử lý ít ảnh hơn cùng lúc
3. Restart server để clear GPU memory

### Vấn Đề 6: Slow Performance

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

### Vấn Đề 7: Template Not Found

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


