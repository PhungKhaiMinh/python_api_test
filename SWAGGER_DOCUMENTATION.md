# Swagger UI Documentation

## Tổng quan

Hệ thống HMI OCR API đã được tích hợp **Swagger UI** (OpenAPI 2.0) để cung cấp giao diện tương tác với API trực quan và dễ sử dụng.

## Cài đặt

### 1. Cài đặt thư viện Flasgger

```bash
pip install flasgger
```

Hoặc cài đặt tất cả dependencies:

```bash
pip install -r requirements.txt
```

### 2. Khởi động server

```bash
python python_api_test/app.py
```

### 3. Truy cập Swagger UI

Mở trình duyệt và truy cập:

```
http://localhost:5000/apidocs
```

## Cấu trúc module Swagger

### 1. `utils/swagger_config.py`

File cấu hình chính cho Swagger UI:
- `get_swagger_config()`: Cấu hình Swagger UI (đường dẫn, static files)
- `get_swagger_template()`: Template cơ bản (title, description, tags)
- `init_swagger(app)`: Khởi tạo Swagger cho Flask app

### 2. `utils/swagger_specs.py`

File chứa tất cả Swagger specifications cho từng API endpoint:
- Mỗi function trả về một docstring với format Swagger/OpenAPI
- Dễ dàng chỉnh sửa và bảo trì
- Tách biệt hoàn toàn khỏi logic business

**Ví dụ:**

```python
def get_upload_image_spec():
    """Swagger spec for POST /api/images"""
    return """
    Upload image và thực hiện OCR
    ---
    tags:
      - Image Processing
    parameters:
      - name: file
        in: formData
        type: file
        required: true
    responses:
      200:
        description: Successfully processed image
    """
```

### 3. Swagger Injection trong `app.py`

Swagger documentation được inject trực tiếp trong `app.py`:
- Được gọi tự động khi server khởi động
- Inject docstrings từ `swagger_specs.py` vào tất cả route functions
- Xử lý tất cả routes: system, image, machine, decimal, reference

## Danh sách API endpoints được document

### System (4 endpoints)
- `GET /` - Server status
- `GET /debug` - Debug information
- `GET /api/performance` - Performance statistics
- `GET /api/history` - OCR history with filters

### Image Processing (6 endpoints)
- `POST /api/images` - Upload và OCR (PaddleOCR algorithm)
- `POST /api/images/ocr` - Alternative OCR endpoint
- `GET /api/images` - List images
- `GET /api/images/<filename>` - Get image
- `DELETE /api/images/<filename>` - Delete image
- `GET /api/images/hmi_detection/<filename>` - HMI detection image

### Decimal Places (6 endpoints) ⭐ UNIFIED API
- `GET /api/decimal_places` - All config
- `POST /api/decimal_places` - Update config
- `GET /api/decimal_places/<machine_type>/<screen_name>` - ⭐ UNIFIED GET (supports query params)
- `POST /api/decimal_places/<machine_type>/<screen_name>` - ⭐ UNIFIED POST (supports query params)
- `POST /api/set_decimal_value` - Set single value
- `POST /api/set_all_decimal_values` - Set all values

**Note**: Unified API `/api/decimal_places/<machine_type>/<screen_name>` hỗ trợ:
- Standard screens: `GET /api/decimal_places/F41/Injection`
- Reject_Summary với machine_code: `GET /api/decimal_places/F1/Reject_Summary?machine_code=IE-F1-CWA01`
- Reject_Summary với machine_code + sub_page: `GET /api/decimal_places/F1/Reject_Summary?machine_code=IE-F1-CWA01&sub_page=1`

### Machine Management (7 endpoints)
- `GET /api/machines` - All machines
- `GET /api/machines/<area_code>` - By area
- `GET /api/machine_screens/<machine_code>` - Machine screens
- `POST /api/set_machine_screen` - Set current screen
- `GET /api/current_machine_screen` - Get current
- `GET /api/machine_screen_status` - Get status
- `POST /api/update_machine_screen` - Update screen

### Reference Images (4 endpoints)
- `POST /api/reference_images` - Upload reference
- `GET /api/reference_images` - List references
- `GET /api/reference_images/<filename>` - Get reference
- `DELETE /api/reference_images/<filename>` - Delete reference

**Tổng cộng: 27 API endpoints** (v3.0 PaddleOCR Edition)

## Cách sử dụng Swagger UI

### 1. Xem danh sách API

- Truy cập `/apidocs`
- Các API được phân nhóm theo tags (Image Processing, Decimal Places, ...)
- Click vào tag để xem chi tiết

### 2. Xem chi tiết API

- Click vào endpoint để xem:
  - Parameters (query, path, body)
  - Request format
  - Response format
  - Example values

### 3. Test API trực tiếp

1. Click nút **"Try it out"**
2. Nhập parameters cần thiết
3. Click **"Execute"**
4. Xem response ngay trong UI

### 4. Tải API spec

- Truy cập: `http://localhost:5000/apispec.json`
- Tải về file JSON để import vào Postman hoặc công cụ khác

## Cách thêm/sửa documentation cho API mới

### Bước 1: Thêm spec function vào `swagger_specs.py`

```python
def get_my_new_api_spec():
    """Swagger spec for GET /api/my_new_endpoint"""
    return """
    My new API description
    ---
    tags:
      - My Category
    parameters:
      - name: param1
        in: query
        type: string
        required: true
    responses:
      200:
        description: Success
    """
```

### Bước 2: Inject vào route function

**Cách 1: Trong route file**

```python
@my_bp.route('/api/my_new_endpoint', methods=['GET'])
def my_new_endpoint():
    """My new API"""
    try:
        from utils.swagger_specs import get_my_new_api_spec
        my_new_endpoint.__doc__ = get_my_new_api_spec().strip()
    except:
        pass
    
    # Your logic here
    return jsonify({"status": "ok"})
```

**Cách 2: Trong `inject_swagger_docs.py`** (khuyến nghị)

```python
def inject_all_swagger_docs():
    # ...
    my_routes.my_new_endpoint.__doc__ = swagger_specs.get_my_new_api_spec().strip()
```

### Bước 3: Khởi động lại server

```bash
python python_api_test/app.py
```

## Format Swagger Specification

### Basic structure

```yaml
---
tags:
  - Category Name
summary: Short description
description: Long description (optional)
parameters:
  - name: param_name
    in: query|path|formData|body
    type: string|integer|file|object
    required: true|false
    description: Parameter description
    example: "example_value"
responses:
  200:
    description: Success description
    schema:
      type: object
      properties:
        field1:
          type: string
  400:
    description: Bad request
```

### File upload

```yaml
consumes:
  - multipart/form-data
parameters:
  - name: file
    in: formData
    type: file
    required: true
```

### JSON body

```yaml
parameters:
  - name: body
    in: body
    required: true
    schema:
      type: object
      required:
        - field1
      properties:
        field1:
          type: string
```

## Troubleshooting

### 1. Swagger UI không hiển thị

**Kiểm tra:**
- Flasgger đã được cài đặt chưa?
- Server có in thông báo `[OK] Swagger UI initialized` không?
- Truy cập đúng URL: `http://localhost:5000/apidocs`

### 2. API không xuất hiện trong Swagger

**Kiểm tra:**
- Route có được register vào Blueprint chưa?
- Blueprint có được register vào app chưa?
- Docstring có format đúng không? (phải có `---`)

### 3. API xuất hiện nhưng thiếu thông tin

**Kiểm tra:**
- Docstring injection có chạy thành công không?
- Check console log: `[OK] Swagger documentation injected`
- Spec function trong `swagger_specs.py` có đúng format không?

### 4. Lỗi khi Execute API

**Kiểm tra:**
- Parameters có đúng type không?
- Required parameters đã được nhập chưa?
- Server có đang chạy không?

## Lợi ích của cấu trúc module riêng

### 1. Dễ bảo trì
- Tất cả Swagger specs tập trung trong một file
- Không làm rối logic business code
- Dễ tìm kiếm và chỉnh sửa

### 2. Tái sử dụng
- Có thể dùng lại specs cho documentation khác
- Export ra OpenAPI JSON để dùng với các công cụ khác

### 3. Tách biệt concerns
- Logic business: Trong route files
- API documentation: Trong swagger_specs.py
- Configuration: Trong swagger_config.py

### 4. Version control
- Dễ theo dõi thay đổi documentation
- Conflict ít hơn khi nhiều người làm việc

## Best Practices

1. **Luôn document tất cả endpoints**
   - Bao gồm cả error responses
   - Cung cấp examples cho parameters

2. **Sử dụng tags hợp lý**
   - Nhóm APIs liên quan vào cùng tag
   - Tên tag rõ ràng, dễ hiểu

3. **Mô tả chi tiết parameters**
   - Type chính xác
   - Required hay optional
   - Example values

4. **Update documentation khi thay đổi API**
   - Đảm bảo docs luôn sync với code
   - Test lại sau mỗi lần sửa

5. **Sử dụng Response schemas**
   - Định nghĩa rõ structure của response
   - Giúp frontend developers hiểu dễ hơn

## Kết luận

Swagger UI giúp:
- Tương tác với API dễ dàng hơn
- Testing nhanh chóng
- Documentation tự động và chính xác
- Giảm thời gian onboarding cho developers mới

Với cấu trúc module riêng, việc bảo trì và mở rộng trở nên đơn giản và hiệu quả hơn.

