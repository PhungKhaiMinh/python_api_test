# Python API Test

## Giới thiệu
Python API Test là một ứng dụng web API được xây dựng bằng Flask để xử lý và trích xuất thông tin từ hình ảnh màn hình HMI (Human-Machine Interface). Ứng dụng cung cấp chức năng OCR (Optical Character Recognition) để nhận dạng và trích xuất các số từ vùng quan tâm (ROI) được định nghĩa trước trên màn hình.

## Tính năng chính
- Tải lên và quản lý hình ảnh màn hình HMI
- Xác định vùng quan tâm (ROI) trong hình ảnh
- Thực hiện OCR để trích xuất số từ các vùng đã xác định
- Quản lý cấu hình máy và màn hình
- Quản lý cấu hình số thập phân cho các giá trị được trích xuất

## Cài đặt và thiết lập cho Windows 11

### Yêu cầu hệ thống
- Windows 11 với quyền quản trị viên
- Python 3.7 trở lên
- Pip (trình quản lý gói Python)
- Các thư viện: Flask, OpenCV, NumPy, EasyOCR, Scikit-image

### Cài đặt trên Windows 11 với IIS

#### 1. Cài đặt Python
1. Tải Python từ trang chủ: https://www.python.org/downloads/
2. Chọn phiên bản Python 3.7 trở lên (khuyến nghị là Python 3.10 cho Windows 11)
3. Trong quá trình cài đặt, đảm bảo đánh dấu các tùy chọn:
   - "Add Python to PATH" (quan trọng)
   - "Install for all users" (nếu bạn muốn tất cả người dùng đều có thể sử dụng)
   - "Install pip" (đã được chọn mặc định)
4. Kiểm tra cài đặt Python thành công bằng cách mở Windows Terminal (nhấn tổ hợp Windows + X, chọn "Terminal" hoặc "Windows Terminal") và nhập:
   ```
   python --version
   pip --version
   ```

#### 2. Cài đặt IIS trên Windows 11
1. Nhấp chuột phải vào nút Start Windows, chọn "Settings" (Cài đặt)
2. Trong cài đặt, chọn "Apps" > "Optional features" > "More Windows features" (hoặc "Turn Windows features on or off")
3. Trong cửa sổ Windows Features, tìm và đánh dấu chọn:
   - "Internet Information Services"
   - Mở rộng "Internet Information Services" và đảm bảo các tính năng sau được chọn:
     - Web Management Tools > IIS Management Console
     - World Wide Web Services > Application Development Features > CGI
4. Nhấn OK và đợi Windows 11 hoàn tất việc cài đặt các tính năng
5. **Quan trọng**: Sau khi cài đặt, khởi động lại máy tính để đảm bảo mọi thay đổi được áp dụng

#### 3. Cài đặt URL Rewrite Module cho IIS
1. Tải URL Rewrite Module từ trang web Microsoft: https://www.iis.net/downloads/microsoft/url-rewrite
2. Chạy tệp cài đặt với quyền Administrator (nhấp chuột phải vào tệp, chọn "Run as administrator")
3. Làm theo hướng dẫn trên màn hình để hoàn tất cài đặt

#### 4. Cài đặt WFASTCGI
1. Mở Windows Terminal với quyền Administrator (nhấp chuột phải vào biểu tượng Windows Terminal trong Start menu, chọn "Run as administrator")
2. Nhập lệnh sau để cài đặt wfastcgi:
   ```
   pip install wfastcgi
   ```
3. Kích hoạt wfastcgi bằng lệnh:
   ```
   wfastcgi-enable
   ```
4. Sau khi chạy lệnh, bạn sẽ thấy một thông báo kèm theo đường dẫn Python. Sao chép toàn bộ đường dẫn này (ví dụ: "C:\Program Files\Python310\python.exe|C:\Program Files\Python310\Lib\site-packages\wfastcgi.py") và lưu lại, vì bạn sẽ cần nó trong các bước sau.

#### 5. Cài đặt và Cấu hình Ứng dụng
1. Tải hoặc clone repository về máy tính
2. Đặt thư mục project vào đường dẫn dễ truy cập, ví dụ: `C:\python_api_test`
3. Mở Windows Terminal với quyền Administrator, điều hướng đến thư mục project:
   ```
   cd C:\python_api_test
   ```
4. Cài đặt các thư viện phụ thuộc:
   ```
   pip install -r requirements.txt
   ```
   Lưu ý: Quá trình này có thể mất 10-15 phút tùy thuộc vào tốc độ mạng và cấu hình máy tính của bạn. EasyOCR là một gói lớn với nhiều dependencies.

#### 6. Cấu hình IIS trên Windows 11
1. Mở IIS Manager bằng một trong các cách sau:
   - Cách 1: Nhấn tổ hợp phím Windows + S (hoặc nhấp vào biểu tượng Search), tìm kiếm "IIS" hoặc "Internet Information Services" và chọn "Internet Information Services (IIS) Manager"
   - Cách 2: Mở Windows Terminal với quyền Administrator và nhập lệnh: `start inetmgr`
   - Cách 3: Nhấn tổ hợp phím Windows + R, gõ "control panel" và nhấn Enter, sau đó điều hướng đến System and Security > Administrative Tools > Internet Information Services (IIS) Manager

2. Trong cửa sổ IIS Manager:
   - Ở panel bên trái, mở rộng server của bạn, nhấp chuột phải vào "Sites" > "Add Website"
   - Nhập thông tin:
     - Site name: Python_API_Test
     - Physical path: đường dẫn đến thư mục project (ví dụ: C:\python_api_test)
     - Binding: Type = http, IP address = All Unassigned, Port = 5000 (hoặc cổng khác nếu 5000 đã được sử dụng)
     - Host name: để trống
   - Nhấn OK

3. Chọn trang web vừa tạo (Python_API_Test) trong panel bên trái
4. Nhấp đúp vào "Handler Mappings" trong view chính
5. Trong view Handler Mappings, nhấp vào "Add Module Mapping..." ở panel bên phải
6. Điền thông tin sau:
   - Request path: *
   - Module: FastCgiModule
   - Executable: Dán đường dẫn bạn đã sao chép từ bước 4.4 
     (Ví dụ: C:\Program Files\Python310\python.exe|C:\Program Files\Python310\Lib\site-packages\wfastcgi.py)
     Lưu ý: Thay phần sau dấu | thành đường dẫn đến file wsgi.py trong thư mục dự án của bạn
   - Name: FlaskHandler
7. Nhấn OK và chọn "Yes" khi được hỏi về việc tạo FastCGI application

#### 7. Kiểm tra và cập nhật web.config
1. Điều hướng đến thư mục dự án (ví dụ: C:\python_api_test)
2. Nếu đã có sẵn file web.config, mở nó bằng Notepad hoặc editor khác để kiểm tra
3. Nếu chưa có hoặc cần cập nhật, tạo/chỉnh sửa file web.config với nội dung sau:
   ```xml
   <?xml version="1.0" encoding="UTF-8"?>
   <configuration>
     <system.webServer>
       <handlers>
         <add name="FlaskHandler" path="*" verb="*" modules="FastCgiModule" 
              scriptProcessor="[Đường dẫn Python và wfastcgi đã lưu ở bước 4.4]" 
              resourceType="Unspecified" requireAccess="Script" />
       </handlers>
       <rewrite>
         <rules>
           <rule name="Static Files" stopProcessing="true">
             <match url="^static/.*" ignoreCase="true" />
             <action type="Rewrite" url="{R:0}" />
           </rule>
           <rule name="Flask Application" stopProcessing="true">
             <match url="(.*)" ignoreCase="true" />
             <action type="Rewrite" url="wsgi.py" />
           </rule>
         </rules>
       </rewrite>
     </system.webServer>
     <appSettings>
       <add key="WSGI_HANDLER" value="app.app" />
       <add key="PYTHONPATH" value="[Đường dẫn đến thư mục project]" />
     </appSettings>
   </configuration>
   ```
   Thay thế các phần trong ngoặc vuông bằng đường dẫn thực tế trên máy tính của bạn.

#### 8. Cấu hình Quyền truy cập thư mục trên Windows 11
1. Mở File Explorer, điều hướng đến thư mục project (ví dụ: C:\python_api_test)
2. Nhấp chuột phải vào thư mục > chọn "Properties"
3. Chuyển đến tab "Security"
4. Nhấp vào "Edit..." > "Add..."
5. Trong hộp thoại "Select Users or Groups", nhập "IIS_IUSRS" vào ô "Enter the object names to select", nhấp "Check Names" để xác nhận, sau đó nhấp "OK"
6. Chọn nhóm "IIS_IUSRS" vừa thêm và đánh dấu cho phép các quyền:
   - Read & Execute
   - List folder contents
   - Read
7. Nhấn "Apply" và "OK" để lưu thay đổi
8. Lặp lại quy trình tương tự cho các thư mục con: uploads, roi_data, ocr_results để đảm bảo IIS có quyền đọc và ghi vào các thư mục này

#### 9. Khởi động lại IIS trên Windows 11
1. Mở Windows Terminal với quyền Administrator
2. Nhập lệnh:
   ```
   iisreset
   ```
   Hoặc, nếu lệnh trên không hoạt động:
   - Nhấn tổ hợp phím Windows + X, chọn "Terminal (Admin)"
   - Nhập lệnh: `net stop was /y && net start w3svc`

   Hoặc khởi động lại thông qua Services:
   - Nhấn tổ hợp phím Windows + R, gõ "services.msc" và nhấn Enter
   - Tìm "World Wide Web Publishing Service"
   - Nhấp chuột phải và chọn "Restart"

#### 10. Kiểm tra ứng dụng
1. Mở Microsoft Edge hoặc trình duyệt web khác
2. Truy cập: http://localhost:5000
3. Nếu mọi thứ được cấu hình đúng, bạn sẽ thấy thông báo "Server is running" và danh sách các endpoints có sẵn

#### 11. Khắc phục sự cố trên Windows 11
- **Lỗi 500 - Internal Server Error**: 
  - Kiểm tra logs IIS trong `C:\inetpub\logs\LogFiles`
  - Mở Event Viewer (Nhấn Windows + R, gõ "eventvwr.msc") và kiểm tra Windows Logs > Application để tìm lỗi liên quan đến IIS

- **HTTP Error 403.14 - Forbidden**:
  - Đảm bảo rằng handler đã được đăng ký đúng cách trong IIS
  - Kiểm tra lại file web.config

- **Ứng dụng không chạy**: 
  - Đảm bảo rằng tất cả các thư viện Python đã được cài đặt đúng bằng cách chạy: `pip list`
  - Thử chạy ứng dụng trực tiếp bằng cách gọi: `python wsgi.py` để xem có lỗi nào xuất hiện không

- **Lỗi quyền truy cập**: 
  - Mở PowerShell với quyền Administrator và chạy lệnh sau để kiểm tra quyền:
    ```
    icacls "C:\python_api_test"
    ```
  - Đảm bảo nhóm IIS_IUSRS có quyền (RX) trên thư mục

- **Lỗi FastCGI**: 
  - Đảm bảo wfastcgi đã được cấu hình đúng bằng cách kiểm tra trong IIS Manager:
    1. Chọn server trong panel bên trái
    2. Nhấp đúp vào "FastCGI Settings"
    3. Xác nhận rằng đường dẫn Python và wfastcgi đã được liệt kê

- **Lỗi Windows Defender Firewall**: 
  - Mở Windows Defender Firewall với Advanced Security (tìm kiếm "wf.msc" trong Start)
  - Thêm một quy tắc cho phép kết nối đến cổng 5000 (hoặc cổng bạn đã cấu hình)

- **Không thể mở IIS Manager**: 
  - Đảm bảo đã cài đặt đầy đủ IIS, bao gồm IIS Management Console
  - Khởi động lại máy tính sau khi cài đặt IIS
  - Thử mở thông qua Run command: Win+R > "inetmgr"

## Cấu trúc dự án
```
/python_api_test/
  |- app.py               # File chính khởi chạy ứng dụng Flask
  |- config.py            # Cấu hình chung
  |- uploads/             # Nơi lưu trữ hình ảnh được tải lên
  |- roi_data/            # Nơi lưu trữ dữ liệu về các vùng ROI
  |- ocr_results/         # Nơi lưu trữ kết quả OCR
  |- requirements.txt      # Danh sách các thư viện cần thiết
  |- start_server.bat     # Tập lệnh để khởi động server
  |- wsgi.py              # Tập tin WSGI để chạy ứng dụng
  |- README.md            # Tài liệu hướng dẫn sử dụng
```

## API Documentation

## 1. Test Endpoint
- **Endpoint:** `/`
- **Method:** `GET`
- **Description:** Check if the server is running.
- **Response:**
  ```json
  {
      "status": "Server is running",
      "endpoints": ["/api/images"]
  }
  ```

## 2. Debug Information
- **Endpoint:** `/debug`
- **Method:** `GET`
- **Description:** Get detailed debug information about the server and its routes.
- **Response:**
  ```json
  {
      "server_info": {
          "upload_folder": "path/to/upload",
          "roi_data_folder": "path/to/roi_data",
          "ocr_results_folder": "path/to/ocr_results",
          "allowed_extensions": ["png", "jpg", "jpeg", "gif"],
          "max_content_length": 16777216
      },
      "routes": [
          {
              "endpoint": "home",
              "methods": ["GET"],
              "route": "/"
          },
          ...
      ],
      "environment": {
          "host": "localhost",
          "remote_addr": "127.0.0.1",
          "user_agent": "User-Agent"
      }
  }
  ```

## 3. Upload Image
- **Endpoint:** `/api/images`
- **Method:** `POST`
- **Description:** Upload an image and perform OCR on defined ROIs.
- **Form Data:**
  - `file`: The image file to upload.
  - `machine_code`: The machine code (e.g., "F1").
  - `screen_id`: The screen ID (e.g., "Faults").
  - `template_image`: (Optional) The path to a template image.
- **Response:**
  ```json
  {
      "filename": "uploaded_image.png",
      "machine_code": "F1",
      "screen_id": "Faults",
      "timestamp": "2023-01-01 12:00:00",
      "template_path": "path/to/template",
      "results": [
          {
              "roi_index": "ROI_0",
              "text": "123",
              "confidence": 0.99,
              "has_text": true,
              "original_value": "123"
          }
      ]
  }
  ```

## 4. Get Images
- **Endpoint:** `/api/images`
- **Method:** `GET`
- **Description:** Retrieve the latest OCR results.
- **Response:**
  ```json
  {
      "filename": "ocr_result_20230101_120000_uploaded_image_F1_Faults.json",
      "machine_code": "F1",
      "screen_id": "Faults",
      "timestamp": "2023-01-01 12:00:00",
      "results": [...]
  }
  ```

## 5. Get Image
- **Endpoint:** `/api/images/<filename>`
- **Method:** `GET`
- **Description:** Retrieve a specific uploaded image.
- **Response:** The image file.

## 6. Delete Image
- **Endpoint:** `/api/images/<filename>`
- **Method:** `DELETE`
- **Description:** Delete a specific uploaded image.
- **Response:**
  ```json
  {
      "message": "Image uploaded_image.png has been deleted successfully"
  }
  ```

## 7. Get Aligned Image
- **Endpoint:** `/api/images/aligned/<filename>`
- **Method:** `GET`
- **Description:** Retrieve an aligned image.
- **Response:** The aligned image file.

## 8. Get Processed ROI Image
- **Endpoint:** `/api/images/processed_roi/<filename>`
- **Method:** `GET`
- **Description:** Retrieve a processed ROI image.
- **Response:** The processed ROI image file.

## 9. Get Machine Information
- **Endpoint:** `/api/machines`
- **Method:** `GET`
- **Description:** Get information about machines and their screens.
- **Response:**
  ```json
  {
      "machines": [
          {
              "machine_code": "F1",
              "name": "Machine 1"
          }
      ]
  }
  ```

## 10. Set Current Machine and Screen
- **Endpoint:** `/api/set_machine_screen`
- **Method:** `POST`
- **Description:** Set the current machine and screen based on provided parameters.
- **Form Data:**
  - `machine_code`: The machine code.
  - `screen_id`: The screen ID.
- **Response:**
  ```json
  {
      "message": "Machine and screen selection updated successfully",
      "machine": {
          "machine_code": "F1"
      },
      "screen": {
          "id": 1,
          "screen_id": "Faults"
      }
  }
  ```

## 11. Get Current Machine and Screen
- **Endpoint:** `/api/current_machine_screen`
- **Method:** `GET`
- **Description:** Get the currently selected machine and screen information.
- **Response:**
  ```json
  {
      "machine": {
          "machine_code": "F1",
          "name": "Machine 1"
      },
      "screen": {
          "id": 1,
          "screen_id": "Faults"
      }
  }
  ```

## 12. Check Machine and Screen Status
- **Endpoint:** `/api/machine_screen_status`
- **Method:** `GET`
- **Description:** Check the status of the current machine and screen configuration.
- **Query Parameters:**
  - `machine_code`: (Optional) The machine code.
  - `screen_id`: (Optional) The screen ID.
- **Response:**
  ```json
  {
      "machine_code": "F1",
      "machine_name": "Machine 1",
      "screen_id": 1,
      "screen_name": "Faults",
      "has_roi": true,
      "roi_count": 3,
      "has_decimal_config": true,
      "is_fully_configured": true,
      "roi_status": [...]
  }
  ```

## 13. Update Decimal Places Configuration
- **Endpoint:** `/api/decimal_places/<machine_code>/<screen_name>`
- **Method:** `POST`
- **Description:** Update the decimal places configuration for a specific screen.
- **Request Body (JSON):**
  ```json
  {
      "key1": value1,
      "key2": value2
  }
  ```
- **Response:**
  ```json
  {
      "message": "Decimal places configuration updated successfully",
      "machine_code": "F1",
      "screen_name": "Faults",
      "changes": {
          "added": {},
          "updated": {}
      },
      "config": {...}
  }
  ```

## 14. Upload Reference Image
- **Endpoint:** `/api/reference_images`
- **Method:** `POST`
- **Description:** Upload a reference template image for alignment.
- **Form Data:**
  - `file`: The reference image file.
  - `machine_code`: The machine code.
  - `screen_id`: The screen ID.1
- **Response:**
  ```json
  {
      "message": "Reference template uploaded successfully",
      "template": {
          "filename": "template_F1_Faults.jpg",
          "path": "/api/reference_images/template_F1_Faults.jpg",
          "size": 12345,
          "dimensions": "1920x1080"
      }
  }
  ```

## 15. Get Reference Images
- **Endpoint:** `/api/reference_images`
- **Method:** `GET`
- **Description:** Retrieve a list of uploaded reference template images.
- **Response:**
  ```json
  {
      "reference_images": [...],
      "count": 2
  }
  ```

## 16. Delete Reference Image
- **Endpoint:** `/api/reference_images/<filename>`
- **Method:** `DELETE`
- **Description:** Delete a specific reference template image.
- **Response:**
  ```json
  {
      "message": "Reference template template_F1_Faults.jpg has been deleted successfully"
  }
  ```

## Cách sử dụng

### Thiết lập máy và màn hình
1. Gọi API `/api/set_machine_screen` để thiết lập máy và màn hình hiện tại:
   ```
   POST /api/set_machine_screen
   Content-Type: application/x-www-form-urlencoded
   
   machine_code=F1&screen_id=1
   ```

### Tải lên hình ảnh và thực hiện OCR
1. Gọi API `/api/images` để tải lên hình ảnh:
   ```
   POST /api/images
   Content-Type: multipart/form-data
   
   file=@path_to_your_image.jpg
   machine_code=F1
   screen_id=1
   ```

2. Kết quả OCR sẽ được trả về trong response, bao gồm các số được trích xuất từ các vùng ROI đã định nghĩa.

### Cấu hình số chữ số thập phân
1. Gọi API `/api/set_decimal_value` để cập nhật số chữ số thập phân cho một ROI:
   ```
   POST /api/set_decimal_value
   Content-Type: application/x-www-form-urlencoded
   
   machine_code=F1&screen_id=1&roi_index=0&value=2
   ```

## Gỡ lỗi
- Xem log server trong terminal
- Sử dụng endpoint `/debug` để xem thông tin debug chi tiết
---