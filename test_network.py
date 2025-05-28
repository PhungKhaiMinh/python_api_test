#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket
import requests
import time

def get_local_ips():
    """Lấy tất cả IP addresses của máy"""
    hostname = socket.gethostname()
    local_ips = []
    
    # Lấy IP từ hostname
    try:
        local_ips.append(socket.gethostbyname(hostname))
    except:
        pass
    
    # Lấy IP bằng cách kết nối đến external server
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ips.append(s.getsockname()[0])
        s.close()
    except:
        pass
    
    return list(set(local_ips))

def test_flask_connectivity():
    """Test kết nối đến Flask app"""
    ips = get_local_ips()
    port = 5000
    
    print("=== NETWORK CONNECTIVITY TEST ===")
    print(f"Detected IPs: {ips}")
    print()
    
    for ip in ips:
        url = f"http://{ip}:{port}"
        print(f"Testing: {url}")
        
        try:
            response = requests.get(url, timeout=5)
            print(f"  ✅ SUCCESS: Status {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"  ❌ CONNECTION REFUSED: App not running or port blocked")
        except requests.exceptions.Timeout:
            print(f"  ⏰ TIMEOUT: Network issue")
        except Exception as e:
            print(f"  ❌ ERROR: {str(e)}")
        print()
    
    # Test localhost
    localhost_url = f"http://localhost:{port}"
    print(f"Testing: {localhost_url}")
    try:
        response = requests.get(localhost_url, timeout=5)
        print(f"  ✅ SUCCESS: Status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"  ❌ CONNECTION REFUSED: App not running")
    except Exception as e:
        print(f"  ❌ ERROR: {str(e)}")

if __name__ == "__main__":
    test_flask_connectivity() 