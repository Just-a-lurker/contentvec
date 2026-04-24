import os

# Đảm bảo đường dẫn này khớp với máy của bạn
files_to_check = [
    "D:/Tools/Projects/contentvec/dataset/spk2info.dict",
    "D:/Tools/Projects/contentvec/dataset/manifest/spk2info.dict"
]

for file_path in files_to_check:
    if os.path.exists(file_path):
        size = os.path.getsize(file_path)
        print(f"--- File: {os.path.basename(file_path)} ---")
        print(f"Kích thước: {size} bytes")

        try:
            with open(file_path, 'rb') as f:
                header = f.read(15)
                print(f"Header (15 bytes đầu): {header}")

                # Phân tích chữ ký
                if header.startswith(b'\x80'):
                    print("Định dạng: Pickle nhị phân hợp lệ.")
                elif header.startswith(b'PK'):
                    print("Định dạng: PyTorch Zip file hợp lệ.")
                else:
                    print(
                        "CẢNH BÁO: Đây dường như là tệp văn bản thuần túy hoặc tệp bị hỏng, KHÔNG PHẢI file nhị phân!")
        except Exception as e:
            print(f"Không thể đọc file: {e}")
    else:
        print(f"--- File: {os.path.basename(file_path)} ---")
        print("KHÔNG TÌM THẤY FILE!")
    print("-" * 40 + "\n")