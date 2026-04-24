import os

# Sử dụng r"" (raw string) để tránh lỗi ký tự đặc biệt trong đường dẫn Windows
file_path = r"D:\Tools\Projects\contentvec\dataset\manifest\dict.km.txt"

# Đảm bảo thư mục tồn tại trước khi ghi
os.makedirs(os.path.dirname(file_path), exist_ok=True)

with open(file_path, "w", encoding="utf-8") as f:
    for i in range(500):
        # Định dạng: [nhãn] [tần suất_giả]
        # Fairseq yêu cầu một con số đi kèm sau nhãn, thường để là 1
        f.write(f"{i} 1\n")

print(f"Đã tạo xong file dict mới với 500 nhãn tại: {file_path}")