import pickle
import os

# 1. Mở file hiện tại của bạn (cái đang có Key là '0000')
path = r"D:\Tools\Projects\contentvec\dataset\manifest\spk2info.dict"
with open(path, 'rb') as f:
    old_data = pickle.load(f)

new_data = {"train": {}, "valid": {}}

# 2. Thêm đuôi .wav vào tất cả các Key
for split in ['train', 'valid']:
    for key, value in old_data[split].items():
        # Nếu key chưa có đuôi .wav thì thêm vào
        new_key = key if key.endswith(".wav") else f"{key}.wav"
        new_data[split][new_key] = value

# 3. Lưu lại
with open(path, 'wb') as f:
    pickle.dump(new_data, f)

print("✅ Đã sửa Key thành '0000.wav', '0996.wav', ...")
print("Bây giờ bạn hãy chạy lại lệnh TRAIN.")
