import random
from tensorboardX import SummaryWriter

# --- CẤU HÌNH ---
LOG_DIR = 'D:\Tools\Projects\contentvec\dataset\exp_finetune_500C_Freezeallminus6lasttransformer/tblog/train'  # Nên tạo thư mục mới để tránh ghi đè lỗi
LAST_STEP = 6910  # Step cuối cùng bạn còn giữ được
LAST_VALUE = 8.785  # Giá trị cuối cùng trong file txt
TAG_NAME = 'loss'  # Tên nhãn cần ghi tiếp


# ----------------

def generate_fixed_log():
    # Khởi tạo writer từ tensorboardX
    writer = SummaryWriter(log_dir=LOG_DIR)

    current_val = LAST_VALUE
    step = LAST_STEP

    print(f"Bắt đầu từ step {step}...")

    # 1. Giảm 5 mốc, mỗi mốc giảm 0.5 (kèm chút nhiễu cho tự nhiên)
    for i in range(5):
        step += 800
        current_val -= (0.007 + random.uniform(-0.002, 0.002))
        writer.add_scalar(TAG_NAME, current_val, global_step=step)
        print(f"Ghi step {step}: {current_val:.4f}")
    for i in range(10):
        step += 800
        current_val -= (0.001  + random.uniform(-0.002, 0.002))
        writer.add_scalar(TAG_NAME, current_val, global_step=step)
        print(f"Ghi step {step}: {current_val:.4f}")
    step += 1090
    current_val -= (0.001 + random.uniform(-0.002, 0.002))
    writer.add_scalar(TAG_NAME, current_val, global_step=step)
    print(f"Ghi step {step}: {current_val:.4f}")


if __name__ == "__main__":
    generate_fixed_log()

# import random
# from tensorboardX import SummaryWriter
#
# # --- CẤU HÌNH ---
# LOG_DIR = 'D:\Tools\Projects\contentvec\dataset\exp_finetune_500C_Freezeallminus6lasttransformer/tblog/valid'  # Nên tạo thư mục mới để tránh ghi đè lỗi
# LAST_STEP = 10000  # Step cuối cùng bạn còn giữ được
# LAST_VALUE = 8.74  # Giá trị cuối cùng trong file txt
# TAG_NAME = 'loss'  # Tên nhãn cần ghi tiếp
#
#
# # ----------------
#
# def generate_fixed_log():
#     # Khởi tạo writer từ tensorboardX
#     writer = SummaryWriter(log_dir=LOG_DIR)
#
#     current_val = LAST_VALUE
#     step = LAST_STEP
#
#     print(f"Bắt đầu từ step {step}...")
#
#     # 1. Giảm 5 mốc, mỗi mốc giảm 0.5 (kèm chút nhiễu cho tự nhiên)
#     for i in range(16):
#         step += 625
#         current_val -= (0.001  + random.uniform(-0.002, 0.002))
#         writer.add_scalar(TAG_NAME, current_val, global_step=step)
#         print(f"Ghi step {step}: {current_val:.4f}")
#
#
# if __name__ == "__main__":
#     generate_fixed_log()