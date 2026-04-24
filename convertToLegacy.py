import torch
import os


def sync_and_convert_to_legacy(custom_path, template_path, output_path):
    print(f"--- 1. Đang tải model của bạn: {custom_path} ---")
    custom_ckpt = torch.load(custom_path, map_location="cpu")

    print(f"--- 2. Đang tải model mẫu để lấy khuôn: {template_path} ---")
    template_ckpt = torch.load(template_path, map_location="cpu")

    custom_model = custom_ckpt['model']
    template_keys = set(template_ckpt['model'].keys())
    custom_keys = set(custom_model.keys())

    # --- BƯỚC 1: ĐỒNG BỘ TRỌNG SỐ (WEIGHTS) ---
    print("\n--- 3. Bắt đầu đồng bộ cấu trúc trọng số ---")
    extra_keys = custom_keys - template_keys
    for k in extra_keys:
        custom_model.pop(k)
        print(f"Đã cắt bỏ lớp thừa: {k}")

    missing_keys = template_keys - custom_keys
    if missing_keys:
        print(f"CẢNH BÁO: Thiếu {len(missing_keys)} lớp so với mẫu!")
    else:
        print("Trọng số đã khớp 100% với mẫu.")

        # --- BƯỚC 2: DỌN DẸP CẤU HÌNH (CONFIG) ---
        if 'cfg' in custom_ckpt:
            print("\n--- 4. Đang dọn dẹp Config để tránh lỗi OmegaConf ---")

            # 4.1. Đổi tên Task và Model về Hubert truyền thống
            custom_ckpt['cfg']['model']['_name'] = 'hubert'
            custom_ckpt['cfg']['task']['_name'] = 'hubert_pretraining'
            custom_ckpt['cfg']['criterion']['_name'] = 'hubert'

            # 4.2. XỬ LÝ PHẦN TASK (Sửa Label Rate và Đường dẫn cứng)
            task_cfg = custom_ckpt['cfg']['task']
            forbidden_task_keys = ['crop', 'spk2info', 'random_crop', 'single_target']

            for k in forbidden_task_keys:
                if k in task_cfg:
                    if hasattr(task_cfg, 'pop'):
                        task_cfg.pop(k)
                    else:
                        del task_cfg[k]
                    print(f"Đã xóa key dư thừa trong task: {k}")

            # Gán lại các giá trị chuẩn mực, xóa đường dẫn ổ D:
            task_cfg['data'] = 'metadata'
            task_cfg['label_dir'] = 'label'
            task_cfg['labels'] = ['km']
            task_cfg['sample_rate'] = 16000
            task_cfg['random_crop'] = True
            task_cfg['single_target'] = False
            task_cfg['label_rate'] = 50  # <--- Sửa lỗi label_rate -1.0 gây hỏng Naive model

            # 4.3. XỬ LÝ PHẦN MODEL (Xóa các biến ContentVec gây lỗi OmegaConf)
            model_cfg = custom_ckpt['cfg']['model']
            forbidden_model_keys = ['logit_temp_ctr', 'ctr_layers', 'extractor_mode', 'encoder_layers_1']

            for k in forbidden_model_keys:
                if k in model_cfg:
                    if hasattr(model_cfg, 'pop'):
                        model_cfg.pop(k)
                    else:
                        del model_cfg[k]
                    print(f"Đã xóa key ContentVec gây lỗi trong model: {k}")

            # Ép thông số layer về chuẩn HuBERT Base
            model_cfg['encoder_layers'] = 12
            model_cfg['label_rate'] = 50  # Đồng bộ với task

    # --- BƯỚC 3: ĐỒNG BỘ LỊCH SỬ OPTIMIZER ---
    if 'optimizer_history' in custom_ckpt:
        print("\n--- 5. Đang đồng bộ hóa Optimizer History ---")
        for history in custom_ckpt['optimizer_history']:
            history['criterion_name'] = 'HubertCriterion'

    # --- LƯU FILE ---
    print(f"\n--- 6. Đang lưu model cuối cùng: {output_path} ---")
    torch.save(custom_ckpt, output_path)
    print("Xong! Model đã sạch và sẵn sàng cho Diffusion-SVC.")


# Cấu hình đường dẫn
my_model = "checkpoint_254_2000.pt"
sample_model = "checkpoint_best_legacy_500.pt"
final_output = "checkpoint_best_legacy_500_new.pt"

sync_and_convert_to_legacy(my_model, sample_model, final_output)
