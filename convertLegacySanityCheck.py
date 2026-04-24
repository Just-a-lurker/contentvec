import torch
import logging

# Cấu hình Logger giống hệt Fairseq
logging.basicConfig(
    format='%(asctime)s | INFO | %(name)s | %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger_task = logging.getLogger("fairseq.tasks.hubert_pretraining")
logger_model = logging.getLogger("fairseq.models.hubert.hubert")


def compare_models(custom_path, template_path):
    # 1. Load models
    logging.info(f"--- Đang phân tích Model Của Bạn: {custom_path} ---")
    custom_ckpt = torch.load(custom_path, map_location="cpu")

    logging.info(f"--- Đang phân tích Model Mẫu: {template_path} ---")
    template_ckpt = torch.load(template_path, map_location="cpu")

    # --- PHẦN 2: IN LOG CONFIG CHO CẢ 2 ĐỂ SO SÁNH ---

    # --- In Config của Model Của Bạn ---
    print("\n" + ">" * 20 + " CẤU HÌNH MODEL CỦA BẠN " + "<" * 20)
    if 'cfg' in custom_ckpt:
        logger_task.info(f"Task Config: {dict(custom_ckpt['cfg'].get('task', {}))}")
        logger_model.info(f"Model Config: {dict(custom_ckpt['cfg'].get('model', {}))}")

    # --- In Config của Model Mẫu ---
    print("\n" + ">" * 20 + " CẤU HÌNH MODEL MẪU (TEMPLATE) " + "<" * 20)
    if 'cfg' in template_ckpt:
        logger_task.info(f"Task Config: {dict(template_ckpt['cfg'].get('task', {}))}")
        logger_model.info(f"Model Config: {dict(template_ckpt['cfg'].get('model', {}))}")

    # --- PHẦN 3: SO SÁNH KEYS TRỌNG SỐ ---
    custom_model = custom_ckpt['model']
    template_model = template_ckpt['model']

    custom_keys = set(custom_model.keys())
    template_keys = set(template_model.keys())

    extra_keys = custom_keys - template_keys
    missing_keys = template_keys - custom_keys

    print("\n" + "=" * 60)
    if not extra_keys and not missing_keys:
        logging.info("THÀNH CÔNG: Cấu trúc trọng số 2 model giống hệt nhau")
    else:
        if extra_keys:
            print(f"THỪA ({len(extra_keys)} keys) trong model của bạn:")
            for k in sorted(extra_keys): print(f"   - {k}")

        if missing_keys:
            print(f"\nTHIẾU ({len(missing_keys)} keys) so với mẫu:")
            for k in sorted(missing_keys): print(f"   - {k}")
    print("=" * 60)

    # --- PHẦN 4: KIỂM TRA SHAPE ---
    sample_key = "encoder.layers.0.self_attn.out_proj.weight"
    if sample_key in custom_model and sample_key in template_model:
        print(f"\nKiểm tra Shape lớp '{sample_key}':")
        print(f"   Model của bạn: {custom_model[sample_key].shape}")
        print(f"   Model mẫu    : {template_model[sample_key].shape}")


# Chạy lệnh
my_model = "checkpoint_best_legacy_500_new.pt"
sample_model = "checkpoint_best_legacy_500.pt"
compare_models(my_model, sample_model)
