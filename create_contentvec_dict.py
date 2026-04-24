import os
import random
import pickle
import numpy as np
import parselmouth
from tqdm import tqdm
from resemblyzer import VoiceEncoder, preprocess_wav

#CẤU HÌNH
manifest_dir = "D:/Tools/Projects/contentvec/dataset/manifest"
audio_root_dir = "D:/Tools/Projects/contentvec/dataset/wavs"
train_tsv = os.path.join(manifest_dir, "train.tsv")
output_dict_path = os.path.join(manifest_dir, "spk2info.dict")
samples_per_speaker = 200



def get_f0_with_parselmouth(filepath):
    try:
        snd = parselmouth.Sound(filepath)
        pitch = snd.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]
        if len(pitch_values) == 0: return 0.0, 0.0, 50.0
        return pitch_values.mean(), pitch_values.max(), max(50.0, pitch_values.min())
    except:
        return 0.0, 0.0, 50.0


def generate_nested_spk2info():
    speaker_files = {}
    with open(train_tsv, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]
        for line in lines:
            rel_path = line.split('\t')[0]
            # Xử lý lấy ID người nói (nam/nu)
            speaker_id = os.path.dirname(rel_path).replace('\\', '/').split('/')[0]
            full_path = os.path.join(audio_root_dir, rel_path)
            if speaker_id not in speaker_files: speaker_files[speaker_id] = []
            speaker_files[speaker_id].append(full_path)

    encoder = VoiceEncoder()
    # Đây là nơi lưu trữ dữ liệu người nói thực tế
    extracted_data = {}

    for speaker_id, files in speaker_files.items():
        sampled_files = random.sample(files, min(len(files), samples_per_speaker))
        print(f"Đang trích xuất: {speaker_id}...")

        embeds, f0_means, f0_maxs, f0_mins = [], [], [], []
        for filepath in tqdm(sampled_files):
            f0_mean, f0_max, f0_min = get_f0_with_parselmouth(filepath)
            if f0_mean > 0:
                f0_means.append(f0_mean);
                f0_maxs.append(f0_max);
                f0_mins.append(f0_min)
            try:
                wav = preprocess_wav(filepath)
                embeds.append(encoder.embed_utterance(wav))
            except:
                continue

        avg_embed = np.maximum(0, np.mean(embeds, axis=0).astype(np.float32))
        stats = (float(np.mean(f0_means)), float(np.max(f0_maxs)), float(np.min(f0_mins)))
        extracted_data[speaker_id] = (avg_embed, stats)

    # ĐÓNG GÓI CẤU TRÚC:key 'train' và 'valid' ở ngoài cùng
    final_dict = {
        'train': extracted_data,
        'valid': extracted_data  # Dùng chung dữ liệu người nói cho cả tập valid
    }

    with open(output_dict_path, 'wb') as f:
        pickle.dump(final_dict, f, protocol=3)
    print(f"\nĐã tạo xong spk2info.dict với cấu trúc lồng ghép!")


if __name__ == "__main__":
    generate_nested_spk2info()