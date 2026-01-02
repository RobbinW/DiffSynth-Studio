import os
import json
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ================= 配置区域 =================

# 1. 原始视频所在的 chunk 目录列表
SOURCE_VIDEO_CHUNKS = [
    "/data/dex/RoboTwin/lerobot_data/huggingface/lerobot/RoboTwin/videos/chunk-000/observation.images.cam_high",
    "/data/dex/RoboTwin/lerobot_data/huggingface/lerobot/RoboTwin/videos/chunk-001/observation.images.cam_high",
    "/data/dex/RoboTwin/lerobot_data/huggingface/lerobot/RoboTwin/videos/chunk-002/observation.images.cam_high"
]

# 2. [修改] 直接指向 episodes.jsonl 文件
SOURCE_META_PATH = "/data/dex/RoboTwin/lerobot_data/huggingface/lerobot/RoboTwin/meta/episodes.jsonl"

# 3. 输出目录 (会自动创建)
OUTPUT_DIR = "/data/dex/DiffSynth-Studio/data/RoboTwin/wanvideo_dataset"

# 4. 模式: 'symlink' (推荐, 秒级完成, 不占空间) 或 'copy' (慢, 占用双倍空间)
MODE = 'copy' 

# ===========================================

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def main():
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 开始处理数据集 (直读模式)...")
    print(f"📂 输出目录: {output_path}")

    # --- 步骤 1: 加载 episodes.jsonl ---
    print(f"📖 正在读取 Meta文件: {SOURCE_META_PATH}")
    
    if not os.path.exists(SOURCE_META_PATH):
        print(f"❌ 错误: 找不到文件 {SOURCE_META_PATH}")
        return

    episodes_data = load_jsonl(SOURCE_META_PATH)
    
    # 建立映射: episode_index -> prompt_text
    # 数据格式示例: {"episode_index": 0, "tasks": ["Trigger the switch..."], ...}
    episode_prompt_map = {}
    
    for item in episodes_data:
        ep_idx = item.get('episode_index')
        tasks_list = item.get('tasks', [])
        
        # 提取 Prompt
        if ep_idx is not None and tasks_list:
            # 直接取列表里的第一个字符串作为 Prompt
            prompt_text = tasks_list[0] if isinstance(tasks_list, list) else str(tasks_list)
            episode_prompt_map[ep_idx] = prompt_text
            
    print(f"   - 已加载 {len(episode_prompt_map)} 条 Episode 描述")

    # --- 步骤 2: 扫描视频文件 ---
    print("🔍 正在扫描视频文件...")
    video_files = []
    for chunk_dir in SOURCE_VIDEO_CHUNKS:
        path = Path(chunk_dir)
        if not path.exists():
            print(f"⚠️ 跳过不存在的目录: {path}")
            continue
        # 查找所有 MP4
        files = list(path.glob("*.mp4"))
        video_files.extend(files)
    
    video_files.sort(key=lambda x: x.name) # 排序保证顺序
    print(f"   - 共找到 {len(video_files)} 个视频文件")

    # --- 步骤 3: 匹配并生成 ---
    csv_rows = []
    
    for src_file in tqdm(video_files, desc="Processing"):
        # 解析文件名: episode_000000.mp4 -> int(0)
        try:
            episode_id_str = src_file.stem.split('_')[-1]
            episode_id = int(episode_id_str)
        except ValueError:
            print(f"⚠️ 跳过格式不对的文件: {src_file.name}")
            continue

        # 直接查字典获取 Prompt
        prompt = episode_prompt_map.get(episode_id, "")
        
        # 如果 prompt 为空，打印一条警告，方便debug
        if not prompt:
            # print(f"⚠️ Warning: Episode {episode_id} 没有找到对应的 Prompt")
            pass

        # 构造新文件名
        new_filename = f"video_{episode_id:06d}.mp4" 
        dst_file = output_path / new_filename

        # 建立软链接
        if not dst_file.exists():
            if MODE == 'symlink':
                try:
                    os.symlink(src_file, dst_file)
                except OSError:
                    shutil.copy2(src_file, dst_file)
            else:
                shutil.copy2(src_file, dst_file)

        # 添加到 CSV 数据
        csv_rows.append({
            "video": new_filename,
            "prompt": prompt
        })

    # --- 步骤 4: 保存 metadata.csv ---
    print("💾 正在保存 metadata.csv...")
    df = pd.DataFrame(csv_rows)
    
    # 确保 video 在前，prompt 在后
    df = df[['video', 'prompt']]
    
    csv_path = output_path / "metadata.csv"
    # quoting=1 (csv.QUOTE_ALL) 会给所有非数字字段加双引号，完美符合要求
    df.to_csv(csv_path, index=False, quoting=1)
    
    print(f"✅ 处理完成！")
    print(f"   - CSV路径: {csv_path}")
    print(f"   - 总数据量: {len(df)}")

if __name__ == "__main__":
    main()