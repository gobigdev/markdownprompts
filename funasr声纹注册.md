import os
import json
import torch
import torchaudio
import numpy as np
from funasr import AutoModel

class SpeakerRegistry:
    def __init__(self, db_path="speaker_db.json", device="cuda:0"):
        self.db_path = db_path
        self.device = device
        
        print("正在初始化 FunASR 声纹与 VAD 模型...")
        # 1. 初始化专门的声纹识别模型 (3D-Speaker 旗舰 ERes2Net)
        self.sv_model = AutoModel(
            model="iic/speech_eres2net_sv_zh-cn_16k-common", 
            sv_model_revision="v1.0.0",
            device=self.device
        )
        # 2. 初始化 VAD 模型进行静音切除预处理
        self.vad_model = AutoModel(
            model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            device=self.device
        )
        
        # 加载本地声纹数据库
        self.speaker_db = self._load_db()

    def _load_db(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "r", encoding="utf-8") as f:
                print(f"成功加载本地声纹库，当前已注册 {len(self.speaker_db)} 个用户。")
                return json.load(f)
        return {}

    def _save_db(self):
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.speaker_db, f, indent=4, ensure_ascii=False)
        print("声纹库更新并成功保存到本地。")

    def preprocess_audio(self, audio_path):
        """
        预处理流程：
        1. 检查并强制重采样为 16000Hz，转换为单声道。
        2. 运行 VAD，切除无声和静音片段，拼接纯人声。
        """
        print(f"[预处理] 正在读取音频: {audio_path}")
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # 步骤 1: 转换为单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # 步骤 2: 重采样至 16000Hz
        TARGET_SR = 16000
        if sample_rate != TARGET_SR:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=TARGET_SR)
            waveform = resampler(waveform)
            
        # 保存为一个临时标准 WAV 文件用于 VAD 和声纹分析
        temp_standard_path = "temp_preprocessed.wav"
        torchaudio.save(temp_standard_path, waveform, TARGET_SR)
        
        # 步骤 3: 运行 VAD 进行人声活动检测（切除静音）
        print("[预处理] 正在通过 VAD 切除静音...")
        vad_res = self.vad_model.generate(input=temp_standard_path)
        
        # 根据 VAD 返回的时间戳重新裁剪拼接音频
        # vad_res[0]['value'] 通常是一个包含多段 [start_ms, end_ms] 的列表
        if vad_res and 'value' in vad_res[0] and len(vad_res[0]['value']) > 0:
            segments = vad_res[0]['value']
            pure_voice_chunks = []
            
            # 转换为采样的 sample 点索引进行裁剪
            # 1ms = 16 个采样点 (在 16000Hz 下)
            for start_ms, end_ms in segments:
                start_sample = int(start_ms * 16)
                end_sample = int(end_ms * 16)
                pure_voice_chunks.append(waveform[:, start_sample:end_sample])
                
            # 拼接所有有人声的片段
            processed_waveform = torch.cat(pure_voice_chunks, dim=1)
            torchaudio.save(temp_standard_path, processed_waveform, TARGET_SR)
            print(f"[预处理] 静音切除完毕。原始有效长度约: {processed_waveform.shape[1]/16000:.2f} 秒。")
            return temp_standard_path
        else:
            print("[警告] VAD 未在音频中检测到有效人声！将直接使用原音频进行后续操作。")
            return temp_standard_path

    def register_speaker(self, name, audio_path):
        """
        声纹注册核心函数
        """
        # 1. 触发预处理
        cleaned_audio = self.preprocess_audio(audio_path)
        
        print(f"[注册] 正在提取 {name} 的声纹特征向量...")
        # 2. 提取声纹特征
        sv_res = self.sv_model.generate(input=cleaned_audio)
        
        # 3. 清理临时文件
        if os.path.exists(cleaned_audio):
            os.remove(cleaned_audio)
            
        if sv_res and len(sv_res) > 0 and 'spk_embedding' in sv_res[0]:
            embedding = sv_res[0]['spk_embedding']
            
            # 将 Tensor 转化为 list 方便转为 JSON 存储
            if isinstance(embedding, torch.Tensor):
                embedding_list = embedding.cpu().numpy().tolist()
            else:
                embedding_list = np.array(embedding).tolist()
                
            # 4. 写入本地“数据库”
            self.speaker_db[name] = embedding_list
            self._save_db()
            print(f"🎉 恭喜！用户【{name}】声纹注册成功！")
            return True
        else:
            print(f"❌ 注册失败：未能从音频中提取到有效的声纹特征向量。")
            return False


if __name__ == "__main__":
    # 实例化注册器（会自动检测并利用你的 5060 Ti 显卡）
    registry = SpeakerRegistry(db_path="my_speakers.json", device="cuda:0")
    
    # ==========================================
    # 示例操作 1: 注册新用户
    # ==========================================
    # 准备一段只有你说英文/中文的干净音频文件（建议3-10秒，可以是 mp3/wav/m4a 等）
    USER_NAME = "张三"
    REGISTER_AUDIO = "zhangsan_register.wav" 
    
    # 真实运行前请确保本地有对应的音频文件，解除注释即可运行
    # registry.register_speaker(name=USER_NAME, audio_path=REGISTER_AUDIO)
    
    # ==========================================
    # 示例操作 2: 再次注册另一个用户
    # ==========================================
    # registry.register_speaker(name="李四", audio_path="lisi_voice.mp3")
