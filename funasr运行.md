import json
from funasr import AutoModel

def run_transcription(audio_path):
    print("正在初始化 FunASR 模型...")
    
    # 初始化模型
    # model: 多语言语音识别大模型 (支持强制英文)
    # sv_model: 声纹识别模型 (用于提取说话人特征)
    model = AutoModel(
        model="iic/SenseVoiceSmall",            # 阿里多语言旗舰小模型
        sv_model="iic/speech_eres2net_sv_zh-cn_16k-common", # 声纹模型
        sv_model_revision="v1.0.0",
        vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", # 语音活动检测
        device="cuda:0"                         # 强制使用显卡加速
    )

    print(f"正在处理音频文件: {audio_path} ...")
    
    # 开始推理
    res = model.generate(
        input=audio_path,
        batch_size=1,
        language="en",          # 👈 核心参数：强制要求输出语言为英文
        use_itn=True            # 是否将数字等转化为文本逆规整（如 10 转化为 ten）
    )

    # 打印原始返回结构方便调试
    print("\n--- 推理完成，返回结果 ---")
    
    # 格式化输出结果
    if res and len(res) > 0:
        # SenseVoice/FunASR 返回的结果通常包含 text 以及包含时间戳和声纹信息的 label
        result_text = res[0].get('text', '')
        print(f"\n【最终转写文本】:\n{result_text}")
        
        # 如果音频包含多个人说话，且声纹模型成功分离：
        if 'timestamps' in res[0] or 'speaker' in res[0]:
            print("\n【说话人日志/声纹详细数据】:")
            print(json.dumps(res[0], indent=2, ensure_ascii=False))
    else:
        print("未检测到有效语音或转写失败。")

if __name__ == "__main__":
    # 替换为你本地的音频文件路径（支持 wav, mp3 等）
    # 建议声纹识别使用 16k 采样率的单声道 WAV 文件效果最好
    AUDIO_FILE = "test_audio.wav" 
    
    run_transcription(AUDIO_FILE)
