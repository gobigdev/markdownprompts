# 安装 whisperx 
pip install whisperx

# 运行转录并启用声纹识别（需要提供 Hugging Face 的 Pyannote token）
whisperx example.wav --model large-v3 --diarize --hf_token YOUR_HF_TOKEN
