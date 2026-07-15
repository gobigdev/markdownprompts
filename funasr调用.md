# 修改 generate 参数示例
res = model.generate(
    input=audio_path,
    prompt="Transcribe the following audio into English text only.", # 注入提示词
    language="en"
)
