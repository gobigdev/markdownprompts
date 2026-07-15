# 1. 创建并激活 Python 虚拟环境
python -m venv funasr_env
# Windows 激活命令：
funasr_env\Scripts\activate
# Linux/Mac 激活命令：
# source funasr_env/bin/activate

# 2. 安装支持 CUDA 的 PyTorch（针对你的 5060 Ti 显卡）
pip3 install torch torchvision audio --index-url https://download.pytorch.org/whl/cu121

# 3. 安装 FunASR 及声纹、语言包核心库
pip install funasr
pip install modelscope
pip install torchaudio
