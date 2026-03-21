```bash
# 进入项目
cd /home/lijie/code/linux-asr-voicetype

# 一键安装（依赖 + 服务）
./install.sh

# UI 服务（安装后默认已启动）
systemctl --user status voicetype-ui.service

# 打开控制 UI（浏览器访问）
# http://127.0.0.1:8790/ui

# 在 UI 里保存配置后，点“启动/重启 ASR”
# 再看 ASR 状态
systemctl --user status voicetype.service
```
