# Frontend Protocol (fcitx5 bridge)

`fcitx5` 薄前端建议只调用以下接口：

1. `POST /v1/session/start`
2. 录音中每 200ms 调用 `POST /v1/session/{id}/chunk`
3. 结束录音调用 `POST /v1/session/{id}/finish`
4. 拿到 `text` 后由前端 commit 到输入法上下文

## 请求示例

```json
POST /v1/session/start
{ "language": "zh" }
```

```json
POST /v1/session/{id}/chunk
{
  "audio_base64": "<mono-pcm16-wav-base64>",
  "sample_rate": 16000
}
```

```json
POST /v1/session/{id}/finish
{ "language": "zh" }
```

## 响应示例

```json
{
  "text": "今天下午三点开会",
  "language": "Chinese",
  "success": true,
  "error": null
}
```

## MVP Toggle Recording API

如果前端不处理音频流，也可以调用录音托管接口：

1. `POST /v1/recording/start`
2. `POST /v1/recording/stop`

其中录音由服务端通过 `arecord` 执行，停止后直接返回识别结果。
