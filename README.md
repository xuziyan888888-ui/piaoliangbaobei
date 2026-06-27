# piaoliangbaobei

AI 换妆换发项目骨架。

当前仓库包含：

- 技术架构评审文档
- 研发实现方案
- MVP 后端服务骨架
- SQLite 任务持久化
- 结构化预处理 / 参考解析协议
- `local_inpaint` 真实模型调用适配层

## 快速开始

这台机器系统默认 `python` 指向 `Python 2.7`，建议直接使用工作区自带的 Python 运行时：

```bash
"C:\Users\19770\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m uvicorn app.main:app --reload
```

启动后可访问：

- `GET /healthz`
- `POST /v1/makeup-transfer/jobs`
- `GET /v1/makeup-transfer/jobs/{job_id}`
- `GET /v1/makeup-transfer/jobs/{job_id}/artifacts`

默认 SQLite 数据库路径：

- `data/piaoliangbaobei.db`

默认结果图输出目录：

- `outputs/`

## 测试图随机生成

项目支持直接使用本地文件路径作为 `source_image` 和 `reference_image`。

当前测试图目录：

- `D:\水木年华\测试图`

目录约定：

- `origin*` 作为原图候选
- `reference*` 作为参考图候选

每次随机抽一张原图和一张参考图进行生成：

```bash
"C:\Users\19770\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" scripts\run_random_generation.py
```

## Local Inpaint 调用模式

当前 `local_inpaint` 支持三种模式：

- `mock`
- `generic_http`
- `ark_http`

默认会读取 `.env.local`。如果你当前已经配置了：

```bash
GENERATION_PROVIDER=ark_http
AI_MAKEUP_PROVIDER=ark_http
```

那么系统会优先尝试走火山视觉异步任务接口。

## Ark HTTP 配置

项目当前支持从 `.env.local` 读取以下字段：

```bash
ARK_IMAGE_EDIT_URL=https://visual.volcengineapi.com
ARK_MODEL=jimeng_t2i_v40
ARK_AUTH_MODE=aksk
ARK_ACCESS_KEY=your_ak
ARK_SECRET_KEY=your_sk

ARK_ACTION=CVSync2AsyncSubmitTask
ARK_GET_ACTION=CVSync2AsyncGetResult
ARK_VERSION=2022-08-31
ARK_REGION=cn-north-1
ARK_SERVICE=cv

ARK_TIMEOUT_SECONDS=120
ARK_POLL_INTERVAL_SECONDS=3
ARK_MAX_POLL_ATTEMPTS=25

ARK_INPAINT_ACTION=CVSync2AsyncSubmitTask
ARK_INPAINT_GET_ACTION=CVSync2AsyncGetResult
ARK_INPAINT_MODEL=jimeng_image2image_dream_inpaint
ARK_MASK_TRANSPORT_MODE=binary_append
```

## 当前 Ark 调用逻辑

`local_inpaint` 在 `ark_http` 模式下会：

1. 使用官方 `volcengine` Python SDK
2. 用 `AK/SK` 签名请求
3. 调用 `CVSync2AsyncSubmitTask`
4. 轮询 `CVSync2AsyncGetResult`
5. 解析返回的 `image_urls` 或 `binary_data_base64`

提交体当前会带上：

- 原图与参考图
- `id_mask / style_mask / accessory_mask / editable_hair_mask`
- 预处理结果
- 结构化妆发特征
- 强度控制参数
- 由参考解析结果生成的 prompt

## 当前状态说明

`ark_http` 链路已经真正打通到火山接口。

已验证结果：

- `AK/SK` 生效
- endpoint 生效
- action 生效
- 请求已经能到达火山视觉服务

说明：

- 使用无效字符串作为图片输入时，接口会返回 `Image Decode Error`
- 这说明当前剩下的问题不是鉴权，而是需要提供真实可用的图片 URL 或 base64

## 回退策略

如果真实服务未配置、请求失败、轮询超时或返回无法解析的结果，系统会自动回退到 `mock` 出图，并在候选图 metadata 中记录：

- `provider_mode`
- `provider_reason`

这样可以保证主流程不断。
