# Reference Parser Eval Set

这个目录用于做参考图解析器的多样本评测，不针对单一图片拟合。

## 文件结构

- `manifest.json`
  - 当前纳入评测集的样本清单
- `annotations/*.json`
  - 每张参考图对应一份轻量标注

## 使用方式

1. 初始化或刷新评测集：

```powershell
C:\Users\19770\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe scripts\bootstrap_reference_eval.py
```

2. 打开 `annotations/*.json`，只填你有把握的字段。

3. 运行评测：

```powershell
C:\Users\19770\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe scripts\run_reference_parser_eval.py
```

## 标注原则

- 只标高价值字段：
  - 发型主结构
  - 刘海
  - side locks
  - 眉毛颜色/冷暖/形状
  - 上眼皮/下眼皮/眼尾颜色
  - 眼影材质
  - 唇色/唇妆质地/冷暖
- 拿不准就留 `null`
- 不要为了提高覆盖率而硬填

## 状态字段

- `pending`
  - 模板刚生成，还没人工确认
- `ready`
  - 已做过基础人工标注，可以参与日常评测
- `approved`
  - 已确认质量较高，可以作为稳定回归样本
