# Ark 完整主链路研发任务单

更新时间：2026-06-27 Asia/Shanghai

## 1. 任务标题

对齐 Ark 完整主链路，区分主链路与 inpaint 兜底链路

## 2. 背景

当前项目目标始终没有变化：

- 保住原图身份
- 迁移参考图发型
- 迁移参考图妆容
- 保留眼镜、发箍等配饰

已知事实是：

- 你之前已经在 Ark 体系内成功实现过“保住本人脸 + 完整迁移妆发”
- 当前仓库虽然也在调用 Ark，但实际命中的是公开 `jimeng_image2image_dream_inpaint` 子能力
- 当前仓库没有复现成功时那套 `Ark 完整流水线原生底层 + 工程控制闭环`

因此，当前研发工作的重点不应再是继续盲调现有 `inpaint` 效果，而应先完成主链路对齐。

参考文档：

- [Ark成功链路对齐核查](/D:/水木年华/Ark成功链路对齐核查.md:1)
- [AI换妆换发研发实现方案](/D:/水木年华/AI换妆换发研发实现方案.md:1)
- [Reference Parser V2设计](/D:/水木年华/Reference Parser V2设计.md:1)

## 3. 当前状态

### 3.1 当前仓库已完成

- 两阶段局部编辑骨架已打通：
  - `hair_only`
  - `makeup_only`
  - stage 1 输出接 stage 2 输入
- 原图真实 mask 已落盘
- 参考图结构化解析已接入
- Ark inpaint 请求已修正为 `source + active_edit_mask`

关键文件：

- [`app/services/orchestrator.py`](/D:/水木年华/app/services/orchestrator.py:19)
- [`app/services/generator.py`](/D:/水木年华/app/services/generator.py:9)
- [`app/services/model_clients.py`](/D:/水木年华/app/services/model_clients.py:151)
- [`app/services/preprocess.py`](/D:/水木年华/app/services/preprocess.py:20)
- [`app/services/reference_parser.py`](/D:/水木年华/app/services/reference_parser.py:94)

### 3.2 当前仓库真实命中的能力

当前真实调用的是：

- `ARK_INPAINT_MODEL=jimeng_image2image_dream_inpaint`
- 异步接口：`CVSync2AsyncSubmitTask` / `CVSync2AsyncGetResult`
- 图像输入形态：`source_image + active_edit_mask`

这条链路属于：

- Ark 体系中的公开 `inpaint` 子能力

而不是：

- 成功样例对应的完整 Ark 主链路等价实现

### 3.3 当前已知问题

- 人脸仍会被美化改写
- 发型对参考图迁移不稳定
- 妆容更多表现为泛化美化，而不是完整参考迁移
- 当前 `global_reference` 仍是 mock，不是真实强参考链路

## 4. 核心结论

### 4.1 正确口径

项目后续统一口径如下：

- `Ark` 仍然是主链路平台
- 当前仓库只实现了 `Ark` 的公开 `inpaint` 子链路
- 当前仓库没有对齐到 `Ark` 完整流水线原生主链路

### 4.2 当前链路如何分层

主链路：

- `Ark 完整流水线原生底层`

兜底链路：

- `jimeng_image2image_dream_inpaint + two_stage_local_edit`

### 4.3 当前最不该做的事

在主链路未确认前，不应继续把主要精力放在：

- 微调现有 inpaint prompt
- 继续只补局部 mask
- 继续假设“再调一调参数就能等价复现成功链路”

这些动作只能优化兜底链路，不能把它升级为完整主链路。

## 5. 任务目标

本任务的目标不是“继续优化当前出图效果”，而是：

1. 明确成功链路在 Ark 体系中的能力形态
2. 明确当前仓库与成功链路之间缺失的模块层
3. 明确仓库接下来应补哪条主链路，而不是继续误把 inpaint 当主链路

## 6. 非目标

本任务暂不包含以下内容：

- 不继续盲调 `jimeng_image2image_dream_inpaint` 的出图参数
- 不以“修到当前效果更好一点”为阶段目标
- 不先做新的 UI 或前端配置面板
- 不先做新的大规模模型实验

## 7. 待办拆解

### P0：主链路能力确认

目标：

- 确认成功链路在 Ark 体系中的真实能力形态

待办：

1. 梳理你成功样例所依赖的页面能力名称、模式开关、输入形态
2. 对照 Ark 公开能力文档，区分：
   - 完整主链路能力
   - 公开 inpaint 子能力
   - 可能的页面原生编排层
3. 明确当前仓库缺失的是：
   - 主干能力入口
   - 页面编排层
   - 还是两者都有

产出物：

- 一份“Ark 主链路能力对齐结论”

### P0：代码现状与成功链路差异表

目标：

- 把当前仓库和成功链路差异从“感觉不一样”变成模块级差异清单

待办：

1. 列出成功链路应有模块：
   - 预处理修复
   - 身份锁定流
   - 参考图强输入
   - 双分支解耦
   - 动态区域权重
   - 后处理回填
   - 相似度过滤重试
2. 对照当前仓库逐项标记：
   - 已实现
   - 部分实现
   - 未实现

产出物：

- 一份“主链路模块差异清单”

### P1：仓库改造方案

目标：

- 在能力确认后，决定仓库如何接主链路

候选方向：

1. 新建真实 `global_reference` worker
2. 保留当前 `two_stage_local_edit` 作为兜底链路
3. 在编排层增加主链路 / 兜底链路切换策略

产出物：

- 一份“仓库改造设计说明”

### P1：验收与回归框架

目标：

- 防止后面又回到“主链路和兜底链路混在一起”的状态

待办：

1. 定义主链路验收样例
2. 定义兜底链路验收样例
3. 定义身份相似度、发型迁移、妆容迁移、配饰保留四个验收维度

产出物：

- 一份“主链路验收标准”

## 8. 建议执行顺序

建议按下面顺序推进：

1. 先完成 `Ark 主链路能力确认`
2. 再完成 `当前仓库 vs 成功链路模块差异表`
3. 再决定是否新建真实 `global_reference` worker`
4. 最后才进入代码改造与效果调优

## 9. 验收标准

本任务完成的标志不是“当前图看起来稍微更好了”，而是满足以下标准：

1. 能清楚区分：
   - Ark 主链路
   - Ark inpaint 兜底链路
2. 能明确说明当前仓库缺的是哪些模块层
3. 能明确下一步代码应该接哪条链路
4. 后续开发不再把 `jimeng_image2image_dream_inpaint` 误当完整主链路

## 10. 一句话结论

当前项目下一步最重要的不是继续调 `inpaint` 效果，而是先把仓库从“Ark 公开 inpaint 子链路”对齐到“Ark 完整主链路”。
