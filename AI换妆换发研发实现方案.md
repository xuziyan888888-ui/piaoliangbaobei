# AI 换妆换发研发实现方案

## 1. 文档目标

本文档基于 [AI换妆换发技术架构评审版.md](D:/水木年华/AI换妆换发技术架构评审版.md)，将评审版中的总体方案进一步落地为可研发、可联调、可拆任务的实现方案。

本文重点回答五个问题：

1. 系统应该拆成哪些服务和模块
2. 一次在线推理任务的完整执行时序是什么
3. 核心输入输出协议怎么定义
4. 任务状态、质检、重试和链路切换怎么做
5. MVP 第一版如何控制范围并快速上线

## 2. 研发落地原则

### 2.1 产品目标优先级

第一优先级不是“尽量像参考图”，而是：

1. 还是本人
2. 配饰不丢
3. 妆发有明显变化

只有在身份稳定和配饰稳定的前提下，才逐步增强妆发还原度。

### 2.2 实现原则

- 先做稳定链路，再做增强链路
- 先做局部可控编辑，再做全局风格增强
- 先做自动质检，再做大规模放量
- 先固化模块职责，不固化单一底模

### 2.3 MVP 技术策略

第一版建议默认主打：

- 预处理
- 结构化参考解析
- 局部 inpaint 妆发迁移
- 配饰回填
- 身份质检

全局生成链路作为增强链路，先以开关或灰度方式接入。

## 3. 系统服务拆分

建议按“在线编排服务 + 模型能力服务 + 资产服务”拆分。

### 3.1 Inference API

职责：

- 对外接收请求
- 参数校验
- 创建任务
- 查询结果
- 返回评分摘要

建议接口：

- `POST /v1/makeup-transfer/jobs`
- `GET /v1/makeup-transfer/jobs/{job_id}`
- `GET /v1/makeup-transfer/jobs/{job_id}/artifacts`

### 3.2 Task Orchestrator

职责：

- 驱动完整任务状态机
- 调度预处理、参考解析、生成、后处理、质检
- 根据质检结果决定重试或切链路

建议能力：

- 串行主流程控制
- 候选图并行生成
- 自动重试策略
- 主备链路切换

### 3.3 Preprocess Service

职责：

- 人脸检测
- 关键点提取
- 人像分割
- 遮挡修复
- 身份编码
- 面部结构估计

输入：

- 原图

输出：

- `face_bbox`
- `pose`
- `landmarks_106`
- `id_mask`
- `style_mask`
- `accessory_mask`
- `editable_hair_mask`
- `id_embedding`
- `face_mesh`

### 3.4 Reference Parser Service

职责：

- 参考图局部分割
- 妆发结构化解析
- 负向约束提取
- 输出标准化 schema

输入：

- 参考图

输出：

- `hair_features`
- `bangs`
- `makeup_features`
- `texture_features`
- `negative_constraints`

### 3.5 Generation Router

职责：

- 根据任务模式决定走全局链路还是局部链路
- 接收质检反馈后切换生成链路
- 管理候选图生成次数和参数微调

输入：

- 预处理结果
- 参考解析结果
- 业务控制参数

输出：

- 候选图生成请求

### 3.6 Global Generation Worker

职责：

- 调用基于 Ark 的全局参考生成能力
- 批量生成候选图

定位：

- 增强链路
- 擅长整体发型变化和整体感迁移

### 3.7 Local Inpaint Worker

职责：

- 调用局部重绘能力
- 基于局部 mask 做妆容和头发重绘

定位：

- 首版主力链路
- 成功案例兜底链路

说明：

- 当前成功样本的最终执行模型可配置为“即梦AI-图片生成 4.0”

### 3.8 Postprocess Service

职责：

- 配饰回填
- 边缘融合
- 光影统一
- 局部细节增强

### 3.9 Quality Scoring Service

职责：

- 计算身份分
- 计算妆发迁移分
- 计算配饰保留分
- 计算瑕疵分
- 给出最终排序分

### 3.10 Artifact Store

职责：

- 存储输入图
- 存储中间蒙版
- 存储候选图
- 存储最终图
- 存储结构化解析结果和评分结果

建议：

- 原图、参考图、结果图放对象存储
- 中间结构化 JSON、评分结果、任务状态放数据库

## 4. 在线时序设计

## 4.1 主流程时序

```text
Client
 -> Inference API
 -> Task Orchestrator
 -> Preprocess Service
 -> Reference Parser Service
 -> Generation Router
 -> Global Generation Worker or Local Inpaint Worker
 -> Postprocess Service
 -> Quality Scoring Service
 -> Task Orchestrator
 -> Inference API
 -> Client
```

### 4.2 详细执行步骤

1. 用户提交原图、参考图和控制参数
2. `Inference API` 创建任务记录，返回 `job_id`
3. `Task Orchestrator` 拉起任务主流程
4. `Preprocess Service` 输出原图预处理结果
5. `Reference Parser Service` 输出参考图结构化特征
6. `Generation Router` 选择生成链路
7. 生成服务并行产出 `N` 张候选图
8. `Postprocess Service` 对每张候选图执行回填与融合
9. `Quality Scoring Service` 给每张候选图打分
10. `Task Orchestrator` 决定：
   - 直接输出
   - 调整参数重试
   - 切换到局部链路
11. 任务完成后回写最终图和评分摘要

### 4.3 推荐执行策略

MVP 第一版建议：

1. 默认先走局部链路
2. 候选图数量 `N=4`
3. 若局部链路妆发变化不足，再增强局部风格强度
4. 第二阶段再加入“先全局后局部”的混合策略

## 5. 任务状态机设计

建议任务状态如下：

- `created`
- `preprocessing`
- `parsing_reference`
- `generating`
- `postprocessing`
- `scoring`
- `retrying`
- `succeeded`
- `failed`

### 5.1 状态流转

```text
created
 -> preprocessing
 -> parsing_reference
 -> generating
 -> postprocessing
 -> scoring
 -> succeeded

若质检不通过：
scoring -> retrying -> generating

若重试次数超限：
retrying -> failed
```

### 5.2 失败原因分类

建议定义标准失败码：

- `FACE_NOT_FOUND`
- `REFERENCE_PARSE_FAILED`
- `MASK_INVALID`
- `IDENTITY_SCORE_TOO_LOW`
- `ACCESSORY_PRESERVE_FAILED`
- `TRANSFER_SCORE_TOO_LOW`
- `GENERATION_TIMEOUT`
- `POSTPROCESS_FAILED`

## 6. 核心数据结构设计

### 6.1 请求结构

```json
{
  "source_image": "url-or-base64",
  "reference_image": "url-or-base64",
  "mode": "full_transfer",
  "preserve_accessories": true,
  "makeup_strength": 0.75,
  "hairstyle_strength": 0.85,
  "identity_lock_strength": 0.95,
  "candidate_count": 4
}
```

### 6.2 任务主表

建议表名：`makeup_transfer_job`

建议字段：

- `job_id`
- `user_id`
- `status`
- `mode`
- `source_image_url`
- `reference_image_url`
- `result_image_url`
- `selected_pipeline`
- `retry_count`
- `identity_score`
- `transfer_score`
- `accessory_score`
- `artifact_penalty`
- `final_score`
- `failure_code`
- `failure_reason`
- `created_at`
- `updated_at`

### 6.3 候选图表

建议表名：`makeup_transfer_candidate`

建议字段：

- `candidate_id`
- `job_id`
- `pipeline_type`
- `generation_round`
- `image_url`
- `postprocessed_image_url`
- `identity_score`
- `transfer_score`
- `accessory_score`
- `artifact_penalty`
- `final_score`
- `is_selected`
- `metadata_json`
- `created_at`

### 6.4 预处理结果表

建议表名：`makeup_transfer_preprocess_result`

建议字段：

- `job_id`
- `face_bbox_json`
- `pose_json`
- `landmarks_url`
- `id_mask_url`
- `style_mask_url`
- `accessory_mask_url`
- `editable_hair_mask_url`
- `id_embedding_vector_ref`
- `face_mesh_ref`
- `created_at`

### 6.5 参考解析结果表

建议表名：`makeup_transfer_reference_parse`

建议字段：

- `job_id`
- `hair_features_json`
- `bangs_json`
- `makeup_features_json`
- `texture_features_json`
- `negative_constraints_json`
- `raw_vlm_output`
- `normalized_prompt_json`
- `created_at`

## 7. 生成路由与重试策略

### 7.1 链路选择规则

MVP 第一版建议默认策略：

- `mode = makeup_only`：优先局部链路
- `mode = hair_only`：优先局部链路
- `mode = full_transfer`：优先局部链路，后续再灰度全局链路

第二阶段建议策略：

- 姿态差异小、无遮挡、参考图整体发型变化明显：先全局链路
- 正脸强约束、眼镜明显、证件照、配饰强保护：先局部链路

### 7.2 质检触发重试规则

若满足以下任一条件，进入重试：

- `identity_score < identity_threshold`
- `accessory_score < accessory_threshold`
- `transfer_score < transfer_threshold`
- `artifact_penalty > artifact_threshold`

### 7.3 重试顺序建议

第一次重试：

- 保持链路不变
- 轻微增强妆容强度或发型强度

第二次重试：

- 强化负向约束
- 调整局部 mask 边界

第三次重试：

- 切换生成链路

建议重试上限：

- `max_retry_count = 3`

## 8. 评分实现建议

### 8.1 身份分

实现：

- 使用 ArcFace 或 AdaFace 重新编码原图和结果图
- 计算 cosine similarity

建议：

- 作为硬门槛，低于阈值直接淘汰

### 8.2 妆发迁移分

建议拆成两个子分：

- `hair_transfer_score`
- `makeup_transfer_score`

其中：

- 发型看轮廓、刘海、颅顶、鬓发
- 妆容看底妆、腮红、眼妆、唇妆的可见程度和匹配程度

### 8.3 配饰分

建议检查：

- 眼镜是否存在
- 镜框线条是否连续
- 发箍是否保留

### 8.4 瑕疵分

重点检查：

- 发际线脏边
- 镜框穿模
- 左右不对称
- 嘴型变形
- 眼周异常涂抹

### 8.5 最终排序

```text
final_score =
  0.45 * identity_score +
  0.30 * transfer_score +
  0.15 * accessory_score -
  0.10 * artifact_penalty
```

说明：

- `identity_score` 同时承担硬门槛角色
- `final_score` 用于通过候选图的最终排序，而不是兜底放过低身份分结果

## 9. 生成控制字段到模型控制的映射

前端不直接操作底层扩散参数，而通过业务字段映射到生成控制。

### 9.1 前端业务字段

- `makeup_strength`
- `hairstyle_strength`
- `identity_lock_strength`
- `preserve_accessories`
- `mode`

### 9.2 后端映射逻辑

示例：

- `makeup_strength`
  - 影响眼妆、唇妆、腮红、底妆的 `intensity` 放大系数
- `hairstyle_strength`
  - 影响头发轮廓、刘海、颅顶和鬓发的迁移强度
- `identity_lock_strength`
  - 影响身份区域 loss 权重或链路选择倾向
- `preserve_accessories`
  - 决定是否强制执行 `accessory_mask` 硬回填

## 10. 伪代码

### 10.1 主任务编排

```python
def run_makeup_transfer_job(job):
    preprocess = preprocess_source(job.source_image)
    ref_parse = parse_reference(job.reference_image)

    pipeline = select_pipeline(job, preprocess, ref_parse)
    retry_count = 0
    best_candidate = None

    while retry_count <= MAX_RETRY_COUNT:
        candidates = generate_candidates(
            pipeline=pipeline,
            job=job,
            preprocess=preprocess,
            ref_parse=ref_parse,
        )

        postprocessed = [postprocess(c, preprocess) for c in candidates]
        scored = [score_candidate(job.source_image, img, preprocess, ref_parse) for img in postprocessed]

        best_candidate = pick_best_valid_candidate(scored)
        if best_candidate is not None:
            return finalize_job(job, best_candidate, pipeline, retry_count)

        retry_count += 1
        pipeline, job, preprocess, ref_parse = prepare_retry(
            pipeline, job, preprocess, ref_parse, retry_count
        )

    return fail_job(job, "NO_VALID_CANDIDATE")
```

### 10.2 链路选择

```python
def select_pipeline(job, preprocess, ref_parse):
    if job.mode in ["makeup_only", "hair_only"]:
        return "local_inpaint"

    if has_strong_accessory(preprocess) or is_id_photo_like(job.source_image):
        return "local_inpaint"

    if pose_gap_is_small(preprocess, ref_parse):
        return "global_reference"

    return "local_inpaint"
```

### 10.3 候选图校验

```python
def pick_best_valid_candidate(scored_candidates):
    valid = []
    for c in scored_candidates:
        if c.identity_score < IDENTITY_THRESHOLD:
            continue
        if c.accessory_score < ACCESSORY_THRESHOLD:
            continue
        if c.transfer_score < TRANSFER_THRESHOLD:
            continue
        valid.append(c)

    if not valid:
        return None

    valid.sort(key=lambda x: x.final_score, reverse=True)
    return valid[0]
```

## 11. MVP 研发排期建议

### 11.1 第一阶段

目标：

- 打通最小可用链路

范围：

1. Inference API
2. Task Orchestrator
3. Preprocess Service
4. Reference Parser Service 基础版
5. Local Inpaint Worker
6. Postprocess Service 基础回填
7. Quality Scoring Service 基础版

### 11.2 第二阶段

目标：

- 提升效果和通过率

范围：

1. Global Generation Worker
2. 自动重试
3. 候选图排序
4. 更细粒度妆容解析
5. 局部细节增强

### 11.3 第三阶段

目标：

- 增强用户可控性和复杂场景覆盖

范围：

1. 多参考图
2. 用户可编辑局部蒙版
3. 发型模板召回
4. 风格偏好学习

## 12. 研发分工建议

可按以下方向拆分任务：

### 12.1 后端编排

- API 接入
- 任务系统
- 状态机
- 数据库与对象存储

### 12.2 CV 与预处理

- 检测
- 分割
- mask 修正
- 身份编码
- 3D 结构估计

### 12.3 参考解析与提示工程

- 结构化 schema
- VLM 解析归一化
- 负向约束生成
- 业务字段映射

### 12.4 生成链路

- 全局链路接入
- 局部链路接入
- 生成参数模板
- 多候选图生成

### 12.5 后处理与质检

- 配饰回填
- 边缘融合
- 评分模型
- 候选图排序

## 13. 最终建议

从研发视角，最关键的不是先追求一个“最强模型”，而是尽快搭出一条能稳定跑通的系统链路。

推荐执行顺序：

1. 先把 `id_mask/style_mask/accessory_mask + identity_score` 跑通
2. 先接局部 inpaint 链路做 MVP
3. 先做自动质检，不依赖人工挑图
4. 再逐步加入全局生成链路和更复杂的参考解析

这样最容易产出第一版“明显变妆变发，但还是本人”的稳定结果。
