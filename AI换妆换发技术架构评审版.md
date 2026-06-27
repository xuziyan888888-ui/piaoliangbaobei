# AI 换妆换发技术架构评审版

## 1. 目标定义

本项目目标不是“换脸”，而是实现一条保身份的人像妆发迁移链路：

- 输入原图 A：保留人物身份
- 输入参考图 B：提取发型、刘海、彩妆和整体妆造风格
- 输出结果 C：仍然是 A 本人，但妆发效果接近 B

系统必须同时满足两类约束。

### 1.1 身份保真

- 脸型、五官、骨相、年龄感、表情和身份观感不被替换
- 肤色基调不被替换，但允许底妆层面的明度、均匀度和质感变化
- 眼镜、发箍等指定配饰不丢失

### 1.2 妆发迁移完整

- 发型轮廓、刘海形态、两侧鬓发、颅顶蓬松度可见迁移
- 底妆、腮红、眼妆、唇妆具有足够可感知强度

## 2. 架构结论

推荐架构不是单模型直出，而是五层流水线：

1. 预处理层：做人脸检测、解析分割、配饰保护、遮挡修复
2. 参考解析层：把参考图拆成结构化妆发特征
3. 生成层：执行“身份锁定 + 妆发迁移”
4. 后处理层：做边缘融合、像素回填、细节修复
5. 质检层：自动过滤不像本人或迁移不完整的结果

技术上建议采用：

- 主链路：全局参考生成
- 兜底链路：局部 inpaint 重绘
- 统一排序：基于身份分、妆发分、配饰分和瑕疵分做候选图重排

结论上，首版系统最稳的设计不是追求一次生成成功，而是追求：

- 多模块约束
- 主备链路互补
- 自动打分与自动重试

## 3. 总体架构图

```text
原图 A
  -> 预处理层
参考图 B
  -> 参考解析层
      -> 生成层
      -> 后处理层
      -> 质检层
      -> 结果输出
      -> 自动重试 / 切换兜底链路
```

## 4. 模块拆解

### 4.1 预处理层

目标：在进入生成前，明确三件事：

- 哪些区域必须保留
- 哪些区域允许编辑
- 哪些区域需要保护或回填

#### 4.1.1 人脸检测与姿态估计

可选实现：

- RetinaFace
- SCRFD

输出：

- `face_bbox`
- `yaw/pitch/roll`
- 粗人脸区域

#### 4.1.2 高精度关键点

可选实现：

- 106 点或更高精度 landmark 模型

输出：

- 眼、鼻、嘴、下颌线、眉毛位置
- 镜框近邻边界辅助信息

#### 4.1.3 人脸 / 头发 / 配饰解析分割

可选实现：

- BiSeNet face parsing
- BeautySeg 类解析模型
- 自定义语义分割模型

输出三类核心蒙版：

- `id_mask`：眼、鼻、脸型、骨相轮廓
- `style_mask`：头发、刘海、唇妆、腮红、眼妆、底妆可编辑区域
- `accessory_mask`：眼镜、发箍、耳饰

#### 4.1.4 遮挡补偿与蒙版修正

方法：

- 形态学扩张
- 发际线推断
- 镜框邻域剔除

目标：

- 修复头顶可编辑区域残缺
- 降低因发箍和碎发造成的分割不完整

建议额外输出：

- `editable_hair_mask`

#### 4.1.5 身份编码

可选实现：

- ArcFace
- AdaFace

输出：

- `id_embedding: 512-d`

#### 4.1.6 轻量 3D 面部结构估计

可选实现：

- 3DMM
- face mesh

输出：

- `face_mesh`
- 可选法线 / 曲率先验

#### 4.1.7 预处理层输出协议

```json
{
  "face_bbox": [0, 0, 0, 0],
  "pose": { "yaw": 0.0, "pitch": 0.0, "roll": 0.0 },
  "landmarks_106": "...",
  "id_mask": "binary mask",
  "style_mask": "binary mask",
  "accessory_mask": "binary mask",
  "editable_hair_mask": "binary mask",
  "id_embedding": "512-d vector",
  "face_mesh": "mesh tensor"
}
```

工程要点：

- `id_mask` 与 `style_mask` 不能重叠过大，否则生成控制会互相打架
- 眼镜必须作为单独保护区域处理，不能只依赖默认 face parsing 类别
- 发箍遮挡下的头顶区域建议单独生成 `editable_hair_mask`

### 4.2 参考解析层

目标：只读取参考图中的妆发风格，不引入参考人物身份。

#### 4.2.1 参考图分割

提取以下局部区域：

- 头发
- 刘海
- 嘴唇
- 眼妆
- 腮红

#### 4.2.2 视觉结构化解析

可选实现：

- VLM 结构化抽取
- 标签分类器 + 规则模板

输出设计原则：

- 不建议把参考解析写死成少量固定枚举示例
- 建议采用“固定骨架 + 开放取值 + 强度量化 + 缺失显式表达”的协议
- 结构上按头发、刘海、底妆、腮红、修容、高光、眉毛、眼妆、睫毛、卧蚕、唇妆、照片风格、整体氛围拆分
- 取值上允许模型根据参考图自由输出更细粒度标签，例如高马尾、无刘海、上扬眼线、灰棕眉色
- 数值上对关键风格项补充 `intensity`、`confidence`、局部位置和形状描述，方便后续生成控制

建议输出字段拆解如下。

发型维度：

- `hair.style`
- `hair.updo_type`
- `hair.length`
- `hair.parting`
- `hair.texture`
- `hair.color`
- `hair.volume_crown`
- `hair.volume_side`
- `hair.side_locks`
- `hair.hairline_exposure`

说明：

- `hair.updo_type` 不应只局限于丸子头、低盘发，也应支持高马尾、低马尾、半扎发、编发、短 bob、层次发等开放取值
- 发型控制不应只看类别，还应看颅顶高度、两侧蓬松度、发尾质感、鬓发存在与否

刘海维度：

- `bangs.exists`
- `bangs.type`
- `bangs.density`
- `bangs.length`
- `bangs.curve`
- `bangs.gap_ratio`

说明：

- 必须显式支持“无刘海”，不能强行归入某一种刘海类型
- `bangs.exists = false` 时，其余字段可为 `none` 或空值

妆容维度：

- `base_makeup`
- `blush`
- `contour`
- `highlight`
- `eyebrow`
- `eyeliner`
- `eyeshadow`
- `eyelashes`
- `aegyo_sal`
- `lips`

其中建议至少覆盖：

- 底妆：遮瑕度、明度提升、均匀度、光泽度、粉感、妆面完成度
- 腮红：颜色、位置、形状、范围、强度
- 修容：颜色、鼻影、下颌线、面中修容、强度
- 高光：颜色、鼻梁高光、颧骨高光、眼下提亮、强度
- 眉毛：眉型、眉色、粗细、眉峰、毛流感、强度
- 眼线：颜色、走向、长度、粗细、尾部方向、强度
- 眼影：主色、副色、分布区域、渐变方式、珠光或哑光质感、强度
- 睫毛：上睫毛密度、下睫毛密度、长度、卷翘度、簇感、强度
- 卧蚕：是否存在、提亮色、阴影色、形状、强度
- 唇妆：颜色、边缘清晰度、是否咬唇、光泽度、饱和度、强度

质感与风格维度：

- `skin_finish`
- `photo_style`
- `overall_vibe`

说明：

- `skin_finish` 表示底妆质感，例如 matte、semi_matte、creamy、dewy
- `photo_style` 表示照片呈现质感，例如 id_photo、studio、outdoor_natural、soft_light
- `overall_vibe` 表示整体审美氛围，例如 commute、clean、elegant、sweet、cool_tone
- 底妆质感与照片风格不能混在同一个字段中

#### 4.2.3 负向约束抽取

明确排除：

- 参考人物脸
- 参考人物肤色差异中的身份特征
- 参考图衣服
- 参考图背景

#### 4.2.4 输出协议

```json
{
  "hair_features": {
    "style": "updo",
    "updo_type": "high_ponytail",
    "length": "long",
    "parting": "center",
    "texture": "soft_wave",
    "color": {
      "label": "dark brown",
      "hex": "#4A352A",
      "confidence": 0.84
    },
    "volume_crown": 0.82,
    "volume_side": 0.46,
    "hairline_exposure": 0.58,
    "side_locks": {
      "exists": true,
      "length": "medium",
      "curl": 0.55,
      "intensity": 0.62
    }
  },
  "bangs": {
    "exists": false,
    "type": "none",
    "density": 0.0,
    "length": "none",
    "curve": "none",
    "gap_ratio": 0.0
  },
  "makeup_features": {
    "base_makeup": {
      "finish": "semi_matte",
      "coverage": 0.72,
      "brightness_shift": 0.18,
      "evenness": 0.81,
      "glow": 0.24,
      "powderiness": 0.16,
      "intensity": 0.68
    },
    "blush": {
      "color": "#E7A3B1",
      "placement": "central_cheek",
      "shape": "rounded",
      "range": 0.44,
      "intensity": 0.57
    },
    "contour": {
      "color": "#8B6A58",
      "nose_contour": 0.41,
      "cheek_contour": 0.36,
      "jaw_contour": 0.22,
      "intensity": 0.38
    },
    "highlight": {
      "color": "#F3E2D2",
      "nose_highlight": 0.48,
      "cheek_highlight": 0.35,
      "under_eye_highlight": 0.42,
      "intensity": 0.46
    },
    "eyebrow": {
      "shape": "soft_arch",
      "color": "#5A4638",
      "thickness": 0.43,
      "arch": 0.39,
      "hair_texture": 0.31,
      "intensity": 0.52
    },
    "eyeliner": {
      "color": "#3B2B24",
      "style": "upward_wing",
      "length": 0.58,
      "thickness": 0.33,
      "tail_direction": "up",
      "intensity": 0.63
    },
    "eyeshadow": {
      "main_color": "#B98F86",
      "secondary_color": "#7E5E59",
      "placement": "outer_corner_focus",
      "gradient": "soft",
      "finish": "matte",
      "intensity": 0.49
    },
    "eyelashes": {
      "upper_density": 0.64,
      "lower_density": 0.26,
      "length": 0.57,
      "curl": 0.61,
      "cluster_style": "natural_separated",
      "intensity": 0.54
    },
    "aegyo_sal": {
      "exists": true,
      "highlight_color": "#F2DDD3",
      "shadow_color": "#9C7B6A",
      "shape": "soft_parallel",
      "intensity": 0.32
    },
    "lips": {
      "color": "#C97E7B",
      "shape": "soft_full",
      "edge_blur": 0.52,
      "gloss": 0.18,
      "saturation": 0.46,
      "intensity": 0.59
    }
  },
  "texture_features": {
    "skin_finish": "semi_matte",
    "photo_style": "clean_portrait",
    "overall_vibe": "commute"
  },
  "negative_constraints": [
    "do not inherit reference identity",
    "do not inherit reference clothing",
    "do not inherit reference background"
  ]
}
```

工程要点：

- 参考解析结果建议结构化落盘，方便调参与回放
- VLM 输出不要直接原样拼 prompt，需要模板归一化
- 不要把少量示例字段误当成完整枚举表，协议应允许开放取值
- 所有缺失信息都应显式表达，例如 `bangs.exists = false`
- 妆容强度要转成可控数值字段，便于前端出强度滑杆
- 建议对关键字段同时输出 `label/value`、`intensity`、`confidence`，便于生成控制和质量分析

### 4.3 生成层

目标：在生成过程中实现身份锁定和妆发迁移的解耦控制。

#### 4.3.1 主链路：全局参考生成

输入：

- 原图 A
- `id_embedding`
- `id_mask`
- `style_mask`
- `editable_hair_mask`
- `accessory_mask`
- `face_mesh`
- 参考图结构化妆发特征

控制思路：

- 身份区域：高权重锁定
- 妆容区域：中高强度风格迁移
- 头发区域：高强度风格迁移
- 配饰区域：禁改或后回填

适用场景：

- 原图与参考图姿态差异不大
- 发型迁移需要较强整体感

风险：

- 风格强度过高会带来轻微换脸感
- 头发和脸部过渡处容易有生成痕迹

说明：

- 对外统一表述为“基于 Ark 的生成能力”
- 对内不把某一个具体底模写死，保留模型替换和路由空间

#### 4.3.2 兜底链路：局部 inpaint 重绘

输入：

- 原图 A
- 局部编辑蒙版，仅包含头发和彩妆区域
- 参考图妆发特征

控制思路：

- 蒙版外像素完全复用原图
- 蒙版内仅重绘头发与妆容

适用场景：

- 首版产品
- 对“像本人”要求极高的证件照或强正脸场景
- 全局链路 `identity_score` 不稳定时

风险：

- 全局发型体积变化受限
- 大幅发型改造能力弱于全局生成

成功案例说明：

- 成功案例中最终执行模型为“即梦AI-图片生成 4.0”
- 更准确的说法不是“抽象的 inpaint 概念成功”，而是前面工程链路先把任务整理成保身份的局部妆发编辑，再由即梦AI-图片生成 4.0 执行生成

#### 4.3.3 链路切换策略

建议按以下顺序执行：

1. 先跑主链路，生成 `N = 4~8` 张候选图
2. 做身份分和妆发完整度打分
3. 若最高分结果仍低于阈值，则切换局部 inpaint
4. 若局部 inpaint 仍不足，再触发自动 prompt 强化和重试

#### 4.3.4 生成层核心参数

首版建议对业务层只暴露少量抽象参数：

- `makeup_strength: 0~1`
- `hairstyle_strength: 0~1`
- `identity_lock_strength: 0~1`
- `preserve_accessories: bool`
- `mode: full_transfer | hair_only | makeup_only`

不要直接把底层扩散参数完整暴露给产品层，否则线上调参会失控。

### 4.4 后处理层

目标：降低生成痕迹，恢复成可交付照片。

#### 4.4.1 配饰像素回填

- 将 `accessory_mask` 区域回填原图像素
- 优先保护：眼镜、发箍、非编辑服饰边缘

#### 4.4.2 边缘融合

重点处理：

- 发际线
- 刘海与额头边缘
- 镜框与发丝交界
- 腮红与原皮肤边缘

#### 4.4.3 光影统一

- 保持原图主光方向
- 调整参考风格对原图曝光、白平衡的扰动

#### 4.4.4 细节增强

- 发丝修复
- 镜框边缘锐化
- 嘴唇和眼妆局部清晰度增强

工程要点：

- 回填发生在融合前后都可以尝试，但建议至少保留一个最终硬回填步骤
- 超分不要对整图统一做强增强，优先局部增强，否则会放大皮肤假感

### 4.5 质检层

目标：自动决定结果是否可用，而不是完全依赖人工挑图。

#### 4.5.1 指标设计

1. 身份相似度分
   计算方式：原图 vs 结果图的人脸 embedding 相似度
2. 妆发迁移完整度分
   检查发型轮廓、刘海、腮红、唇色、眼妆是否达到最小可见阈值
3. 配饰保留分
   检查眼镜是否存在、镜框是否连续、发箍是否丢失
4. 视觉瑕疵分
   检查发丝断裂、左右不对称、嘴部变形、眼镜穿模

#### 4.5.2 决策逻辑

建议使用统一排序分：

```text
final_score =
  0.45 * identity_score +
  0.30 * transfer_score +
  0.15 * accessory_score -
  0.10 * artifact_penalty
```

硬门槛规则：

- `identity_score` 低于阈值：直接淘汰，不参与妥协排序
- `accessory_score` 低于阈值：优先走回填或重试
- `transfer_score` 低于阈值：强化妆发 prompt 或切换链路再试

说明：

- `identity_score` 是第一优先级 gating 指标，不允许通过 `final_score` 高分来掩盖身份失败

## 5. 在线推理时序

```text
User
 -> Inference API
 -> Preprocess
 -> Ref Parser
 -> Generator
 -> Postprocess
 -> Quality Check
 -> 返回结果图和评分摘要

若质检不通过：
Quality Check -> 自动重试 / 切换兜底链路 -> 再评分 -> 返回最优结果
```

详细时序：

1. 用户提交原图 A、参考图 B 和控制参数
2. 原图进入预处理，输出身份向量、蒙版、mesh
3. 参考图进入结构化解析，输出妆发结构化特征
4. 主链路生成多候选图
5. 候选图进入像素回填与融合修复
6. 质检层评分排序
7. 返回最优结果或继续自动重试

## 6. API 设计建议

### 6.1 请求协议

```json
{
  "source_image": "url-or-base64",
  "reference_image": "url-or-base64",
  "mode": "full_transfer",
  "preserve_accessories": true,
  "makeup_strength": 0.75,
  "hairstyle_strength": 0.85,
  "identity_lock_strength": 0.95,
  "candidate_count": 6
}
```

### 6.2 返回协议

```json
{
  "job_id": "task_xxx",
  "result_image": "url",
  "scores": {
    "identity_score": 0.94,
    "transfer_score": 0.87,
    "accessory_score": 0.98,
    "artifact_penalty": 0.06,
    "final_score": 0.90
  },
  "metadata": {
    "pipeline": "global_then_postprocess",
    "retry_count": 1
  }
}
```

## 7. 模型与能力选型建议

建议区分“必选能力”和“可替换实现”，避免方案绑死在单点模型上。

### 7.1 必选能力

- 人脸检测
- 高精度关键点
- 人脸 / 头发 / 配饰分割
- 身份特征编码
- 参考图结构化解析
- 图像生成或局部重绘
- 结果质检

### 7.2 可替换实现

- 检测模型可替换
- 分割模型可替换
- VLM 可替换
- 生成模型可替换

换言之，真正需要固化的是模块职责，不是某一个具体模型名字。

## 8. 首版 MVP 建议

### 8.1 必做

1. 原图身份向量提取
2. `id_mask/style_mask/accessory_mask` 三类蒙版
3. 局部 inpaint 链路
4. 配饰像素回填
5. 身份分 + 配饰分双重质检

### 8.2 第二阶段补强

1. 全局生成主链路
2. VLM 结构化参考解析
3. 自动重试与候选排序
4. 光影统一与局部细节超分

### 8.3 第三阶段优化

1. 多参考图融合
2. 风格强度个性化控制
3. 发型模板检索与参考图相似检索
4. 用户编辑闭环

## 9. 验收指标建议

首版建议用一组可测指标来验收，而不是只看主观效果。

### 9.1 离线指标

- 身份相似度均值
- 身份相似度 P10
- 配饰保留成功率
- 发型迁移完整度通过率
- 妆容可见度通过率
- 生成失败率

### 9.2 在线指标

- 任务成功率
- 平均生成耗时
- 自动重试占比
- 用户二次生成率
- 用户保存率

### 9.3 人工评测维度

- 是否一眼看出还是本人
- 是否明显迁移了参考发型
- 是否明显迁移了参考妆容
- 是否存在眼镜、发箍或边缘破损
- 是否有 AI 生成违和感

## 10. 风险点

### 10.1 技术风险

1. 分割失效
   发箍、反光镜片、碎发、低对比背景会破坏分割质量
2. 身份约束不足
   风格迁移强时容易带入参考图脸感
3. 局部重绘边缘假
   发际线、镜框交界、额头阴影容易露馅
4. 参考图差异过大
   姿态差异、光线差异、发量差异过大时，迁移稳定性下降

### 10.2 产品风险

1. 用户期望“完全复制参考图”
   实际上系统必须优先保留本人身份，不能承诺 100% 复制
2. 用户上传图质量不稳定
   低清、遮挡、侧脸、强曝光会显著降低成功率

## 11. 推荐的评审结论

如果目标是尽快上线一个可用且稳定的版本，建议技术路线如下：

1. 首版先以局部 inpaint + 身份锁定 + 配饰回填为主
2. 全局参考生成作为增强链路，而不是唯一链路
3. 必须建设自动质检和候选排序，不要依赖单次出图
4. 方案对外统一表述为：基于 Ark 的生成能力，叠加自研工程化控制链路

这条路线的核心价值是：先保证“还是本人”，再逐步把“妆发更像参考图”做强。

## 12. 适合在评审会上直接说的总结

这不是单纯的参考图生成，也不是换脸，而是一条保身份的人像妆发迁移流水线。

技术关键不在某一个模型，而在于把预处理、区域约束、身份编码、参考解析、主备生成、配饰回填和自动质检整合成稳定系统。

从落地角度，建议先做局部可控、可验收的 MVP，再逐步补强全局发型迁移能力。
