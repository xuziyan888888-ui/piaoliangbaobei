# Reference Parser V2 设计

## 1. 目标

当前项目中的参考图提取如果继续依赖整图裁块、明暗统计和颜色启发式规则，会持续出现这些问题：

- 把盘发、后梳发误判成披发
- 把无刘海误判成有刘海，或反过来
- 把半光泽奶油肌误判成雾面底妆
- 无法稳定识别修容、高光、眉型、睫毛、眼线形状

所以 `Reference Parser v2` 的目标不是“补更多阈值”，而是把参考图解析升级成真正的语义理解链路：

1. 先做人脸妆发分区，不再直接整图猜
2. 发型提取改成结构识别，不再只看颜色统计
3. 妆容提取改成部位理解，不再只取唇色、腮红和底妆 finish
4. 最后由 VLM 做整体验证和语义补充

一句话总结：

- V1 是像素启发式猜测
- V2 是“分区 -> 结构识别 -> 部位理解 -> VLM 总结”的多阶段解析

## 2. 总体流程

建议将参考图解析拆成四个连续模块：

1. Region Parser
2. Hair Structure Parser
3. Makeup Attribute Parser
4. VLM Validator and Summarizer

执行顺序如下：

```text
reference image
  -> face / hair / accessory segmentation
  -> fine-grained makeup region extraction
  -> hair structure classification
  -> makeup component classification and regression
  -> VLM consistency check and caption completion
  -> normalized reference schema
```

## 3. 阶段一：人脸妆发分区

V2 必须先切出局部区域，而不是先做整图判断。建议至少输出以下区域：

- 头发整体
- 刘海
- 发际线
- 眉毛
- 上眼皮
- 下眼睑
- 睫毛
- 唇部
- 腮红区
- 鼻梁高光区
- 脸侧修容区

这一层的目的不是直接生成最终标签，而是保证后面的判断全部建立在干净局部区域上。

这一层缺失时，系统最容易出现两类致命误判：

- 发型结构误判：盘发看成披发，贴头后梳看成偏分披发
- 底妆质感误判：高光和阴影被混进整体肤质，导致半光泽肌被误判成纯雾面

### 3.1 推荐输出

```json
{
  "region_masks": {
    "hair": "mask",
    "bangs": "mask",
    "hairline": "mask",
    "brow_left": "mask",
    "brow_right": "mask",
    "upper_eyelid_left": "mask",
    "upper_eyelid_right": "mask",
    "lower_eyelid_left": "mask",
    "lower_eyelid_right": "mask",
    "eyelashes_upper": "mask",
    "eyelashes_lower": "mask",
    "lips": "mask",
    "blush_left": "mask",
    "blush_right": "mask",
    "nose_highlight": "mask",
    "contour_left": "mask",
    "contour_right": "mask"
  }
}
```

## 4. 阶段二：发型结构识别

发型提取不能再只输出一个 `style=down`。建议拆成两层。

### 4.1 基础结构层

用于判断大类：

- 披发
- 盘发
- 高马尾
- 低盘发
- 半扎发
- 短发
- bob
- 双丸子
- 后梳贴头盘发

### 4.2 属性层

用于补充分布和细节：

- 分缝
- 刘海
- 长度
- 卷度
- 蓬松度
- 鬓发
- 发际线露出
- 发色
- 发丝光泽
- 服帖度

### 4.3 设计原则

- 先判断结构，再补充属性，不要反过来
- `style` 只负责大类，不能承载所有细节
- `updo_type` 需要开放取值，不能只写死丸子头、低盘发
- 必须显式支持“无刘海”
- 发型描述必须兼顾“形状、走势、材质、暴露程度”四个维度

### 4.4 建议结构

```json
{
  "hair_features": {
    "primary_style": "slicked_back_updo",
    "secondary_style": "tight_bun",
    "length": "long",
    "parting": "none_or_natural_back",
    "bangs": {
      "exists": false,
      "type": "none"
    },
    "texture": "straight_sleek",
    "hair_color": "natural_black",
    "crown_volume": 0.42,
    "side_volume": 0.12,
    "hairline_exposure": 0.93,
    "side_locks": {
      "exists": false
    }
  }
}
```

## 5. 阶段三：妆容部位理解

妆容提取不能只看：

- 唇色
- 腮红颜色
- 底妆 finish

V2 需要把妆容拆成真正可控的部位理解对象。

### 5.1 底妆

至少提取：

- 光泽度
- 遮瑕感
- 提亮度
- 均匀度
- 粉感
- 完妆质感

### 5.2 眉毛

至少提取：

- 眉型
- 眉色
- 粗细
- 眉峰
- 毛流感

### 5.3 眼妆

至少提取：

- 眼影主色
- 眼影辅色
- 眼影范围
- 眼线样式
- 眼线颜色
- 睫毛浓密度
- 睫毛卷翘度
- 下眼妆存在感

### 5.4 修容和高光

至少提取：

- 修容颜色
- 鼻修容强度
- 脸侧修容强度
- 高光颜色
- 鼻梁高光分布
- 鼻尖高光分布
- 面中高光分布

### 5.5 唇妆

至少提取：

- 唇色
- 边缘清晰度
- 光泽
- 饱和度
- 唇峰形状
- 是否咬唇

### 5.6 建议结构

```json
{
  "makeup_features": {
    "base": {
      "finish": "semi_glowy",
      "coverage": 0.78,
      "brightness": 0.64,
      "powderiness": 0.18
    },
    "brows": {
      "shape": "straight_soft_arch",
      "color": "dark_gray_brown",
      "hair_texture": "defined_hair_strokes"
    },
    "eyes": {
      "eyeshadow_main": "peach_brown",
      "eyeliner_style": "thin_lifted",
      "eyeliner_color": "black",
      "lash_density": 0.72,
      "lash_curl": 0.76
    },
    "contour": {
      "color": "taupe_brown",
      "nose_contour": 0.46,
      "face_contour": 0.38
    },
    "highlight": {
      "color": "ivory_champagne",
      "nose_bridge": 0.62,
      "nose_tip": 0.58,
      "forehead": 0.35
    },
    "blush": {
      "color": "soft_peach_pink",
      "intensity": 0.34
    },
    "lips": {
      "color": "cool_true_red",
      "gloss": 0.18,
      "saturation": 0.88,
      "edge_definition": 0.66
    }
  }
}
```

## 6. 阶段四：VLM 整体验证和补充

前面三个阶段主要负责结构化字段，最后再让 VLM 负责两件事：

1. 检查结构化字段之间是否互相矛盾
2. 生成整体语义总结，补充人类可读描述

VLM 适合补的内容包括：

- 后梳贴头盘发
- 冷艳红唇
- 野生毛流眉
- 半光泽奶油肌

这一层的职责不是替代前面的分割与分类，而是：

- 做 consistency check
- 做 caption completion
- 做最终 prompt 归一化

### 6.1 建议输出

```json
{
  "style_caption": "slicked-back clean updo with radiant skin and statement red lips",
  "consistency_flags": [],
  "field_confidence_overrides": {
    "hair.primary_style": 0.92,
    "makeup.base.finish": 0.87
  }
}
```

## 7. V1 与 V2 的本质差异

| 维度 | V1 | V2 |
| --- | --- | --- |
| 输入理解方式 | 整图裁块 + 颜色/明暗启发式 | 分割优先 + 结构识别 + 部位理解 |
| 发型判断 | 容易误把盘发看成披发 | 先判断发型结构，再补属性 |
| 刘海判断 | 靠局部暗区猜测 | 有独立 bangs 区域和显式 exists 字段 |
| 底妆判断 | 容易把高光混进 finish | 先拆区域，再判断 finish / glow / powderiness |
| 妆容字段 | 少量粗粒度字段 | 可控的多部位结构化字段 |
| 可扩展性 | 调阈值成本高，上限低 | 可替换模型，可扩 schema |

## 8. 开发落地顺序

建议按以下顺序升级，不要一次全重写。

### 8.1 第一步：先升级 schema

先把当前 `ReferenceParseResult` 和相关字段定义扩成 V2 需要的结构，哪怕初始阶段先允许一部分字段为空。

目标：

- 先解决“结构表达能力不够”的问题
- 不让后续模型能力被旧字段限制

### 8.2 第二步：替换 `reference_parser.py`

把当前启发式参考图解析器升级为多阶段解析器：

- region extraction
- hair structure parser
- makeup attribute parser
- VLM summary merger

这个阶段允许保留 fallback 规则，但 fallback 只作为兜底，不能再作为主逻辑。

### 8.3 第三步：接入生成层

把新的字段真正接进：

- prompt 构建
- `provider_request_features`
- 调试输出
- 生成日志

不要只解析出来却不喂给模型。

### 8.4 第四步：接入可视化验图

建议每次解析输出：

- 原图
- 参考图
- 区域分割可视化
- 结构化 JSON
- 中英双语摘要

这样才能快速定位是“解析错了”还是“生成没听话”。

## 9. 对当前项目的直接结论

基于当前项目现状，可以明确下结论：

- 继续微调 `reference_parser.py` 的阈值，不足以达到产品级效果
- 当前方案还能作为 demo 或 fallback，但不能作为正式参考图理解主链路
- 如果目标是“参考图发型和妆容提取像人工描述一样稳定”，必须切到 V2 路线

因此，后续研发建议统一按以下表述推进：

- 当前版本：启发式参考图提取
- 下一版本：分割优先的结构化参考图解析 V2
