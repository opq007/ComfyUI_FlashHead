# FlashHead ComfyUI 节点：长视频内存优化 + 非方形输出支持

> 状态：已批准，待实施
> 日期：2026-08-28
> 参考项目：https://github.com/Bilal140202/TalkDrive_by_BilalAnsari（作者博客：https://ansaribilal.com/blog/talkdrive-soulx-flashhead-colab-talking-head-2026/）

## 背景与目标

当前 `ComfyUI_FlashHead` 实现存在两个已知局限：

1. **长视频内存不友好** —— 所有分块帧被 `generated_list` 全量累积在 CPU RAM 中，内存随视频长度线性增长，容易爆内存。
2. **只支持方形(512×512)输出** —— 模型训练于 512×512 人脸裁剪，原图被压缩，方形以外的宽高比不友好。

目标：将 TalkDrive 项目的相关改良思路迁移到本项目中，并额外修复长视频的真实内存瓶颈。

---

## 关键发现（调研结论）

### TalkDrive 真正解决了什么

TalkDrive 针对 SoulX-FlashHead 源码打了 **3 处补丁**：

| Patch | 目标文件 | 作用 |
|-------|---------|------|
| A | `flash_head/utils/cpu_face_handler.py` | 用 **OpenCV DNN(ResNet-SSD) + Haar cascade** 重写人脸检测器，移除 mediapipe 依赖（mediapipe 原生库与 CUDA 运行时冲突）。接口 `(bboxes, scores)` 相对坐标 `[0,1]` 不变。 |
| B | `flash_head/utils/facecrop.py` | 增加 **bbox 落盘**：把人脸裁剪的精确像素坐标 `{bbox, img_w, img_h}` 写入临时 JSON，供合成器读取粘贴位置。 |
| C | 应用入口(gradio_app.py) | 注入 **原分辨率/16:9 合成器** `composite_to_original()`：读 bbox → 跳过近方形原图 → 逐帧把 512×512 clip resize 到 bbox 区域 → 羽化混合回原图画布 → ffmpeg 重封装音频。 |

### TalkDrive 没有解决什么

TalkDrive **没有改动音频分块/长视频内存**。其对话中体现的"5 秒音频分片"是 **SoulX 原生的滑动窗口分块**（`generate()` 中 `slice_len = frame_num — motion_frames_num`，每 chunk 丢弃重叠帧、只解码新帧），当前仓库**已经实现**（`nodes.py` 的 `stream` 模式，`deque` 滑窗）。

**结论**：TalkDrive 只解决"方形输出"（Patch C + bbox 落盘），长视频内存问题需要在本项目单独设计。

### 当前仓库长视频的真实瓶颈

是 **RAM（CPU 内存），不是 VRAM**。`nodes.py` 190–209 行：

```python
generated_list = []                      # 全部累积
for chunk in slices:
    video = run_pipeline(pipeline, ...)  # 每 chunk 产出 ~slice_len 帧
    generated_list.append(video.cpu())   # ← 每个 chunk 都 CPU() 进内存列表
...
self.save_video(generated_list, ...)     # 全部保留到循环结束才写盘
```

即使每 chunk 只产出 ~24 帧，`generated_list` 仍把**整段长视频的帧全部堆在 CPU RAM**。5 分钟视频 ≈ 7500 帧，峰值几百 MB~数 GB，且随长度线性增长。

---

## 迁移改动点

### 迁移范围 1 — 人脸检测选用 MediaPipe（不采用参考的 OpenCV 替换）

**文件**：`flash_head/utils/cpu_face_handler.py`（保持原样，未改）

参考项目将 mediapipe 替换为 OpenCV DNN/Haar，**是因为它在 Colab 上装不上 mediapipe（与 CUDA/Py3.11 冲突）**，属被迫 workaround，而非性能更优方案。本项目的 ComfyUI 环境 mediapipe 可用。

效率对比：

| 维度 | 原始 MediaPipe | OpenCV DNN/参考 |
|------|---------------|-----------------|
| 速度 | **~2 ms/帧**（专用轻量模型） | ResNet-SSD CPU 下 ~30–100+ ms/帧；Haar 更慢 |
| 精度 | 对侧脸/遮挡鲁棒 | ResNet-SSD 尚可，Haar 老旧易漏检 |
| 依赖 | 已装 mediapipe | 需 opencv + 额外下载 caffemodel/prototxt |

**决策**：使用本项目原始 MediaPipe 实现更高效，故**不迁移** OpenCV 替换。仅保留参考项目 Patch B（bbox 落盘）——其与检测器实现无关，facecrop 通过 `CPUFaceHandler.detect()` 接口（相对坐标）获取 bbox，mediapipe 与 OpenCV 均可满足。

> 注意：`cpu_face_handler.py` 若要启用 `use_face_crop`/`composite_to_full`，其 `detect()` 必须返回相对坐标 `[0,1]` 的 bbox + scores；MediaPipe 原实现满足此契约。

### 迁移范围 2 — bbox 落盘（对应 TalkDrive Patch B，适配 ComfyUI）

**文件**：`flash_utils/facecrop.py`

在 `get_scaled_bbox()` 内，将 `scaled_bbox + [img_w, img_h]` 写入临时 JSON。**改动点**：

- 路径从参考项目的 `/tmp/_talkdrive_crop_bbox.json` 改为 **ComfyUI 临时目录**（`folder_paths.get_temp_directory()`，与 `nodes.py` 中现有音频/图像临时文件一致），避免硬编码 /tmp。
- **track 唯一化**：用 uuid 使每次合成的 bbox 文件唯一，避免多节点并发互相串（ComfyUI 可能并行跑多个工作流）。
- 保留"bbox 文件缺失则合成器忽略"的非致命兜底。
- bbox 文件路径需传递到合成器。

### 迁移范围 3 — 合成器（对应 TalkDrive Patch C → 纯函数化）

**文件**：新增 `flash_utils/composite.py`

把 `composite_to_original()` 落成独立模块，改造点：

- 路径从 `/tmp` 改为传入 bbox 文件路径参数。
- 函数签名：`composite_to_original(video_path, original_image_path, bbox_path)`。
- 逐帧读原视频 → 缩放到 bbox → 羽化混合回原图画布 → 流式写 → ffmpeg 重音。
- 跳过近方形原图（宽高比 < 1.15 时不合成，直接返回）。
- 本身已逐帧流式处理，内存安全。

### 迁移范围 4 — 长视频内存修复：双缓冲异步落盘（本仓库真实瓶颈）

**文件**：`nodes.py`

**核心改造**：用**边算边写 + 异步双缓冲**替换 `generated_list` 全量缓存。

方案要点：

1. 循环前**打开 imageio writer** 指向临时 raw 视频。
2. 每个 chunk 的 `video.cpu()` 落为 Numpy 帧，放入**有界队列**（容量 N=2~4）即返回，不等落盘。
3. 一个**后台线程**消费队列，逐帧写入 imageio writer。
4. 循环结束后关闭 writer，ffmpeg 合成音频（保留现有 merge 逻辑）。

内存收益：从"整个视频的帧"降为"单 chunk + 队列缓冲"的恒定值。

速度影响：**不触碰 GPU 推理路径**（DiT denoise + VAE decode 是耗时大头）。双缓冲使 CPU 落盘与下一 chunk 的 GPU 推理**重叠**，对总耗时影响 ≈ 0。绝不能使用"先全算完再统一写盘"的朴素串行（会让 GPU 在落盘时空等，可能拖慢）。

### 迁移范围 5 — 接线与开关（Sampler 节点参数）

**文件**：`nodes.py` 的 `RunningHub_FlashHead_Sampler`

新增可选参数：

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `use_face_crop` | BOOLEAN | `False` | 是否启用 face crop。开启时可获得 bbox |
| `composite_to_full` | BOOLEAN | `False` | 是否将 512×512 合成回原图分辨率/宽高比 |

- 接喂 `prepare_params(... us_face_crop=use_face_crop)`。
- `composite_to_full` 开启且 `use_face_crop` 开启时，在流式写盘/合成视频后调 `composite_to_original()` 得到原分辨率输出。
- **默认全部关闭** → 行为与现状完全一致，无缝兼容现有工作流。

---

## 实施顺序（每步独立可验证）

1. 范围 1：`cpu_face_card.py` → OpenCV 检测。
2. 范围 2：`faecause.py` → bbox 落盘。
3. 范围 3：新增 `composite.py` 合成器。
4. 范围 4：`nodes.py` → 流式异步落盘。
5. 范围 5：Sampler 节点参数接线。

依赖关系：1→2→3 功能链；4 独立；5 聚合 2/3/4。

---

## 验证检查单

- [ ] 使用 MediaPipe 人脸检测（保留原始高效实现），未被替换为更慢的 OpenCV。
- [ ] face crop 后存在唯一性 bbox JSON 文件。
- [ ] composite_to_original() 对 16:9 原图输出原分辨率视频；近方形原图直接跳过。
- [ ] 长视频（≥2 分钟）生成时内存峰值恒定（不随时长线性增长）。
- [ ] 默认参数（use_face_crop=False, composite_to_full=False）下行为与现状一致。
- [ ] 生成速度不因流式落盘而回退（对比同输入）。