# SageAttention T4 (sm75) BF16 兼容修复

> 状态：已提交
> 日期：2026-08-28
> 涉及：`flash_head/src/modules/flash_head_model.py`

## 问题

T4 (sm75, Turing) + 社区魔改 sageattention 的 ComfyUI 环境运行报错：

```
RuntimeError: at::Tensor qk_int8_sv_f16_accum_f32_attn(...) failed to dispatch data type BFloat16
```

触发点：`flash_head_model.py` 的 `flash_attention()` → `SAGE_ATTN_AVAILABLE` 分支 → `sageattn(q, k, v)` → 魔改版直达 sm80 CUDA kernel。

## 根因

1. **Turing tensor core 无 BF16 MMA**。T4 (sm75) 原生只支持 fp16/int8/int4，bf16 需要 sm_80+（官方 issue #63/#137 确认，ptxas 报 `Feature '.bf16' requires sm_80`）。
2. 模型 `param_dtype=torch.bfloat16`（`flash_head_pipeline.py:62`），Q/K/V 全是 bf16。
3. 社区魔改 sageattention 的 SM75 内核（`qk_int8_sv_f16_accum_f32_attn`）**只 dispatch FP16**，收到的输出张量是 bf16 → dispatch 失败。
4. 官方 `sageattn()` 路由对 sm75 直接 `raise ValueError: Unsupported CUDA architecture: sm75`（core.py L143-157），根本不提供 bf16 路径。

## 修复方案

**保留 T4 上的 SageAttention 加速**（不降级、不禁用），pre-Ampere 强制 **FP16 cast**：

| 函数 | 改动 |
|------|------|
| `_is_pre_ampere()` | 新增：`device_capability` major < 8（Turing/Volta：T4/V100 等） |
| `_run_sageattn()` | 新增：rearrange 到 `b n s d` → **pre-Ampere 时 `q.half()/k.half()/v.half()`** → `sageattn()` → 回排。try/except 包裹，kernel 真不可用时才 fallback SDPA（最后防线） |
| `flash_attention()` | SAGE 分支改为调用 `_run_sageattn()`，**只要安装了 sageattention 就启用**（无 capability 禁用门控） |
| `SelfAttention.use_usp` | 保持无条件 `AttnType.SAGE_AUTO`（同步，不因 capability 禁用） |
| 新增 `from loguru import logger` | `_run_sageattn` 的 warning 需要 |

### 关键决策

- **不再是**"sm75 禁用 sageattn → SDPA"。那样会丧失用户装魔改 sageattention 想要的加速。
- **不是** `.to(v.dtype)` 简单对齐——那只是把 q/k/v 对齐到同为 bf16，仍会触发 bf16 dispatch 失败。必须 `.half()` 显式转 FP16。
- 与社区 SM75 fork（XUANNISSAN/SageAttention-SM75-path、FearL0rd）的通行做法一致：**FP16 输入 → int8 QK + FP16 PV kernel 正常 dispatch**。
- Ampere+ (sm80+) 保持原 dtype（bf16）——官方支持，无需 cast，不损失精度。

## 备选方案（未采用，供后续参考）

- **官方 SageAttention-1 分支**（纯 Triton）：官方对 pre-Ampere 的建议，但需要换装包，且魔改 SM75 fork 已能跑。
- **全局模型 fp16**：把 `param_dtype` 从 bf16 改 fp16。可根治但会损失精度、且需确认 VAE/wav2vec 兼容性——不如仅在 attention 输入处 cast 精准。

## 验证

- [x] `py_compile` 通过
- [x] AST 检查：`_run_sageattn` = pre-Ampere fp16-cast + sageattn + SDPA 仅兜底
- [x] 无 `_sageattn_supported` 残留引用（上一版错误开关已移除）
- [x] SAGE 分支条件 == `SAGE_ATTN_AVAILABLE`（装了就启用，无条件）
- [x] T4 上魔改 sageattention 正常 dispatch（fp16），加速保留