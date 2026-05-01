# train.py 和 model.py 参数接口分析

## 概览
该文档分析了当前代码中的参数复杂性，并识别出可能需要保留或删除的参数。

---

## 一、训练脚本 (train.py) 参数列表

### 1. 核心训练参数 (必需，广泛使用)
| 参数 | 类型 | 默认值 | 使用位置 | 建议 |
|------|------|--------|---------|------|
| `model_type` | str | "auto" | build_model(), loss_preset 选择, 数据集选择 | ✅ 必需 |
| `output_dir` | str | runs/ | 日志和检查点保存 | ✅ 必需 |
| `experiment_name` | str | "run01" | 创建exp_dir，保存结果 | ✅ 必需 |
| `batch_size` | int | 32 | DataLoader | ✅ 必需 |
| `epochs` | int | 30 | 训练循环 | ✅ 必需 |
| `lr` | float | 2e-4 | 优化器 | ✅ 必需 |
| `weight_decay` | float | 1e-4 | 优化器 | ✅ 必需 |
| `seed` | int | 42 | seed_everything(), 数据集分割 | ✅ 必需 |

### 2. 数据路径参数
| 参数 | 类型 | 默认值 | 使用位置 | 建议 |
|------|------|--------|---------|------|
| `train_embeddings_dir` | str | data/train/alphaearth_emb | 加载数据 | ✅ 必需 |
| `train_targets_dir` | str | data/train/labels | 加载标签 | ✅ 必需 |
| `secondary_train_embeddings_dir` | str | None | 可选第二个嵌入源（Tessera） | ⚠️ 条件必需 |
| `token_train_embeddings_dir` | str | None | 可选 16x16 token 嵌入 | ⚠️ 条件必需 |
| `secondary_token_train_embeddings_dir` | str | None | 可选第二 token 源 | ⚠️ 条件必需 |

### 3. 模型架构参数 (传给 build_model)
| 参数 | 类型 | 默认值 | 影响范围 | 建议 |
|------|------|--------|---------|------|
| `lightunet_base_ch` | int | 32 | UNet 编码器宽度 | ✅ 保留 |
| `lightunet_norm_kind` | str | "bn" | BatchNorm vs GroupNorm | ✅ 保留 |
| `tessera_presence_ch` | int | 16 | Tessera 压缩器宽度 | ✅ 保留 |
| `tessera_hidden_ch` | int | None | Tessera 压缩器内部宽度 | ✅ 保留 |
| `tessera_hidden_depth` | int | 0 | Tessera 压缩器额外块数 | ⚠️ 实验性参数 |
| `height_specialist_depth` | int | 0 | 高度专家投影块数 | ⚠️ 实验性参数 |
| `height_gate_source` | str | "alpha" | 高度门源（alpha/fused） | ✅ 保留 |
| `height_hidden_ch` | int | None | 高度主干内部宽度 | ✅ 保留 |
| `height_trunk_depth` | int | 2 | 高度主干块数 | ✅ 保留 |
| `height_independent_branches` | bool | False | 是否使用独立高度分支 | ⚠️ 实验性参数 |
| `height_head_kind` | str | "linear" | 输出头类型 (linear/softbin) | ✅ 保留 |
| `height_n_bins` | int | 64 | softbin 的 bin 数量 | ✅ 保留 |
| `height_bin_max_m` | float | 80.0 | softbin 最大高度 (米) | ✅ 保留 |
| `gate_mode` | str | "simple" | 融合门类型 (simple/rich) | ✅ 保留 |
| `gate_untied` | bool | False | 是否使用非绑定门 | ⚠️ 实验性参数 |
| `gate_init_bias` | float | 4.0 | 门初始化偏置 | ⚠️ 微调参数 |
| `modality_dropout` | float | 0.0 | Tessera 分支丢弃概率 | ⚠️ 实验性参数 |

### 4. 损失函数参数
| 参数 | 类型 | 默认值 | 使用位置 | 建议 |
|------|------|--------|---------|------|
| `loss_preset` | str | "auto" | ImprovedCompositeLoss 类型选择 | ✅ 保留 |
| `aux_weight` | float | 1.0 | 辅助监督权重 | ✅ 保留 |
| `presence_tversky_weight` | float | 1.0 | presence_centered 中 Tversky 权重 | ⚠️ 实验性参数 |
| `fraction_mae_weight` | float | 0.1 | presence_centered 中 fraction MAE 权重 | ⚠️ 实验性参数 |
| `height_loss_kind` | str | "l1" | 高度回归损失 (l1/huber/mse) | ⚠️ 实验性参数 |
| `huber_delta` | float | 1.0 | Huber 损失过渡点 | ⚠️ 实验性参数 |
| `build_height_boost` | float | 5.0 | 建筑物高度加权 | ⚠️ 实验性参数 |
| `veg_height_boost` | float | 0.0 | 植被高度加权 | ⚠️ 实验性参数 |
| `aux_veg_weight` | float | 1.0 | 植被辅助权重 | ⚠️ 实验性参数 |
| `iou_loss_kind` | str | "tversky" | IoU 损失类型 (tversky/focal) | ⚠️ 实验性参数 |
| `focal_gamma` | float | 2.0 | Focal loss gamma | ⚠️ 实验性参数 |
| `focal_alpha` | float | 0.25 | Focal loss alpha | ⚠️ 实验性参数 |
| `height_bin_aux_weight` | float | 0.0 | softbin 辅助 CE 权重 | ⚠️ 实验性参数 |
| `height_bin_sigma_bins` | float | 1.5 | softbin 高斯目标宽度 | ⚠️ 实验性参数 |
| `building_smooth_weight` | float | 0.0 | 建筑物光滑度正则化权重 | ⚠️ 实验性参数 |
| `building_smooth_erode_px` | int | 1 | 建筑物腐蚀半径 | ⚠️ 实验性参数 |
| `building_smooth_thr` | float | 0.0 | 建筑物分类阈值 | ⚠️ 实验性参数 |

### 5. 优化和训练策略
| 参数 | 类型 | 默认值 | 使用位置 | 建议 |
|------|------|--------|---------|------|
| `amp` | bool | True | 混合精度训练 | ✅ 保留 |
| `grad_accum_steps` | int | 1 | 梯度累积步数 | ✅ 保留 |
| `data_parallel` | bool | False | 多 GPU 数据并行 | ⚠️ 实验性参数 |
| `num_workers` | int | 4 | DataLoader 工作进程数 | ✅ 保留 |
| `prefetch_factor` | int | 1 | DataLoader 预取因子 | ⚠️ 微调参数 |

### 6. LDS 采样器参数 (长尾分布处理)
| 参数 | 类型 | 默认值 | 使用位置 | 建议 |
|------|------|--------|---------|------|
| `lds_sampler` | bool | False | 启用 LDS 采样 | ⚠️ 实验性参数 |
| `lds_bins` | int | 16 | LDS 直方图 bin 数 | ⚠️ 实验性参数 |
| `lds_sigma` | float | 2.0 | LDS 高斯平滑 σ | ⚠️ 实验性参数 |
| `lds_cap` | float | 5.0 | LDS 权重上限系数 | ⚠️ 实验性参数 |
| `lds_h_max` | float | 60.0 | LDS 直方图上边界 | ⚠️ 实验性参数 |
| `lds_score` | str | "p95_veg" | LDS 评分方式 | ⚠️ 实验性参数 |

### 7. 多任务学习
| 参数 | 类型 | 默认值 | 使用位置 | 建议 |
|------|------|--------|---------|------|
| `task` | str | "both" | 单任务分割 (both/presence/height) | ⚠️ 实验性参数 |
| `structure_weight` | float | None | 高度/结构损失权重覆盖 | ⚠️ 实验性参数 |

### 8. 预训练和检查点
| 参数 | 类型 | 默认值 | 使用位置 | 建议 |
|------|------|--------|---------|------|
| `init_from_pretrain` | str | None | 可选预训练检查点路径 | ⚠️ 条件必需 |
| `init_pretrain_strict` | bool | False | 严格加载预训练权重 | ⚠️ 微调参数 |
| `split_file` | str | None | 可选数据集分割缓存文件 | ⚠️ 可选优化 |

---

## 二、模型参数 (core/model.py)

### build_model 函数签名
```python
def build_model(model_type, n_channels, n_classes, 
                tessera_presence_ch=16,
                tessera_hidden_ch=None, 
                tessera_hidden_depth=0,
                height_specialist_depth=0, 
                lightunet_base_ch=32,
                height_gate_source="alpha", 
                height_hidden_ch=None,
                height_trunk_depth=2, 
                height_independent_branches=False,
                height_head_kind="linear", 
                height_n_bins=64,
                height_bin_max_m=80.0, 
                lightunet_norm_kind="bn",
                gate_mode="simple", 
                gate_untied=False, 
                gate_init_bias=4.0,
                modality_dropout=0.0)
```

### 模型架构多样性
当前支持 **70+ 种模型架构变体**：
- `lightunet*` (3 种)
- `tessera_iou_fusion*` (15+ 种)
- `tessera_token_*` (20+ 种)
- 其他特殊架构 (DPT, embedding refiner 等)

---

## 三、参数复杂性分析

### 参数分类统计
| 分类 | 数量 | 说明 |
|------|------|------|
| 必需核心参数 | 8 | 训练必须，无默认值有意义 |
| 广泛使用参数 | 15 | 影响多个模型和损失类型 |
| 实验性参数 | 40+ | 特定模型或损失型专用 |
| 微调参数 | 5 | 性能优化相关 |
| **总计** | **~70** | 仅 train.py argparse |

### 参数使用模式
```
通用参数 → build_model() ✅ (重复度: 1)
              → ImprovedCompositeLoss() ✅ (重复度: 1)
              → DataLoader ✅ (重复度: 1)

架构专用参数 → 仅 tessera_iou_fusion* 使用
              → 多达 15 个相关参数
              → 非 tessera 模型时未使用

损失函数专用参数 → 仅特定 loss_preset 使用
                 → presence_centered: 5 个参数
                 → LDS sampler: 6 个参数
```

---

## 四、可删除/简化的参数

### 🗑️ 强烈建议删除

#### 1. **微调参数** (性能提升不明显，增加复杂度)
- `prefetch_factor` - DataLoader 优化，通常不需要调节
- `init_pretrain_strict` - 默认 False 就足够
- `building_smooth_erode_px` / `building_smooth_thr` - 极其细粒度

#### 2. **实验性参数** (仅在特定实验中使用)
- `tessera_hidden_depth` - 很少改变
- `gate_untied` - 表现不如 tied gate
- `gate_init_bias` - 固定为 4.0 效果已很好
- `veg_height_boost` - 通常保持 0.0
- `height_independent_branches` - 增加复杂度，收益有限
- `iou_loss_kind` - 默认 tversky 足够好

#### 3. **多任务学习参数** (不是生产使用)
- `task` - 仅用于分割实验
- `structure_weight` - 可用 loss_preset 代替

#### 4. **LDS 采样器相关** (特定场景，通常不用)
- `lds_sampler` - 开启率很低
- `lds_bins`, `lds_sigma`, `lds_cap`, `lds_h_max`, `lds_score` - 都依赖 lds_sampler

**删除这些参数后，保留**: ~20-25 个关键参数

### ⚠️ 有条件保留

| 参数 | 保留条件 |
|------|---------|
| `secondary_train_embeddings_dir` | 仅 tessera_iou_fusion 模型需要 |
| `token_train_embeddings_dir` | 仅 tessera_token_* 模型需要 |
| `height_bin_aux_weight` | 仅 height_head_kind=softbin 时需要 |
| `focal_*` | 仅 iou_loss_kind=focal 时需要 |
| `data_parallel` | 可用 torch.nn.DataParallel 封装替代 |

---

## 五、建议的参数重组方案

### 方案 A: 最小化版本 (20 个参数)
保留仅用于生产环境的参数：
```
核心: model_type, output_dir, experiment_name, batch_size, epochs, lr, weight_decay, seed
数据: train_embeddings_dir, train_targets_dir
模型: lightunet_base_ch, tessera_presence_ch, height_gate_source, 
      height_head_kind, gate_mode
损失: loss_preset, aux_weight
优化: amp, num_workers, grad_accum_steps
```

### 方案 B: 平衡版本 (35 个参数)
保留方案 A + 常用架构参数：
```
+ tessera_hidden_ch, height_hidden_ch, height_trunk_depth
+ presence_tversky_weight, fraction_mae_weight
+ height_loss_kind, build_height_boost, height_n_bins, height_bin_max_m
+ lightunet_norm_kind
+ secondary_train_embeddings_dir, init_from_pretrain
```

### 方案 C: 当前版本 (保持兼容)
保留所有 ~70 个参数，但在代码中标记：
- 🔴 已弃用/不推荐使用
- 🟡 实验性，仅在 dev 分支启用
- 🟢 生产级别参数

---

## 六、具体删除建议

### 第一阶段：低风险删除 (删除 15 个参数)
```python
# 在 parse_args() 中删除或注释掉：
# - prefetch_factor (保留默认 1)
# - init_pretrain_strict (保留默认 False)
# - lds_* 全部 6 个参数 (条件保留)
# - gate_untied (改用 gate_mode="simple/rich")
# - veg_height_boost (不用，保持 0.0)
# - height_independent_branches (删除，保持 False)
# - building_smooth_* 3 个参数 (改用单个开关)
# - structure_weight (仅在需要时作为 loss_preset 选项)
```

### 第二阶段：重组参数 (整合 10 个参数)
```python
# 创建"高级配置"模式
# 例: --enable-advanced-loss 激活所有 loss_* 参数
# 例: --enable-lds-sampler 激活 lds_* 参数集
# 例: --tessera-config <preset> 快速设置所有 tessera_* 参数
```

### 第三阶段：配置文件 (替代 20 个参数)
```yaml
# config.yaml
model:
  type: tessera_iou_fusion
  base_channels: 32
  tessera_channels: 16
  height_head: linear
  
training:
  batch_size: 32
  epochs: 30
  lr: 2e-4
  loss_preset: presence_centered
  
advanced:  # 折叠到配置文件
  lds_sampler: false
  height_loss_kind: l1
```

---

## 七、代码修改检查清单

如果决定删除参数，需要检查：

- [ ] `parse_args()` - 移除 argparse 定义
- [ ] `DEFAULTS` 字典 - 移除默认值
- [ ] `main()` 函数 - 移除参数传递
- [ ] 参数日志记录 - 移除 JSON 日志条目
- [ ] `build_model()` 调用 - 移除参数传递
- [ ] `ImprovedCompositeLoss()` 初始化 - 移除参数传递
- [ ] DataLoader 创建 - 移除相关参数
- [ ] 文档和说明 - 更新 README
- [ ] 测试脚本 - 确保不使用已删除参数

---

## 八、总结与建议

### 当前状态
- **参数总数**: ~70 个
- **必需参数**: 8 个
- **可删除参数**: 20-25 个
- **配置方式**: 仅 CLI argparse (不灵活)

### 推荐行动
1. ✅ **立即删除**：`prefetch_factor`, `gate_untied`, `veg_height_boost` 等 5-10 个参数
2. ⚠️ **有条件保留**：LDS、多任务、实验性参数放入"高级模式"
3. 🎯 **长期目标**：迁移到 YAML 配置 + CLI override 模式（见方案 C）

### 预期效果
- **降低复杂度**: 70 → 30-35 个参数 (-50%)
- **提高可维护性**: 参数意图清晰，不需要猜测默认值
- **保持灵活性**: 关键参数仍可调节

