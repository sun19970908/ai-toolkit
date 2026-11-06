# Krea 2 (12B DiT) 16GB 显卡低显存 LoRA 训练实战避坑全指南

本指南总结了在 **RTX 4060 Ti 16GB** 显卡 + **96GB 系统内存** 环境下，使用 [sun19970908/ai-toolkit](https://github.com/sun19970908/ai-toolkit) 成功训练 Krea 2 LoRA 的完整经验与最佳实践配置。

---

## 一、 硬件与环境准备

* **GPU 显存**：16 GB VRAM（如 RTX 4060 Ti / RTX 4070 / RTX 4080 / RTX 5060 Ti 等）
* **系统内存（RAM）**：**强烈建议 32 GB ~ 64 GB 以上**（本机实测 96 GB，是实现 100% 层级置换不爆显存的定海神针）
* **操作系统**：Windows 11 / 10
* **Python 环境**：Python 3.11 + PyTorch 2.6 (`torch==2.6.0+cu124`)
* **代码库**：已包含针对 PyTorch 2.6 语法、单文件 Safetensors 文本端及 patch_size 适配补丁的个人 Fork 仓库：
  ```bash
  git clone https://github.com/sun19970908/ai-toolkit.git
  ```

---

## 二、 踩坑总结与七大核心防爆秘籍

Krea 2 包含 12B 单流 DiT 主干与 4B Qwen3-VL 文本编码器，默认全精度训练需 48GB+ 显存。要在 16GB 显卡上稳如磐石运行，必须遵循以下规则：

### 1. 锁定原生 INT8 量化，杜绝解包还原
* **陷阱**：ai-toolkit 如果未显式声明量化类型，会把官方下载的 12.5GB `convrot8` INT8 权重自动解包成 25GB 全精度浮点，瞬间导致 OOM。
* **解决**：在 `model:` 区块务必设置：
  ```yaml
  quantize: true
  qtype: "convrot8"
  quantize_te: true
  qtype_te: "convrot8"
  ```

### 2. 层卸载（Layer Offloading）比例必须设为 1.0（100%）
* **陷阱**：部分社区早期经验建议 `layer_offloading_transformer_percent: 0.5`（卸载 50%）。这会导致另外 50%（约 6.3GB）模型权重被**永久死锁**在显存中。
* **解决**：利用机器充沛的系统内存，将所有层置换给内存管理，GPU 显存中**零静态权重保留**，只在当前层计算时按需读入（单层仅 ~400MB）：
  ```yaml
  layer_offloading: true
  layer_offloading_transformer_percent: 1.0   # 100% 完全层卸载
  ```

### 3. 分辨率锁定 768（16GB 显卡保命核心）
* **陷阱**：如果分桶列表中包含 `1024`（如 `resolution: [768, 1024]`），在遇到 1024x1024 或长图（832x1248）时，注意力机制计算激活值呈平方倍数剧增（$4096^2$ 比 $2304^2$ 激增 3.16 倍），显存峰值直接逼近 15.5GB 物理极限，一旦累积微小显存碎片就会触发连续 OOM 终止训练。
* **解决**：16GB 显卡**务必锁定 768 分辨率**：
  ```yaml
  datasets:
    - resolution:
        - 768
  ```
  * **优势**：显存占用仅在 **7 ~ 8 GB**，留有足足 **8 GB 巨大安全裕量**，绝不 OOM！而且训练速度提升 3 倍（单步仅需 5~8 秒）。
  * **效果**：训练出的 LoRA 在 ComfyUI 中完全可以直接以 1024x1024 甚至更高分辨率进行推理，细节和解剖学结构完全不打折扣。

### 4. 全局预缓存并彻底卸载文本编码器
* **陷阱**：若在训练循环中实时计算文本和图像特征，Qwen3-VL (4B) 会与 DiT 主干同时争抢显存。
* **解决**：在 `datasets:` 与 `train:` 中开启离线全盘缓存：
  ```yaml
  datasets:
    - cache_latents_to_disk: true
      cache_text_embeddings: true
  train:
    unload_text_encoder: true
  ```
  第一次启动时耗时 1~2 分钟将特征存入 `_latent_cache` 和 `_t_e_cache`，随后彻底卸载释放文本端与 VAE，后续所有训练直接秒级读取硬盘缓存。

### 5. 严禁训练中途采样预览
* **陷阱**：训练每隔 250 步或第 0 步生成 sample 预览图时，推理管线会突发申请 3~4GB 显存，造成尖刺冲击，直接引发不可逆 OOM。
* **解决**：16GB 显卡训练模式下必须彻底关闭中途生图：
  ```yaml
  train:
    disable_sampling: true
  ```

### 6. 流匹配调度器显式声明（避免 Step 0 崩溃）
* **陷阱**：Krea 2 采用 Flow Matching 架构。ai-toolkit 默认将 `noise_scheduler` 继承为传统 `"ddpm"`，导致未执行动态位移计算，引发：
  `ValueError: mu must be passed when use_dynamic_shifting is set to True`。
* **解决**：在 `train:` 区块必须显式写上：
  ```yaml
  train:
    noise_scheduler: "flowmatch"
  ```

### 7. 关于 Triton 警告处理
* **说明**：Windows 控制台出现 `WARNING:torchao.kernel.intmm: Detected no triton...` 是正常现象。Triton 主要是 Linux 内核，ai-toolkit 会自动回退到 PyTorch 原生 CUDA C++ 算子（`torch._int_mm`），对训练精度、质量与收敛**完全没有任何负面影响**，无需理会。

---

## 三、 开箱即用 YAML 配置模板

配置文件位置：`config/train_krea2_anatomy.yaml`

```yaml
job: "extension"
config:
  name: "krea2_character_lora"
  process:
    - type: "diffusion_trainer"
      training_folder: "G:/AI/lora/ai-toolkit/output"
      log_dir: "G:/AI/lora/ai-toolkit/output"          # 开启 TensorBoard 曲线图监控 (localhost:6006)
      device: "cuda:0"
      
      # LoRA 网络配置
      network:
        type: "lora"
        linear: 32                             # Rank 32 最适合角色与解剖学精细度
        linear_alpha: 16

      # 权重保存策略
      save:
        dtype: bf16
        save_every: 250
        max_step_saves_to_keep: 6

      # 数据集与预缓存
      datasets:
        - folder_path: "G:/AI/lora/krea2_lora_dataset"   # 你的数据集图片及txt标注目录
          caption_ext: "txt"
          cache_latents_to_disk: true           # 开启硬盘缓存防爆显存（必需）
          cache_text_embeddings: true          # 预缓存文本特征（必需）
          resolution:
            - 768                                # 16GB 显卡黄金分辨率：显存仅占用 ~8GB，绝不爆显存

      # 训练超参数
      train:
        batch_size: 1
        steps: 1300                             # 30~50张图建议 1000~1500 步
        gradient_accumulation_steps: 1          # 显存充裕可保持 1；也可设 2 增加平滑度
        train_unet: true                        # 专训 DiT 主干
        train_text_encoder: false               # 冻结文本编码器
        gradient_checkpointing: true          # 梯度检查点（16G 显存核心）
        noise_scheduler: "flowmatch"            # 流匹配调度器（必需）
        optimizer: "adamw8bit"                  # 8bit 优化器，压减状态显存
        lr: 0.0001                              # 学习率 1e-4
        unload_text_encoder: true               # 提取完文本特征后彻底从显存释放
        disable_sampling: true                  # 16G 关键参数：关闭中途生图消除显存毛刺

      # 模型与分流卸载配置
      model:
        # 指向本地的 Krea 2 Raw 底模
        name_or_path: "G:/ComfyUI-Easy-Install/ComfyUI-Easy-Install/ComfyUI/models/unet/krea2_raw_int8_convrot.safetensors"
        arch: "krea2"
        quantize: true                          # 保留内置 INT8 权重
        qtype: "convrot8"                       # 严格匹配底模 convrot8，禁止解包
        quantize_te: true                       # 文本端同样匹配 convrot8
        qtype_te: "convrot8"
        low_vram: true
        layer_offloading: true                  # 开启层级动态置换
        layer_offloading_transformer_percent: 1.0 # 100% 层完全按需调用系统内存，释放显存
        model_kwargs:
          # 指向本地的 Qwen3-VL 文本编码器权重
          text_encoder_path: "G:/ComfyUI-Easy-Install/ComfyUI-Easy-Install/ComfyUI/models/text_encoders/qwen3VLInstruct4bHeretic_int8Convrot.safetensors"

      sample:
        sampler: "flowmatch"
        sample_every: 250
        width: 768
        height: 768
        neg: ""
        samples:
          - prompt: "masterpiece, best quality, 1girl, solo"
            neg: ""
            seed: 42
            guidance_scale: 4.0
            sample_steps: 25
```

---

## 四、 训练启动命令

在 PowerShell 中进入 `ai-toolkit` 目录，设置 PyTorch 显存碎片整理环境变量并启动：

```powershell
cd G:\AI\lora\ai-toolkit
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
python run.py config/train_krea2_anatomy.yaml
```

---

## 五、 ComfyUI 推理使用指南

训练完成生成的 LoRA 权重保存在：
`G:\AI\lora\ai-toolkit\output\krea2_anatomy_fix\`

在 ComfyUI 中使用该 LoRA 的黄金法则：
1. **训练底模 vs 推理底模**：
   * 训练阶段：使用 **Krea 2 RAW**（未经蒸馏的高动态范围底模，特征学习最扎实）。
   * 推理阶段：使用 **Krea 2 Turbo**（蒸馏高速版底模，生成快、手感好）。
2. **LoRA 推荐强度**：
   * `0.75 ~ 0.85`（既能充分体现 LoRA 学习到的结构/特征，又不会导致过拟合泛化差）。
3. **推理步数与 CFG**：
   * 采样步数：`6 ~ 8 步`
   * CFG (Guidance Scale)：`1.0`
