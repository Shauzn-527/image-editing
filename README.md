# 图像编辑实践：Pix2Pix-Zero

## 项目简介

本项目基于 Stable Diffusion 与 Pix2Pix-Zero，实现“**不训练模型**”的图像编辑流程：

- 输入一张真实图像与原始 prompt（例如 `a photo of a cat`）
- 通过反演（inversion）得到可编辑 latent
- 构造语义方向（如 `cat -> dog`）并进行编辑去噪
- 输出重建图与编辑图进行对比

项目目标不是追求单一分数，而是让同学理解并掌握：
1. 语义方向如何影响编辑结果
2. 交叉注意力约束如何帮助保持结构
3. 参数如何影响“编辑强度 vs 结构保真”

---

## 1. 主要知识点

1. **DDIM 反演（Inversion）**
   - 将真实图像映射到扩散轨迹中的 latent，便于后续可控编辑。

2. **编辑方向（Edit Direction）**
   - 使用两组句子（source / target）编码后做均值差：
   $$
   \Delta c = \mathbb E[c_{target}] - \mathbb E[c_{source}]
   $$

3. **参考注意力约束（Reference Cross-Attention）**
   - 在编辑过程中对齐注意力图，尽量保持主体结构与布局。


---

## 2. 快速运行

在 `image-edit/` 下运行：

```bash
python pix2pix_zero_demo.py \
  --image pix2pix-zero/test_images/cats/cat_6.png \
  --prompt "a photo of a cat" \
  --model stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --steps 50 \
  --guidance-scale 7.5 \
  --xa-guidance 0.1 \
  --outdir outputs/pix2pix_zero_demo \
  --fp16
```

默认输出：
- `reference_reconstruction.png`
- `edited.png`
- `grid_input_reference_edited.png`

---

## 3. 代码结构

- `pix2pix_zero_demo.py`
  - 命令行入口，读取输入图像并保存结果
- `pix2pix_zero_runner.py`
  - 核心流程（反演 / 参考去噪 / 编辑去噪）
- `pix2pix_zero_prompts.py`
  - 构造 source / target 语义句子集合
- `pix2pix_zero_utils.py`
  - 图像读写与拼图工具
- `pipeline_stable_diffusion_pix2pix_zero.py`
  - Pix2Pix-Zero 所需 pipeline（含 cross-attention 相关逻辑）

---

## 4. 作业说明

本作业采用“**带 TODO 注释、保留可运行参考实现**”的形式。你需要先读懂公式与数据流，再自行实现并替换参考逻辑。

### 4.1 必做部分

| TODO | 文件 | 函数 / 位置 | 任务 |
|------|------|-------------|------|
| TODO 1 | `pix2pix_zero_runner.py` | `compute_edit_direction` | 实现编辑方向计算（source/target 均值差） |
| TODO 2 | `pix2pix_zero_prompts.py` | `_default_animal_sentences`, `create_sentences` | 构造并组织 source/target 句子集合 |

### 4.2 开放实验


1. 自行寻找更多编辑例子（不限于 `cat -> dog`）
   - 例如：物体类别替换、风格变化、场景变化、属性变化等。
2. 系统调参并记录现象
   - 建议重点调整：`steps`、`guidance-scale`、`xa-guidance`。
3. 对比并总结
   - 哪些参数组合更容易“语义变化明显但结构不崩”？
   - 哪些任务更难编辑？原因是什么？

### 4.3 建议提交内容

1. 完成 TODO 后的代码
2. 至少 3 组不同编辑任务的可视化结果
3. 简短实验总结（成功案例 + 失败案例 + 经验）

### 4.4 评估指标（需要报告 CLIPScore）

为便于对“编辑是否朝目标语义靠近”进行定量参考，请在实验记录中**额外报告 CLIPScore**（作为参考指标，不作为唯一评价标准）：

- 对每个编辑任务，计算 `edited.png` 与 **target prompt** 的 CLIPScore。
- 工具/实现：请参考并使用仓库 https://github.com/Taited/clip-score 的实现方式进行计算与汇报

---

## 5. 说明

- 你可以保留默认句子模板，也可以自行设计更有针对性的 source/target 句子集合。
