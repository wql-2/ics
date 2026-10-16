# Qwen3-VL → 轻量开放词汇检测 demo

目标：在 RTX 4090 上部署尽量小的**非 YOLO**目标检测模型，输出边界框和类别，推理时可更换类别名称，不局限于 COCO 的 80 类。

本项目提供能运行的开放词汇基线，以及一个约 **3.80M 参数视觉学生**的研究原型。**尚未用真实 Qwen 标注数据训练，也没有证据表明精度已接近 Qwen3-VL。** 当前实测环境只有 CPU；4090 指令已提供，GPU 性能需要在你的设备上测量。

技术路线、取舍和推荐实验见 [docs/技术路线.md](docs/技术路线.md)，实际完成的验证见 [docs/验证结果.md](docs/验证结果.md)。

## 1. 两条可运行路径

| 路径 | 作用 | 开放词汇的实现 | 当前限制 |
|---|---|---|---|
| `baseline` | 无需额外训练的初始对照 | D-FINE 只提供候选框，TinyCLIP 对区域和任意文本做相似度匹配；密集窗口补充新类别候选 | 候选框有 COCO 偏差，窗口慢且易产生误报；不能作为高精度方案 |
| `student` | 目标部署方案的研究原型 | 将 D-FINE 的固定线性分类头替换为共享的 512 维区域语义投影头，以输入文本原型计算类别分数 | 新语义头必须训练；两步 smoke test 不代表可用模型 |

学生的训练信号包括 Qwen3-VL 生成的框/类别、DETR 框和分类损失、TinyCLIP 区域特征余弦蒸馏，以及区域对文本词表的相似关系蒸馏。Qwen 是离线的语义和定位教师；TinyCLIP 是辅助的语义空间教师。没有直接对齐 Qwen 的自回归 token logits。

`--texts` / `--classes` 是**本次查询的候选词表**。训练后可换成新词表，分类头参数量不随词表大小改变。开放词汇检测仍需文本候选；它不等于一个能无限生成所有物体名称的模型。也可以先让 Qwen 自由发现名称、整理成大型词库，再把词库作为默认输入。

D-FINE 是卷积视觉骨干 + Transformer 编解码器的 DETR 系列混合架构，不是纯 ViT，也不使用 YOLO 检测头。`vendor/dfine` 保留了上游完整源码和许可证；其中的其他架构不会被本 demo 实例化。

## 2. 安装

推荐 Python 3.11/3.12。先单独安装匹配驱动的 CUDA PyTorch，再安装其他依赖。以下是经过本项目 CPU 版本验证的同代 CUDA 组合示例：

```bash
cd tiny-recognizer
python -m venv work/.venv
source work/.venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python fetch_assets.py
```

CPU 可将 PyTorch 索引改为 `https://download.pytorch.org/whl/cpu`。仓库不包含模型权重和中间训练产物。运行 `fetch_assets.py` 可下载约 15MB 的 D-FINE 检测权重、约 45MB 的 TinyCLIP 权重和示例图片，并验证 SHA256。当前云环境此前下载的文件仍位于 `/workspace/tiny-recognizer/assets/`。教师服务使用独立环境，避免 vLLM 的 PyTorch 依赖影响学生训练。

## 3. 立即运行开放词汇基线

```bash
python demo.py detect \
  --image assets/dog.jpg \
  --texts "samoyed dog" "cat" "red backpack" "bicycle helmet" "watering can" \
  --backend baseline --device cuda:0 --out work/baseline
```

输出包括画框图、`predictions.jsonl` 和参数/耗时报告。`samoyed dog`、`bicycle helmet`、`watering can` 不在原 COCO 分类头的类别清单中；这只说明候选名称不受 COCO 清单限制，**不证明这些类别有可靠检测精度**。请观察误报和重复框。

该小型 TinyCLIP 的细粒度识别、背景区分能力有限。基线分数是“余弦相似度 × 候选区域先验的平方根”，不是概率。候选先验来自闭集检测器的最高分；密集窗口使用固定 0.3。`--no-dense` 可提速，但会进一步降低新目标的候选召回。按验证集选择阈值，不能在测试集反复调参。

## 4. 用 Qwen3-VL 离线标注

Qwen3 的纯文本版本不能标注图像，这里需要 **Qwen3-VL**。支持任意 OpenAI Chat Completions 兼容服务，包括本机 vLLM；API 密钥可放入 `QWEN_API_KEY` 环境变量。

在独立的教师环境中安装支持 Qwen3-VL 的 vLLM，参考 [Qwen3-VL 官方部署说明](https://github.com/QwenLM/Qwen3-VL)。单张 4090 可先用 4B 教师，显存余量足够再尝试 8B；下列为启动示例，显存占用并未在本环境验证：

```bash
vllm serve Qwen/Qwen3-VL-4B-Instruct \
  --max-model-len 8192 --max-num-seqs 1 --gpu-memory-utilization 0.90
```

在学生环境运行标注程序。训练集和人工评测集应按场景/拍摄来源分开：

```bash
# 自由发现类别，不传入预设词表
python demo.py label --images data/train \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --base-url http://127.0.0.1:8000/v1 --out work/train_qwen.jsonl

# 也可要求检测某个大型词表；每次分块查询 16 类，避免提示过长
python demo.py label --images data/train --classes data/train_classes.json \
  --model Qwen/Qwen3-VL-4B-Instruct --out work/train_qwen.jsonl
```

教师被要求输出相对坐标 `[0,1000]`，程序严格转换成原图像素 `xyxy`。完整原始回答、模型名、图片 SHA256 和请求配置均缓存以供审查。API 失败、截断、未知类别和非法框会记录到错误文件并使命令失败，**不会被当作“没有目标”**。相同请求可复用缓存；图片内容或查询配置改变则重新请求。

`train_qwen.discovered_classes.json` 可以作为训练词表的初稿。自由命名常有同义词和粒度冲突，训练前需要统一，例如 `bike` / `bicycle`。合并词表时必须同步更新标注中的 `label`。Qwen 对小目标、遮挡、密集实例可能漏框；应抽查和纠正，尤其是把漏框当背景的风险。

## 5. 训练轻量开放词汇学生

**先停止教师服务释放显存，再运行学生训练。**

```bash
python demo.py train \
  --images data/train --labels work/train_qwen.jsonl \
  --classes work/train_qwen.discovered_classes.json \
  --variant nano --device cuda:0 --amp --size 640 --batch 4 \
  --epochs 30 --out work/ov-nano
```

这是起始配置，不是经过寻优的训练配方。代码保留了原始 D-FINE 的定位和匹配损失，但使用简化的数据流程；没有完整的大规模增强、EMA、分布式训练和断点恢复。大数据生产训练需补齐这些工程能力，并做人工标注混合监督。

`--variant micro` 将语义维度前的隐藏宽度由 128 改为 96，解码层数由 3 改为 2，视觉学生约 **2.82M**。它只能继承形状匹配的部分预训练权重，精度可能明显下降；应以 Nano 为基准，用真实验证集判断是否值得压缩。报告会记录实际继承的张量/元素数。`--freeze-backbone` 可用于短期语义对齐预热，之后重新训练需自行加载相应权重扩展训练流程。

训练时完整类别词表只用于语义学习，推理时可更换。类别样本不足、训练类别过窄或 TinyCLIP 没学过的概念，都可能导致新类别识别失败。

## 6. 在部署时更换新类别

文本编码器只在新词到来时使用，可运行在 CPU。预计算和图像推理分开，便于明确参数/显存口径：

```bash
python demo.py encode --texts "screwdriver" "bicycle helmet" "coffee grinder" \
  --device cpu --out work/new_vocab.pt

python demo.py detect --backend student --checkpoint work/ov-nano/last.pt \
  --images data/test --vocab work/new_vocab.pt \
  --device cuda:0 --threshold 0.05 --out work/novel_results
```

词表文件也可以是任意数量名称的 JSON 数组，通过 `encode --classes names.json` 生成原型。默认英文提示模板适合 TinyCLIP 的训练分布；中文名建议先映射到英文，保留中文显示名。多语言文本编码器可替换，但改变语义空间后必须重新对齐学生。

参数统计包含全部实际权重：

| 组件 | 参数 |
|---|---:|
| Nano 视觉学生，语义投影头共享，无类别索引嵌入 | 3,796,583 |
| Micro 视觉学生 | 2,821,490 |
| TinyCLIP 视觉编码器，学生训练辅助/基线使用 | 8,276,992 |
| TinyCLIP 文本编码器，**含词嵌入** | 15,169,024 |
| Nano 学生 + 完整文本编码器 | 18,965,607 |
| Micro 学生 + 完整文本编码器 | 17,990,514 |

“Text-3M”是上游型号名称，不能直接作为完整文本塔的参数量。缓存词表后图像推理只加载约 3.80M / 2.82M 的视觉学生；系统接收任意新词仍需要文本编码器。这里没有把它的成本藏起来。文本原型存储额外为 `类别数 × 512 × dtype字节数`，不属于可训练参数。

## 7. 评测：验证接近教师，而非只验证模仿教师

使用独立人工框标注的 `human_val.jsonl`，和学生输出相同的图片相对路径。不能用教师生成的标签作为真实精度的唯一真值。建议同时保存教师、学生、训练前基线在同一批图片上的输出，分别评测。

```bash
# 评测 AP 时推理保留低分候选，避免提前截断召回
python demo.py detect --backend student --checkpoint work/ov-nano/last.pt \
  --images data/val --classes data/eval_all_classes.json \
  --threshold 0.001 --max-detections 100 --device cuda:0 --out work/student_val

python demo.py evaluate --ground-truth data/human_val.jsonl \
  --predictions work/student_val/predictions.jsonl \
  --base-classes work/train_qwen.discovered_classes.json \
  --student-checkpoint work/ov-nano/last.pt \
  --threshold 0.25 --out work/student_metrics.json
```

报告包含 IoU=0.5 的一对一匹配 Precision/Recall/F1、各类别召回，以及 COCO AP50:95/AP50。AP 使用保存的全部预测分数，F1 使用 `evaluate --threshold`。base/novel 按是否出现在学生检测训练词表中划分；**novel 不代表没在 CLIP 的图文预训练中出现过**。对 Qwen 输出不捏造置信度，主要比较 F1 和召回；无可靠排序分数时不报告教师 AP。

加 `--student-checkpoint` 会按图片内容哈希检查已知的训练/评测重叠；真值需包含 `image_sha256`。评测记录必须包含所有图片，包括输出空列表的图片，以免遗漏困难样本。近重复帧、同场景泄漏和语义同义词泄漏仍需在数据划分阶段处理。

## 8. 没有数据/教师服务时的流程自检

以下合成数据只有几何形状，不是 Qwen 标注，也不用于声称检测精度：

```bash
python demo.py fixtures --out work/fixtures
python demo.py train --images work/fixtures/train --labels work/fixtures/train.jsonl \
  --classes work/fixtures/classes.json --size 128 --batch 2 --epochs 1 --max-steps 2 \
  --out work/smoke-nano
python -m pytest -q tests
```

自检生成的 `last.pt` 只有两步更新，不是交付的可用检测模型。训练后以不含训练类别的新词表做前向验证，只能证明动态词表接口和张量维度正确。

统一 JSONL 格式如下，坐标均为 EXIF 校正后的原图像素：

```json
{"file_name":"a.jpg","width":640,"height":480,"boxes":[{"label":"screwdriver","xyxy":[10,20,120,200],"score":null}],"provenance":"human"}
```

教师/人工标签的 `score` 为 `null`；学生分数为数值。人类标注可附上 `image_sha256`。训练默认跳过空标注，只有人工确认是真负样本时才用 `--include-empty`。

## 9. 第三方来源

- [D-FINE](https://github.com/Peterande/D-FINE)：官方源码按 commit 固定在 `vendor/dfine/UPSTREAM.md`，Apache-2.0 许可证保留。
- [TinyCLIP](https://github.com/wkcn/TinyCLIP)：官方模型配置和公开预训练权重，来源和 SHA256 见下载脚本；其许可见 `assets/TinyCLIP_LICENSE`。
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)：本项目通过可替换的兼容 API 调用，不包含教师权重。
- 示例 dog 图片来自 [PyTorch Hub](https://github.com/pytorch/hub/tree/master/images)。
