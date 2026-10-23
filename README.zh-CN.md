# 在 24GB 显卡上运行 Drama 3

[English](README.md) · [样音试听](https://torenyx.github.io/drama3-24gb/) · [完整实测](benchmarks/README.md) · [安装与运行](docs/setup.md)

用**一张或两张 24GB 显卡**，根据文本和参考音色生成语音。三个配置分别运行原发布 FP8/BF16 权重、Q4_K GGUF 与 INT4。

**已实测 RTX 4090 / Linux。** A 保持主模型与独立 Codec 常驻；C 持久缓存参考编码，并将流式生成与波形解码装入单张卡。

## 选一条路线

| 你的侧重点 | 运行配置 | GPU | 框架 |
|---|---|---|---|
| 保留原发布专家精度 | `a-fp8`（A） | 双卡 | ComfyUI-Drama3，FP8 专家＋BF16 其余部分 |
| 单卡原生 worker | `b-q4`（B） | 单卡；双卡跑独立任务 | speech.cpp 0.8.11，Q4_K GGUF |
| 快速分段生成 | `c-int4`（C） | 双卡；实验性单卡 | SGLang-Omni，本地转换 INT4 |

## 听结果、看耗时

| 原发布 FP8/BF16 · A | Q4_K GGUF · B | INT4 分段 · C |
|---|---|---|
| [试听 / 下载 WAV](media/a-fp8-resident-seed1.wav) | [试听 / 下载 WAV](media/b-q4k-seed1.wav) | [试听 / 下载 WAV](media/c-int4-chunked-trial1.wav) |
| **28.49 秒音频 / 13.06 秒请求** | **37.48 秒音频 / 13.98 秒请求** | **26.63 秒音频 / 4.90 秒请求合计** |
| 双卡 · seed 1 · 保留分块历史 | 单卡 · seed 1 · 原生 worker | 双卡 · trial 1 · 三段独立生成 |

RTX 4090 24GB；同一日语剧本、两段共 40.9 秒参考录音。耗时包含生成与输出波形，参考与服务已准备，不含启动。C 合计三次请求后拼接 PCM。三者的采样与参考处理不同。[网页播放器、台词与对应记录 →](https://torenyx.github.io/drama3-24gb/)

## 跑自带样例

以下使用新建的独立环境、仓库自带的 LibriSpeech 参考录音和已转换权重：

| 配置 | 输出 | 请求耗时 | 生成命令总耗时 |
|---|---|---:|---:|
| A · 双卡 | [27.81 秒 WAV](media/open-a-fp8.wav) | 19.04 秒 | 38.49 秒 |
| B · 单卡 | [26.70 秒 WAV](media/open-b-q4.wav) | 10.11 秒 | 21.97 秒 |
| C · 单卡 | [23.15 秒 WAV](media/open-c-int4.wav) | 4.39 秒 | 64.16 秒 |

A 复用参考编码；B 从 FLAC 准备音色；C 复用参考与编译缓存、重新启动服务。C 此前包含编译的两条输出共耗时 375.24 秒，该次运行接续了启动超时前的部分编译结果。[安装记录、首次编译成本与转写 →](benchmarks/release-smoke.md#isolated-installation-and-public-reference)

需要 Linux x86_64、Python 3.12、`aria2c`、FFmpeg 共享库、C 首次 JIT 编译所需的 CUDA Toolkit/C++ 编译器，以及兼容 CUDA 13 的 NVIDIA 驱动。[准备系统依赖 →](docs/setup.md#prerequisites)

```bash
git clone https://github.com/torenyx/drama3-24gb.git
cd drama3-24gb
PYTHON=python3.12 bash scripts/setup.sh --profile c-int4
source .venv-c-int4/bin/activate

python scripts/download_models.py --profile c-int4 --model-root models
python scripts/sample.py --profile c-int4 --model-root models --gpus 2
```

下载器默认使用固定版本的魔搭镜像，每文件 16 路、四文件并行、支持续传。使用 `--source huggingface` 可切换官方 Hugging Face 固定版本。**A/B/C 共用一份约 41.5GB 官方模型**，转换会额外占用磁盘。

生成入口首次转换权重、准备仓库自带的 [LibriSpeech 参考音频和逐字稿](examples/librispeech/README.md)，启动服务并生成 WAV，结束后关闭自己启动的服务。结果保存在 `outputs/c-int4-2gpu/`，同时写入请求指标 `metrics.json` 和分阶段总耗时 `run.json`。

C 使用 `--gpus 1` 测试单卡。A/B 先安装对应 profile，再修改生成命令中的 `--profile`；A 用双卡，B 用单卡。[复用已有权重、转换空间、自定义音色与服务常驻 →](docs/setup.md)

## 我们做的运行优化

| 增量 | 实测结果 |
|---|---|
| **C：参考编码跨重启缓存，编码器按需加载** | 副卡常驻 **6209→2629 MiB**；首请求首个 PCM **0.632→0.336 秒** |
| **C：按流式窗口预算，CPU 先折叠 Codec 权重** | 主模型和波形解码装入**单张 24GB 卡**，采样峰值 **24022 MiB** |
| **A：独立常驻 Codec、复用模型放置、保持动态权重驻留** | 六次连续请求的后续耗时约 **13–20 秒**，原路径退化到 **39–63 秒** |
| **A：停滞片段局部重试** | 恢复可复现 seed 2 停滞时缺失的两句台词 |

C 缓存减少参考编码和编码器显存，重启对照中的模型启动仍为 41.12→40.74 秒。量化方法与生成内核来自上游。[优化前后数据 →](benchmarks/README.md)

## 质量与实验记录

C 整篇 p=0.9 的八条中，四条明显漏读、三条长静音；改为 p=1、三段独立请求后，该轮八条均未见这两类问题，热请求合计 4.77–5.10 秒。分段入口实现了这套策略，尚未自动转写验收或拦截全部漏读。

[异常开头、静音、提前 EOS 和提示词消融](docs/failures.md) · [原生 Fish Speech 对照](benchmarks/README.md#published-weights-with-native-fish-speech) · [CSV](benchmarks/measured-runs.csv) · [诊断工具](experiments/README.md)

## 上游与许可

[Fish Speech / Drama 3](https://github.com/fishaudio/fish-speech)、[ComfyUI-Drama3](https://github.com/PedroMarinhoDev/ComfyUI-Drama3)、[ComfyUI](https://github.com/Comfy-Org/ComfyUI)、[speech.cpp](https://github.com/nyosegawa/speech.cpp)、[SGLang-Omni](https://github.com/sgl-project/sglang-omni)、[INT4 转换器](https://huggingface.co/fishaudio/drama-3-int4)。固定版本见 [sources.lock.json](sources.lock.json)。

项目胶水采用 GPL-3.0-or-later，标记的适配文件保留 Apache-2.0。模型遵循 Fish Audio Community License；自带 LibriSpeech 参考音频遵循 CC BY 4.0。见 [NOTICE](NOTICE)。
