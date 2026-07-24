# 在 AutoDL RTX 4090D 上跑真实能耗测量 (EcoCompute)

这一套脚本把 `entrypoint.py`（NVML 直接采样的能耗测量）自动化，专门针对
**AutoDL 的 RTX 4090D（Ada 架构，24GB）** 实例，一条命令即可完成
**仓库/依赖安装 → 环境验证 → 全模型×精度扫描 → 结果汇总**，产出可直接用于
论文表格、以及粘贴回 [quantenergy.tech](https://quantenergy.tech) 曲线图的数据。

> RTX 4090D 在本项目里对应架构键 `ada`（`entrypoint.py` 里 `RTX 4090D` / `4090d` /
> `ada` 都会归一化到 `ada`）。目前网站上 Ada 的实测点只到 3B，本套脚本默认会把
> 测量扩展到 **7B**，并额外测 **INT8**（网站现有数据里 INT8 只在 A800 上测过）——
> 这正是补强数据、增强说服力的地方。

---

## 0. 一条命令跑完全流程

在 AutoDL 实例（已挂载 RTX 4090D）的终端里：

```bash
# 建议把仓库放在数据盘，避免占满系统盘
cd /root/autodl-tmp
git clone https://github.com/hongping-zh/ecocompute-mlcube.git
cd ecocompute-mlcube

bash autodl/run_all.sh
```

`run_all.sh` = `00_setup.sh` → `01_verify.sh` → `02_run_sweep.sh` → `03_aggregate.py`。
如果验证失败（例如 GPU/NVML 不可用）它会停下来；想强行继续跑 `--dry_run` 参考值，
用 `FORCE=1 bash autodl/run_all.sh`。

跑完后结果都在 `autodl/results/`，其中 `autodl/results/aggregate/` 是最终产物。

---

## 1. 分步说明

### `00_setup.sh` — 环境安装

```bash
bash autodl/00_setup.sh
# 只装环境、模型稍后再下：
SKIP_DOWNLOAD=1 bash autodl/00_setup.sh
```

- 开启 AutoDL **学术加速**（`source /etc/network_turbo`），并把 Hugging Face 端点
  指向镜像 `https://hf-mirror.com`（国内直连 `huggingface.co` 通常不通）。
- venv、HF 缓存、结果统统放到 **数据盘** `/root/autodl-tmp/ecocompute`（系统盘小）。
- venv 用 `--system-site-packages` 创建，**复用 AutoDL 镜像里已经装好、和驱动匹配的
  PyTorch**，避免自己下 torch 踩 CUDA 版本坑；只有在检测不到可用的 CUDA torch 时，
  才从 `cu121` 源装一份。
- 再装 `transformers / accelerate / bitsandbytes / nvidia-ml-py / sentencepiece` 等。
- 默认预下载 `autodl/models.txt` 里的模型到数据盘缓存。

> 选实例时，选官方 **PyTorch 2.x + CUDA 12.1** 之类的基础镜像最省事。

### `01_verify.sh` — 环境验证（关键）

```bash
bash autodl/01_verify.sh              # 完整验证 + 一次极小真实测量
bash autodl/01_verify.sh --no-measure # 跳过那次极小 GPU 测量
```

会依次检查：依赖能否导入、`torch.cuda` 是否可用、GPU 是否是 RTX 4090D/Ada、
**NVML 能否读到瞬时功率**（这是能不能出“真实测量”的决定性条件）、bitsandbytes 的
NF4 配置能否构建、CLI dry-run 能否产出 `energy.json`，最后用 Qwen2-0.5B 做一次
16 token × 2 次的极小真实测量，确认 `basis: "measured"`。任一 **必需项** 失败会以
非零码退出。

### `02_run_sweep.sh` — 全矩阵扫描（真实测量）

```bash
bash autodl/02_run_sweep.sh
```

对 `autodl/models.txt` 里每个模型 × `$PRECISIONS` 里每个精度调用一次 `entrypoint.py`。
有 GPU 时就是 **NVML 实采** 的真实测量。每个配置写一份：

```
autodl/results/<model_slug>/<precision>/energy.json
```

非 FP16 的run 会在 **同一次运行内** 再测一遍 FP16，从而得到自洽的
`vs_fp16_energy_pct`。某个配置失败（比如大模型 OOM）会记录并跳过，不中断整轮。

常用覆盖（环境变量）：

```bash
PRECISIONS="FP16 NF4"  TOKENS=512  ITERATIONS=20  bash autodl/02_run_sweep.sh
MODELS_FILE=my_models.txt          bash autodl/02_run_sweep.sh
SHARE=1 PREFETCH=1                 bash autodl/02_run_sweep.sh   # 顺带生成分享链接/对比网站预测
DRY_RUN=1                          bash autodl/02_run_sweep.sh   # 无 GPU：走数据集参考值
```

### `03_aggregate.py` — 汇总

```bash
python autodl/03_aggregate.py
```

扫描所有 `energy.json`，在 `autodl/results/aggregate/` 生成：

| 文件 | 用途 |
|---|---|
| `results.csv` | 每个 (模型, 精度) 一行：能耗/功率/吞吐/`vs_fp16`/`basis`/来源 —— 论文表格直接用 |
| `site_dataset.json` | 网站格式的 `{label, models:[{name,size,e:{FP16,NF4,INT8}}]}`，可直接粘回 quantenergy.tech 的 `CURVES` 数据 |
| `curves_anchors.json` | FP16 绝对能耗锚点 `[[N, mJ/token]]` 和各精度 `vs_fp16` 锚点，形状与 `entrypoint.py` 的 `REFERENCE`/估算器一致 |

它还会把你新测的值和网站上现有的 RTX 4090D 锚点做逐项 **百分比差异对比**，方便判断
新数据是否自洽。

---

## 2. 配置：要测哪些模型

编辑 `autodl/models.txt`，每行 `<hf_model_id> <params_b>`：

```
Qwen/Qwen2-0.5B                     0.5
TinyLlama/TinyLlama-1.1B-Chat-v1.0  1.1
Qwen/Qwen2-1.5B                     1.5
Qwen/Qwen2.5-3B                     3.0
Qwen/Qwen2-7B                       7.0
# 24GB 下 NF4/INT8 还能往上加，例如：
# Qwen/Qwen2.5-14B                  14.0
```

前四行对应网站上已有的 RTX 4090D 四个点，用来核对新测量是否吻合；7B 行把 Ada 的
实测范围往外扩，是新数据的主要来源。

---

## 3. 显存参考 (24GB / RTX 4090D)

| 模型规模 | FP16 | NF4 | INT8 |
|---|---|---|---|
| ≤3B  | ✅ | ✅ | ✅ |
| 7B   | ✅（约 14–15GB） | ✅ | ✅ |
| 13–14B | ⚠️ 偏紧/可能 OOM | ✅ | ✅ |

FP16 的大模型如果 OOM，脚本会跳过该配置继续跑；NF4/INT8 通常都能装下。

---

## 4. 常见问题

- **模型下载失败 / 卡住**：确认 `00_setup.sh` 里的镜像端点生效
  （`echo $HF_ENDPOINT` 应为 `https://hf-mirror.com`），必要时手动
  `source /etc/network_turbo` 再重试。
- **`basis` 不是 `measured` / 落回 dataset**：说明 NVML 功率读不到。先跑
  `bash autodl/01_verify.sh` 看 `[nvml power]` 那行；某些消费卡/vGPU/驱动会返回
  `NVML_ERROR_NOT_SUPPORTED`，此时无法出真实能耗，只能拿参考值。RTX 4090D 一般是
  支持功率读取的。
- **系统盘写满**：所有大文件都应落在 `/root/autodl-tmp`。脚本默认已经这么配置；
  只要仓库也 clone 在数据盘即可。
- **想固定依赖版本**：本目录为了在 AutoDL 上稳定运行，用的是较宽松的版本约束而不是
  仓库根目录 `requirements.lock.txt` 里那份（那份是给 CUDA Docker 镜像的）。要严格
  复现论文构建，请用根目录的 `Dockerfile` + `requirements.lock.txt`（但 AutoDL 实例
  本身是容器，通常不方便再跑 Docker，所以这里走的是原生 venv 路径）。

---

## 5. 拿到数据之后

1. 用 `autodl/results/aggregate/results.csv` 做论文/网站的能耗表；
2. 把 `site_dataset.json` 里 `"RTX 4090D"` 那块粘回 quantenergy.tech 前端的 `CURVES`
   数据，曲线图就会用你新测的实测点；
3. 也可以在每次 run 时加 `SHARE=1`，用生成的 `?tab=run&overlay=...` 链接把单点直接
   叠加到网站的 crossover 曲线上。
