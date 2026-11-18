# nvdex-ml-workflow 用户手册

## 1. 它能帮你做什么

这个 skill 用于 NvDEx TPC GAT 项目中的**已注册任务**：检查新的 H5 样本、解释并补齐实验选项、准备单 GPU 训练脚本，以及检查、绘制和分析训练结果。

目前注册的训练任务为 Z、XYZ 顶点重建、A 重建、θ 重建、二分类和五分类。程序每次从当前 `tpc_GAT/task_spec.py` 查询任务定义。它不在普通实验过程中新增物理标签、修改 GAT 主干或自动搜索模型结构。

支持 Linux 上的 `slurm` 和 `direct` 两种方式：前者生成 job 脚本，后者在用户自己的机器上直接运行。当前项目的大型训练仍由用户启动；准备完成、开始运行、检查通过和性能良好会分别报告。

## 2. 文件位置和安装

项目中的发布源位于 `skills/nvdex-ml-workflow/`，其中有 `SKILL.md`、辅助 Python 脚本和配置说明。训练代码和画图程序仍来自同一份 GAT 项目，不能只复制 skill 后就脱离项目使用。

推荐在运行 Codex 的电脑上，安装为当前项目的 skill。在项目根目录打开 PowerShell，先确认目的目录不存在，再执行：

```powershell
$skillSource = Join-Path (Get-Location) 'skills\nvdex-ml-workflow'
$skillDestination = Join-Path (Get-Location) '.agents\skills\nvdex-ml-workflow'
if (Test-Path -LiteralPath $skillDestination) { throw 'Skill destination already exists; review it before updating.' }
New-Item -ItemType Directory -Path (Split-Path -Parent $skillDestination) -Force | Out-Null
Copy-Item -LiteralPath $skillSource -Destination $skillDestination -Recurse
```

如果挂载盘不适合放 skill，可以安装到 Codex 用户 skill 目录。以下路径遵循本机已有 `$CODEX_HOME/skills` 布局；其他主机应按实际配置调整：

```powershell
$skillSource = Join-Path (Get-Location) 'skills\nvdex-ml-workflow'
$codexSkillRoot = if ($env:CODEX_HOME) { Join-Path $env:CODEX_HOME 'skills' } else { Join-Path $env:USERPROFILE '.codex\skills' }
$skillDestination = Join-Path $codexSkillRoot 'nvdex-ml-workflow'
if (Test-Path -LiteralPath $skillDestination) { throw 'Skill destination already exists; review it before updating.' }
New-Item -ItemType Directory -Path $codexSkillRoot -Force | Out-Null
Copy-Item -LiteralPath $skillSource -Destination $skillDestination -Recurse
```

安装一种即可，避免两个同名副本同时出现。安装后在 skill 选择器中检查 `nvdex-ml-workflow`；没有出现时重新打开聊天或重启客户端。skill 的官方加载与调用说明见 [Build skills](https://learn.chatgpt.com/docs/build-skills)。项目副本和安装副本需要同步更新，尤其不能让旧说明调用不同版本的辅助脚本。

Linux 用户可以在自己的项目根目录安装项目级 skill：

```bash
test ! -e .agents/skills/nvdex-ml-workflow || { echo 'Destination exists; review before updating.' >&2; exit 1; }
mkdir -p .agents/skills
cp -R skills/nvdex-ml-workflow .agents/skills/nvdex-ml-workflow
```

IHEP 默认部署中，AI 在本机读取说明，通过 `ihep-codex` 在服务器运行检查；远程项目是 `/lustrefs/juno26/users/liuz3/work/test/GAT`，本地挂载是 `Z:/work/test/GAT`。Windows 的 skill 路径不能直接传给服务器执行。其他用户在自己的 Linux 主机上执行，无需配置这个 SSH 别名或 IHEP 账号。

### 默认部署与自己的环境

`assets/ihep-deployment.json` 同时是实际默认配置和 IHEP 部署示例：

| 项目 | IHEP 默认值 |
| --- | --- |
| 项目回退路径 | `/lustrefs/juno26/users/liuz3/work/test/GAT` |
| Python | `/hpcfs/juno/junogpu/liuz3/conda/envs/ml-v100-torch251/bin/python` |
| 执行方式 | `slurm`，单节点、单 GPU |
| 资源 | gpu / junogpu / normal，5 CPU，64G，16 小时 |

当前目录能找到项目时优先使用当前项目，`--project` 可显式指定。用户自己的环境应选择 `deployment: current`，这会记录执行 helper 的 Python，并默认选择 direct。也可以显式设置顶层 `python`；prepare 必须使用所选 Python 执行。更换部署不会自动改变物理任务、输入特征或训练超参数。

## 3. 最常用的两种提问

### 已经知道实验配置

```text
$nvdex-ml-workflow

输入文件是 /lustrefs/juno26/users/liuz3/work/test/GAT/input/new_sample.h5。
做 theta 重建，使用真实 vertex，radius_mm=4。
使用 theta-angle-v1 参考配置，其余参数沿用配置。
检查输入，生成训练任务，并告诉我如何提交。
```

把示例路径换成真实样本。skill 会匹配 `regression_theta`、`angle` 和 `vertex_relative`，保留你指定的 4 mm 半径；检查数据后，输出最终配置、输入检查 PDF 和 job 脚本。

### 只知道研究目标

```text
$nvdex-ml-workflow

我想用这份 H5 做 theta 重建，但还不知道怎样选择顶点参考和建图参数。
文件是 /lustrefs/juno26/users/liuz3/work/test/GAT/input/new_sample.h5。
先检查字段，解释关键选项，给出起始配置建议，再准备实验。
```

AI 可以先列出文件结构，然后集中询问关键决策，不要求你逐项填写所有超参数。

| 选项 | 含义 |
| --- | --- |
| `truth` | 使用 MC 真实顶点构造相对特征，属于使用额外 truth 信息的实验 |
| `centroid` | 使用所有 hit 的非能量加权 XYZ 几何中心构造相对特征 |
| `none` | 不增加顶点参考特征 |
| `angle` | 直接重建角度，单位为度 |
| `cdf` | 重建 cos(θ) 的训练集经验 CDF，并反变换 |

θ/A 的参考配置不默认开启任何顶点模式，必须由你选择或在对话中明确授权选择。`centroid` 和 `truth` 都需要 `hits/z`。如果目标是不使用 truth，应在请求中直接说明；如果还没有确定，可以要求 AI 比较三者的研究含义。

半径缺省时，可以明确同意采用某个参考配置的起始值，也可以要求先看邻居和边数诊断。参考配置半径是 2 mm；它没有被证明是新样本的最优值。AI 不会仅凭图连通性宣布最优半径。

## 4. 参考配置和参数继承

| 配置名 | 任务 | 目标模式 |
| --- | --- | --- |
| `theta-angle-v1` | θ | angle |
| `theta-cdf-v1` | θ | cdf |
| `A-v1` | A | standard |
| `xyz-v1` | XYZ | minus_one_one |
| `z-v1` | Z | minus_one_one |
| `binary-v1` | 二分类 | none |
| `multiclass-v1` | 五分类 | none |

共同起始设置：30 epochs、batch 64、学习率 3e-4、weight decay 5e-5、128 hidden channels、3 layers、4 heads、seed 42、20% validation、radius 2 mm。A/θ/XYZ 使用 x/y/z/energy/log_energy；Z/PID 使用 x/y/energy/log_energy。这些是明确可追溯的参考配方，不能理解为所有旧结果的原样复现。

显式指定的参数覆盖参考值，覆盖来源保存在 `parameter_origins`。例如 `radius_mm=4` 不会被参考配置中的 2 mm 改回去。自定义配置不选 preset 时，任务、输入、features 和半径需要明确；θ 还需要目标模式，θ/A 需要顶点模式。

### 在自己的 GPU 服务器上直接运行

可以这样告诉 AI：

```text
$nvdex-ml-workflow

我的项目在 /home/alice/GAT，使用 deployment=current、direct 模式，不提交 job。
环境初始化脚本是 /home/alice/setup_ml.sh，使用 GPU 0。
输入文件是 /home/alice/data/new_sample.h5。
使用 theta-angle-v1、真实 vertex、radius_mm=4。
检查数据并生成运行脚本，由我启动训练。
```

AI 应先在目标机器加载你指定的环境，用该环境的 Python 准备实验。生成的 `run.sh` 会再次加载同一脚本，再使用记录的 Python 训练。如果环境已经激活、不需要额外脚本，可以省略 setup_script；在新终端启动时仍需激活相同环境。

对应的 request 示例也保存在 `assets/direct-request.example.json`：

```json
{
  "deployment": "current",
  "preset": "theta-angle-v1",
  "infile": "/home/alice/data/new_sample.h5",
  "vertex_source": "truth",
  "execution": {
    "backend": "direct",
    "device": "cuda",
    "gpu": "0",
    "setup_script": "/home/alice/setup_ml.sh"
  },
  "parameters": {"radius_mm": 4.0}
}
```

setup_script 必须是你信任的、非交互 Bash 脚本，可以为 null。准备和运行时均从项目根目录 source；脚本中的相对路径应以此为准。工具检查所选解释器和包版本，并在启动前核对脚本内容指纹。工具不会猜测一个任意脚本应当激活哪个 Python；应先 source，再用相应 Python prepare。初始化探测超过 60 秒会失败并提示调整脚本。

`gpu: "0"` 表示设置 CUDA_VISIBLE_DEVICES=0；省略/null 会继承当前可见 GPU。direct 要求运行时恰好一张 GPU 可见；SLURM 模式由调度器分配，不允许另设 gpu。CUDA 不可用时明确失败。`device: cpu` 仅用于明确选择的 CPU 实验或小测试，不会自动降级。direct 配置不能填写 SLURM resources。

SLURM 会保留加载 setup 之前调度器提供的 GPU 可见性，避免脚本中的 GPU 设置覆盖分配。source 期间允许普通的未定义变量展开，兼容常见的环境变量追加写法。

## 5. 任务准备后会得到什么

实验目录包含：

- `request_resolved.json`：最终生效的配置和参数来源。
- `inspection.json`：样本数、字段存在性、实际 loader 抽样图诊断和局限。
- `input_checks.pdf`：目标分布、抽样 hit 特征分布与构图诊断图。
- `plan.json`：输入身份、代码指纹和 Python/包版本等运行依据。
- `job.sh` 或 `run.sh`：分别用于 SLURM 和 direct。
- `PLAN.md`：可直接复制的启动命令和输出位置。

检查通过后，按 `PLAN.md` 启动：SLURM 使用 `sbatch .../job.sh`，direct 使用 `bash .../run.sh`。prepare 不启动训练。每个准备目录只允许一次启动；需要重试时重新准备到新目录。输入、相关代码或 setup 脚本改变时会停止，并提示重新检查；脚本指纹在 source 之前核对。

结果位于 SLURM 的 `results/<job_id>/` 或 direct 的 `results/direct/`，训练完成后自动生成 `workflow_review/`。这不要求聊天一直在线，但 direct 是前台进程，关闭终端或断开 SSH 可能中断，用户需要自己管理会话。`workflow_status.json` 记录运行方式、设备和 training/reviewing/complete/failed 状态；complete 不代表物理性能达到目标。

## 6. 查看和分析已有结果

```text
$nvdex-ml-workflow

检查结果目录 /lustrefs/juno26/users/liuz3/work/test/GAT/results/my_run。
把新的检查图放到 check/my_run_review_01，保留原始结果。
分析 theta 是否出现 90 度预测带，并区分观测现象和可能原因。
```

AI 会先核对训练/验证身份、原始 H5 标签、变换和单位，再出图。所有注册任务都有通用图；A/θ/XYZ 还会调用现有专用画图程序。多分类不会套用二分类的“第 2 列就是信号分数”约定。

对于 A，只有可用事例的 E1+E2 经检查近似恒定时，才调用会生成固定 Q 值 Emax 图的专用脚本；总能量变化的样本只输出 A 的通用性能图，并明确提示原因。

输出中的 `metrics.json` 是计算结果，`report.md` 是报告起点，AI 会结合实际图和样本数补充中文分析。原始结果不覆盖；旧 loader 补零但源文件实际缺失的诊断字段，不参与新图。

旧结果没有新 manifest 时，仍可与当前 H5 的标签核对，但无法证明该 H5 和代码就是训练时的原始版本。没有源 H5、训练未完成、标签错位或变换不一致时，会明确失败，不会绕过检查画一份看似正常的图。

## 7. 不通过 AI 也能使用

所有辅助命令都是普通 Python CLI。服务器操作如下；示例 request 路径和输出目录需要换成自己的新目录。

```bash
cd /lustrefs/juno26/users/liuz3/work/test/GAT
PY=/hpcfs/juno/junogpu/liuz3/conda/envs/ml-v100-torch251/bin/python
WF=skills/nvdex-ml-workflow/scripts/workflow.py
"$PY" "$WF" catalog
"$PY" "$WF" inventory --infile input/new_sample.h5
"$PY" "$WF" resolve --request check/theta_request.json
"$PY" "$WF" prepare --request check/theta_request.json --outdir jobs/prepared/theta_01
"$PY" "$WF" review --indir results/my_run --outdir check/my_run_review_01
```

`theta_request.json` 的最小参考配置例子：

```json
{
  "schema_version": 1,
  "preset": "theta-angle-v1",
  "infile": "input/new_sample.h5",
  "vertex_source": "truth",
  "parameters": {"radius_mm": 4.0}
}
```

`resolve` 返回 2 表示还有待选择项，返回 1 表示无效配置或运行错误，返回 0 表示该命令成功。完整配置说明位于 skill 的 `references/workflow.md`。

在其他用户的 Linux 主机上，先加载环境再准备、运行：

```bash
cd /home/alice/GAT
source /home/alice/setup_ml.sh
python skills/nvdex-ml-workflow/scripts/workflow.py --project /home/alice/GAT prepare \
  --request check/theta_request.json --outdir check/prepared/theta_01
bash check/prepared/theta_01/run.sh
```

这里的 request 必须选择 `deployment: current` 和 direct，文件路径需换成真实路径。

## 8. 已知范围

当前支持 Linux/Bash 下的单 GPU direct/SLURM，以及显式 direct CPU；没有验证原生 Windows GPU 训练。复用已有 event-row train/validation split。不会自动去掉边缘事例、修改标签定义、重新生成 H5、无限重试或自动寻找最优超参数。输入检查会完整扫描所选字段；抽样构图有上限，但训练本身没有因此获得显存保证。

输入身份采用路径、大小和修改时间，代码和 setup 入口脚本使用 SHA-256 指纹；这些不是完整归档，setup 内部再 source 的其他文件及任意 shell 状态没有自动封存。准备目录不能直接跨机器搬运，应在目标主机重新准备。H5 中归一化统计的来源需要另行确认；验证集调参结果不能自动当作独立测试集性能。

升级 workflow 后，旧准备目录可能因代码指纹变化而拒绝启动；应根据原始 request 重新准备到新目录。不要修改旧 manifest 绕过检查。
