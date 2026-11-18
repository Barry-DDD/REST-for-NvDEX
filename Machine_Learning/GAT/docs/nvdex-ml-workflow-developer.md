# nvdex-ml-workflow 开发者说明

## 1. 设计目标与责任划分

目标是在不修改现有训练算法的前提下，把已注册任务的日常工作收敛为可检查的流程：自然语言需求 → 明确实验配置 → 输入检查 → 可审查的 direct/SLURM 启动脚本 → 已完成产物检查 → 绘图和解释。

skill 负责语义、选项解释、调用和基于证据的分析。Python 负责配置约束、实际数据读取、事例对齐、变换检查、文件生成和退出状态。训练仍由 `tpc_GAT/run.py` 执行。此版本不实现任务生成器、模型搜索平台或独立调度系统。

```text
SKILL.md + references/workflow.md
           |
      workflow.py
       /    |     \
config     data   results
   |        |       |
TaskSpec  actual   source labels + saved NPZ
          loader    + existing plotters
           |
    unchanged training entry
```

## 2. 源码组织

| 文件 | 作用 |
| --- | --- |
| `skills/nvdex-ml-workflow/SKILL.md` | 场景匹配、缺失决策处理、授权边界、结果解释规则 |
| `agents/openai.yaml` | UI 名称和自动匹配策略 |
| `references/workflow.md` | CLI、配置和部署细节，按需加载 |
| `scripts/workflow.py` | CLI、准备目录、direct/SLURM 脚本和共同执行入口 |
| `scripts/workflow_config.py` | 参考配置、严格解析、argv 和代码指纹 |
| `scripts/workflow_execution.py` | 环境脚本探测/指纹、设备选择、两种启动脚本共享的环境初始化 |
| `assets/ihep-deployment.json` | IHEP 默认路径、Python、SLURM 与资源，兼作部署示例 |
| `assets/direct-request.example.json` | 当前环境下直接运行的请求示例 |
| `scripts/workflow_data.py` | H5 检查、实际 loader 图抽样、输入图 |
| `scripts/workflow_results.py` | 事例/标签/变换核对、指标、通用图和专用图适配 |
| `tests/test_nvdex_ml_workflow.py` | 合成数据、拒绝条件、小规模真实 CPU 训练和结果验证 |
| `tests/test_nvdex_workflow_execution.py` | 默认部署、direct 完整流程、环境脚本与设备边界 |

安装副本只负责让 AI 发现 skill。远程执行优先使用项目发布源中的 helper，不能把 Windows 安装路径当成远程路径。未引入新的包；使用已有 numpy/h5py/PyTorch/PyG/Matplotlib/scienceplots/scikit-learn 环境。

## 3. 任务定义与参考配置

任务语义的来源始终是 `tpc_GAT/task_spec.py`：输出维度、类别映射、A/θ/XYZ 读取、目标变换和 XYZ 指标直接复用，避免第二套物理公式。

参考配置只描述实验起始配方。A/θ/XYZ 的共同参数来自当前 detsim job；Z/PID 使用显式记录的同类起始设置，不声称复现所有历史实验。A/θ 的顶点来源没有默认值，θ 模式必须由显式模式或配置名确定。

配置是 JSON，schema_version=1。根字段限制为 preset/task/infile/vertex_source/target_mode/parameters/resources/python/deployment/execution。未知字段被拒绝，避免拼错参数后静默走默认值。参数通过 argv 数组传给原始 CLI，布尔 flag 和列表按 argparse 语义展开，不拼接任意训练命令。

任务 preset 与部署设置相互独立。默认 deployment=ihep，从版本化资产加载 Python、SLURM 和资源；项目根目录优先显式 --project，其次当前目录发现，最后回退到 IHEP 路径。deployment=current 捕获 helper 的 sys.executable，并默认 direct。显式 python 覆盖部署值；prepare 要求当前解释器与所选解释器一致，避免在 A 环境检查、B 环境训练。

解释器身份比较保留绝对调用路径，不把两个 venv 的 Python 符号链接折叠成同一个系统二进制；运行记录还包含 sys.prefix/base_prefix。若使用另一条解释器路径，应直接用那条路径运行 prepare。

execution 只允许 backend/direct 或 slurm、device/cuda 或 cpu、可选单 GPU 索引字符串 gpu，以及可选 setup_script。direct 不接受 SLURM resources；slurm 只接受 CUDA 且不允许覆盖调度器分配的 GPU。CPU 必须显式选择且仅用于 direct。运行方式不会改变模型、标签、数据划分或训练参数。

`resolve()` 对未确定的科学决策返回 `needs_input` 和结构化 questions，CLI 返回 2。它不在配置不完整时自行决定 truth 或 theta 的训练空间。数值、类型、范围、资源字符串和已有任务组合会进一步验证。`parameter_origins` 记录每个训练参数来自用户、preset 或可见默认值。

第一版仅允许 A/θ 显式使用 truth 顶点，其他任务使用 none；这比底层 loader 的部分可选组合更窄，用于避免顶点重建和 PID 日常流程混入额外 truth 输入。扩展允许组合时，需要明确其研究语义并加测试。

## 4. 输入检查的实际范围

`inventory` 只读取字段、shape、dtype 和 attributes，供选项尚未明确时使用；不会声称文件已经适合训练。

`inspect`/`prepare` 检查所有所选 hit 字段的 shape 和有限值，以分块方式读取，避免把整个 hit 表同时放入内存；检查正能量 hit 契约、整数 module ID、事件 offset/count 的边界和重叠。目标标签由原始读者读取，因此两电子、粒子种类、方向和能量条件与训练一致。分类额外禁止把浮点 raw ID 截断成整数；每个注册类别至少要有两个可用事例。

还会检查输入 event features、所有已存在且被 loader 消费的 event/primary 元数据、stats 长度/有限值/非负标准差/极值顺序、可用事例数量和训练 full-batch 数。这样未抽中的可选元数据异常也会在训练前失败。坐标 stats 与坐标数据分开处理：固定尺度的 x/y/z 不要求出现在 stats/feature_names 中，但实际所需 hits 字段不能缺。

完整选择字段检查后，用真实 `TPCGraphDataset` 构建固定随机种子的少量 CPU 图，核对 node/edge/label 的有限值，并记录节点数、边数和孤立比例。默认抽样 24 个事例；超过 2048 hits 的抽样事例跳过并报告，防止诊断阶段触发二次方内存开销。全部无法抽查时失败。这不是对训练大图内存的保证。

当前 runner 仍读取 H5 中既有的归一化 stats，并按 event row 划分训练/验证；工具会报告其来源和独立性局限，不在这次工作中改变历史算法。

## 5. 准备与执行

`prepare` 在 Linux 上用所选训练 Python 执行，先验证可选环境脚本，再检查数据并创建新目录。SLURM 生成 job.sh，direct 生成 run.sh；均保存配置、输入报告、图、manifest 和 PLAN.md。脚本采用绝对路径和 shell quoting。没有 submit 子命令，不调用 sbatch，也不在 prepare 中启动训练。

setup_script 是用户明确指定的可信文件。准备时，在项目根目录的非交互 Bash 中 source，再用所选 Python 检查解释器和包版本；限制 60 秒，标准输出中允许普通初始化信息，通过固定 JSON 标记解析探测结果。用户需先激活目标环境再 prepare，工具不会推测任意 shell 脚本的 Python。生成的脚本先通过 sha256sum 检查内容，再 source，同样以项目根目录为工作目录，并记录加载标记。execute 会再次核对内容和加载标记；直接绕过 launcher 调用内部 execute 不能跳过这一步。

环境探测与最终 launcher 使用同一个 shell 环境构造函数。source 期间暂时关闭 nounset，兼容 `$LD_LIBRARY_PATH` 等初始未定义的变量；之后恢复严格模式并核对 source 状态。SLURM 在 source 前保存 CUDA_VISIBLE_DEVICES 的存在性和值，source 后恢复调度器分配；direct 则按显式设备选择设置或继承可见性。

manifest 用 path/size/mtime_ns 标识输入；用 SHA-256 记录训练代码、workflow helper、IHEP 默认配置和专用画图程序，并单独记录 setup 入口脚本的内容指纹、Python 和包版本。执行前重新比较。它依赖原始文件仍可用，不是完整 snapshot；嵌套 source 文件、全部 shell 状态、驱动和外部库没有自动封存。更强复现需要另做归档。复制到其他机器或切换环境后应重新 prepare。

`execute` 根据记录的 backend 决定调度约束：slurm 要求数字 SLURM_JOB_ID，结果为 results/<job_id>/；direct 不读取或伪造 job ID，结果为 results/direct/。两者共同验证输入、代码、setup 和 Python/package 身份。CUDA 模式要求恰有一张可见 GPU，失败时不退到 CPU；显式 CPU 模式在子进程隐藏 CUDA。独占创建 launch.lock 防止重复启动。状态包含 backend/device/run_id，按 training → reviewing → complete 变化，异常写 failed 并退出，不自动重试。

训练仍通过 `python -m torch.distributed.run --standalone --nproc_per_node=1` 调用现有 run.py。两种 launcher 都在训练后执行 review，不依赖 AI 聊天常驻。direct 是前台进程，不提供终端断开后的进程保活；也没有新增网络服务或后台代理。

## 6. 结果验证、绘图与分析

结果检查先要求模型文件和最终训练 metadata，核对完整训练 history。根据源 H5 重新读取 raw targets，检查训练/验证集合非空、索引唯一、互不相交、覆盖指定可用事例；evaluation 必须精确对应 validation，并逐事例匹配原始标签。

回归检查前向标签变换、预测反变换、目标单位、仿射常量；CDF 模式从训练标签重拟合并核对保存的 knots。XYZ 保留固定轴顺序。分类检查 raw ID 映射、logits/softmax、预测类别和 confusion matrix，维度取自 TaskSpec，不从二分类惯例推断。

新流程的结果还要与 manifest 核对配置和指纹。旧结果缺少这些证据时，明确标记 legacy，只能证明它与当前可访问源标签一致，不能证明训练时的原始文件身份。

原始 dataset 会给缺失的诊断 metadata 填零。review 在源 H5 检查存在性；源缺失的字段会从新的 plotting copy 中删除。必要目标缺失仍然失败。这样可以避免把缺失的 vertex、能量等解释为真实零值。

所有任务生成通用学习曲线、回归响应/残差或分类 confusion/ROC。A/θ/XYZ 另外复用现有专用 plotter，在 `validated_artifacts/` 中运行；原始训练文件不变。Z/PID 先使用通用图，避免旧 Z 脚本的默认能量 cut 和旧 binary PID 脚本的固定类别假设。

A 的专用脚本隐含固定 Q 的 Emax 换算，因此 adapter 只在所有可用事例 E1+E2 近似恒定（rtol=1e-6，atol=1e-5，采用源字段单位）时启用。源 E1/E2 与 evaluation 元数据也逐事例核对；总能量变化时保留 A 通用图并报告跳过原因。

`metrics.json` 保存可计算的数值，`report.md` 提供报告起点。物理解释由 AI 阅读指标和实际图后完成，不通过固定规则自动宣告“边界效应”或“重建成功”。AI 应记录图是否经过视觉检查，区分观测、推测与下一步检验。

## 7. 测试与维护

服务器执行：

```bash
cd /lustrefs/juno26/users/liuz3/work/test/GAT
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 \
  /hpcfs/juno/junogpu/liuz3/conda/envs/ml-v100-torch251/bin/python \
  -m unittest discover -s tests -p 'test_nvdex*.py' -v
```

测试使用临时合成 H5 和临时输出目录，不读取批量生产数据、不申请 GPU、不调用 scheduler。覆盖：缺失选择、未知参数、显式覆盖、手算 θ/A 标签、真实图边数、坐标 stats 兼容、缺失 truth/坐标、事件 overlap、非法方向/ID、重复准备、无 SLURM 时拒绝执行，以及六个任务和 θ 两种模式的两轮 CPU 训练/结果检查。绘图测试覆盖通用图、θ 专用图和多分类路径。

还覆盖所有现存 loader 元数据的异常、非法归一化统计、准备 manifest 的正常匹配，以及输入/代码/运行配置漂移；总初级能量变化的 A 样本验证了固定 Q 的 Emax 图不会被生成。2026-11-06 的最终 9 项测试全部通过；代表性 PDF 页面完成视觉检查。记录见 `check/nvdex_ml_workflow_qa_20261106/VALIDATION.md`。没有执行真实 SLURM/GPU 训练。

双模式扩展另外测试 IHEP/current 解析、非法运行组合、调度器 GPU 分配继承、缺失/失败/变更的 setup 脚本、无 GPU 和多 GPU 时拒绝 CUDA，以及包含空格和引号的 launcher 路径。direct 完整集成测试实际 source 自定义脚本，执行两轮合成数据 CPU 训练并自动 review，再验证重复运行被拒绝；不需要 SLURM、GPU 或生产样本。

扩展后的最终结果为 **18 项测试全部通过，154.161 秒**，还覆盖 source 的未定义变量兼容、脚本覆盖 GPU 可见性后的恢复，以及 venv 符号链接身份。新记录在 `check/nvdex_ml_workflow_portability_qa_20261106/VALIDATION.md`。真实 CUDA/SLURM GPU 硬件路径仍需用户首次运行时验证。

skill 本身还应经过 skill-creator 的 quick_validate 和真实提示词行为检查；格式校验不等于行为正确。更新代码后重新运行相关测试，更新安装副本。新增注册任务不会自动获得整条 workflow 支持：需审查目标读者、参数约束、结果 schema 和 plot adapter，再补对应测试。

## 8. 已知局限与扩展方向

- 支持 Linux/Bash 下 direct 或单节点 SLURM，均限单 GPU，另有显式 direct CPU；没有原生 Windows GPU、DDP 资源编排或跨任务搜索。
- 没有改变数据划分，没有实现源事件分组、固定独立测试集或训练集专属输入统计。
- 输入 stat 指纹不能抵御内容变化但 size/mtime 被人为保持不变的情况；代码指纹也不是运行环境封存。
- 图诊断是有限抽样，不能证明所有训练图都适配显存。
- 目前通用 PID 图不复现旧脚本全部能量效率和阈值扫描；这类诊断应在明确变量含义、样本权重和阈值选择数据后扩展。
- 90 度带的专门事例诊断可以结合现有 `check/diagnose_theta_failures.py`，但运行前需要核对脚本的实验条件和源文件身份，不能机械套用。

后续优先补实际使用中暴露的具体缺口，不先引入新的任务注册框架或抽象调度平台。
