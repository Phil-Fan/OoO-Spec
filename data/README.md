# OoO-Spec 数据管线说明

## 目录结构

```text
data/
├── raw/                   # 原始训练数据下载位置
├── downloads/             # 上游仓库/HuggingFace snapshot 缓存
├── evalsets/              # 评测集和去污染参考数据
│   ├── apibank/
│   ├── toolalpaca/
│   └── bfcl/
└── train/                 # 处理好的训练/评估中间产物
    ├── src_requests/      # 源请求（去污染、split 后）
    │   ├── train.jsonl    # 9,662 条
    │   └── dev.jsonl      # 529 条
    ├── teacher_traces/    # Qwen2.5-32B-Instruct 生成的归一化调用
    │   ├── train.jsonl
    │   └── dev.jsonl
    └── rows/              # 语义行（sidecar 训练数据）
        ├── train.jsonl    # 75,567 条
        └── dev.jsonl      # 3,239 条
```

## 数据来源与当前状态

### 下载脚本

统一下载入口：

```bash
python data/download_datasets.py
```

脚本会把外部下载缓存放在 `data/downloads/`，并把代码实际读取的文件整理到：

- `data/raw/`
- `data/evalsets/`

如果某个上游文件没有稳定公开路径，可用参数补充：

```bash
python data/download_datasets.py \
  --mobile-actions-url URL \
  --bfcl-v3-simple-url URL
```

### 原始数据集放置位置

| 数据集 | 位置 | 内容 | 条数 |
|---|---|---|---|
| API-Bank train lv1 | `data/raw/api_bank/lv1-train.json` | 原始训练数据 | **6,184** |
| API-Bank train lv2 | `data/raw/api_bank/lv2-train.json` | 原始训练数据 | 9,279 |
| API-Bank train lv3 | `data/raw/api_bank/lv3-train.json` | 原始训练数据 | 1,245 |
| ToolAlpaca | HuggingFace repo/cache `Ahren09/ToolAlpaca` | train / test | **4,046 / 68** |
| BFCL v3 | `data/evalsets/bfcl/` | 多类别评测数据 | — |
| BFCL v4 | `data/evalsets/bfcl_v4/` | 多类别评测数据 | — |

本地数据不在这些默认位置时，可以通过脚本参数或 `OOOSPEC_*` 环境变量指定。

### ToolSpec 已处理的数据（`data/evalsets/`）

| 数据集 | 文件 | 条数 | 用途 |
|---|---|---|---|
| API-Bank | `level-1-api_processed.json` | 399 | 评测 |
| API-Bank | `level-2-api_processed.json` | 67 | 评测 |
| API-Bank | `level-3-api_processed.json` | 131 | 评测 |
| ToolAlpaca | `toolalpaca_processed.json` | 4114 | 评测/训练混合 |

**注意**：ToolSpec 目录下只有 **597 条 API-Bank** 和 **4114 条 ToolAlpaca**，对应的是**评测集**。OoO-Spec 论文要求的训练规模：

- API-Bank 训练集：**6,200 条**（`lv1-train.json` 的 6,184 条非常接近）
- ToolAlpaca 训练集：**3,462 条**（`Ahren09/ToolAlpaca` train 的 4,046 条足够，去污染后保留约 3,462）
- 合计源请求：**9,662 训练 / 529 开发**

因此，OoO-Spec 的数据管线**应基于找到的原始数据构建**，ToolSpec 已处理的数据只用作评测集/格式参考。

## OoO-Spec 数据管线四阶段

### 阶段 A.1：获取原始数据集

需要下载/准备的原始数据：

1. **API-Bank**（原始仓库或官方版本）
2. **ToolAlpaca**（`Ahren09/ToolAlpaca` 或 `tangqiaoyu/ToolAlpaca`，后者用于 Golden Answers）
3. **BFCL Java/JS**（用于零样本迁移评测，不进入训练）

### 阶段 A.2：源请求 split（9,662 / 529）

处理规则（来自 `plan/repro.md` §2.1）：

1. **ToolAlpaca 只取 training file**，不用 eval file。
2. **去污染**：只要某个 API 的 `API name` 或任一 `function name` 出现在评测清单（API-Bank / ToolAlpaca-eval / BFCL Java-JS）中，**整条 API 删除**。
3. **不使用** ToolAlpaca 的 golden answers 和 tool-execution outputs。
4. 用**确定性 prompt-hash split**；断言 request-ID 零重叠且 prompt-hash 零重叠。
5. 最终统计必须复现论文 Table 1。

目标：

| Source | Training | Development |
|---|---|---|
| API-Bank | 6,200 | 335 |
| ToolAlpaca | 3,462 | 194 |
| **Combined** | **9,662** | **529** |

### 阶段 A.3：教师 trace 生成

输入：dialogue + 该请求的 tool schemas  
模型：**Qwen2.5-32B-Instruct**（固定）  
解码：greedy, batch size 1

解析与裁剪：
1. 解析 supported tool-call surface forms 为 strict-JSON domain。
2. 保留**第一个有效调用**。
3. 只保留 `normalized function name` + `argument map`。
4. 丢弃教师 token IDs、chat-template tokens、tool-call control tokens。

产物格式（每条一个对象）：

```json
{
  "request_id": "...",
  "prompt_hash": "...",
  "call": {
    "name": "<normalized_fn>",
    "parameters": {"<arg>": <value>, ...}
  }
}
```

### 阶段 A.4：语义行展开（核心）

在 **Qwen3-0.6B no-thinking chat template** 下渲染。每次归一化调用展开成 **3 类行**：

| Row type | Completion target | 在线是否查询 |
|---|---|---|
| **Function index** | 所选函数在当前 schema 中的局部索引 | ✅ |
| **Argument value or null** | 该参数的紧凑 JSON 值；否则 `null` | ✅ |
| **Direct call（辅助）** | 归一化对象（name + parameters） | ❌ 仅训练 |

展开规则：
1. **每个 schema 定义的参数都有一行**，包括未被选中函数的参数 → target = `null`。
2. **参数 prompt 不接收 function index 的答案**。
3. 每个 prompt 含 dialogue + 带索引的 schemas；prompt 段 label 全部 mask，loss 只算 completion。

目标行数：**75,567 训练 / 3,239 开发**。

## 下一步操作建议

1. 运行 `python data/download_datasets.py` 下载公开数据源。
2. 如果已有本地数据，用脚本参数或 `OOOSPEC_*` 环境变量指定路径。
3. 处理产物统一写入 `data/train/`，该目录默认被 `.gitignore` 忽略。
