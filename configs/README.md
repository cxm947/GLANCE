# configs/

运行配置的集中存放处, 是 EDL-LLM 的**单一配置真相**: `scripts/run_e2e.py`(主线)与
`scripts/run_ablation.py`(消融)都经 `loader.load_config()` 从这里取配置, 不再各自内联 cfg。

## 文件
- `default.yaml` — canonical 论文主线形态(完全展开, 不依赖代码默认值), 产出 `edge_parent_f1=0.8621`
  @ `outputs/`。改这里 = 改基线, 需重测全部报告。每个影响分数的键都显式覆写了
  `pipeline_clean.py` 的代码默认(见文件内注释), 漏一键分数必飘。
- `experiments/*.yaml` — 消融组, 只写相对 `default.yaml` 的覆盖项, deep-merge 到 default 之上。
  每个文件 = 一个单变量消融(C1−识别 / C2−推理 / C3−短期记忆 / C4−长期记忆 / C5−验证 /
  C6−路由回 / C7 naive / C8−推理记忆), 由 `scripts/run_ablation.py` 驱动。
- `loader.py` — `load_env()`(从 profile 对应 secrets / `.env` 注入密钥到 `os.environ`)
  + `load_config(experiment=None, *, profile=None, ...)`(读 default.yaml [+ 实验覆盖] + 注入密钥/路径),
  产出 dict 直接喂 `CleanPipeline(config)`。
- `providers.yaml` — OpenAI 兼容 provider 档案: profile → secrets 文件映射。两套:
  `deepseek`(canonical / 0.8621)与 `qwen`(SiliconFlow 实验 / 备用)。
- `secrets/` — (git 忽略) 明文密钥文件 `.edl_env*`(各含 `EDL_API_KEY` / `EDL_BASE_URL` / `EDL_MODEL`);
  **绝不入库**。

## 关键约束 (铁律)
- `src/` 任何模块**都不 import 本目录** —— configs 只给 scripts / loader 读, 保持 src import 纯净。
- `default.yaml` 是主线 cfg 的唯一真相; 它与 `run_e2e.py` 历史内联 cfg 逐键等价(离线对拍守门过)。
  改任一影响分数的键就改了 headline `0.8621`。
- 密钥**绝不写进任何 yaml/py/json**; `default.yaml` 里 `base_url`/`model` 是非密占位, `api_key` 根本不出现。

## 用法
```python
from loader import load_config      # 脚本里 sys.path 已含 configs/

# 主线 (= run_e2e):
cfg = load_config(memory_dir="outputs/_mem", output_dir="outputs")
# cfg 即可喂 CleanPipeline(cfg)。密钥从 env 注入, 缺 key 显式报错。

# 消融组 (= run_ablation):
cfg = load_config("c7_naive", memory_dir=..., output_dir=...)   # = default + experiments/c7_naive.yaml

cfg = load_config(profile="qwen")   # 切 SiliconFlow Qwen(实验/备用); 也可设环境变量 EDL_PROFILE=qwen
```
**canonical 固定 deepseek(0.8621), 别改 `providers.yaml` 的 `default`。**
