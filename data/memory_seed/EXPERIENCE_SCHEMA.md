# 统一经验体系 (Experience Schema)

GLANCE 的跨报告长期经验过去分散在 4 套异构格式里。本文件定义一个**统一、可扩展**的经验表示，并说明如何新增一条经验。代码侧的载体是 `src/experience.py` 的 `Experience` dataclass；记忆引擎在 `src/memory_engine.py` 暴露 `match_experiences` / `get_structural_experiences` / `experience_index`。

> **向后兼容**：统一层只是在现有 `nodes.json` / `rules.json` / `cases.json` / `transition_matrix_*` 之上**加一层视图**（适配器只读、不改盘上数据）。现有匹配方法（`get_node_memory` / `get_edge_memory` / `get_explicit_rules` / `get_implicit_rules`）保持原样，所有旧调用点不受影响。

## Experience 字段

| 字段 | 含义 |
|---|---|
| `id` | 唯一标识 |
| `scope` | `node` / `edge` / `structural` |
| `polarity` | `neg`/`wrong`（该删）·`pos`/`missing`（该补）·`review`（人工核对）·`find`（漏抽）·`correct`（标注纠偏）·`flag`（提示） |
| `trigger` | 触发条件，见下 |
| `action` | `add_edge`/`remove_edge`/`reverse_edge`/`reconnect`/`reconnect_or_drop`/`reconnect_or_remap`/`drop_node`/`insert_node`/`remap`/`flag` |
| `hint` | 给 LLM/人看的自然语言经验（核心内容） |
| `confidence` | [0,1]，先验置信度（学习经验会随确认次数更新） |
| `provenance` | `{source: seed\|learned, doc_id, count, w_pos, w_neg, ts, corrector_stage}` 出处/审计 |
| `tier` | 仅 edge 规则：`explicit_rule`/`implicit_rule` |
| `hit` | 运行期标注：命中的具体技术对/边/节点（不写盘） |

### `trigger` 子字段
- **node / edge 经验**：`techniques_src`、`techniques_dst`（ATT&CK 技术或父技术列表）、`keywords_src`、`keywords_dst`（证据文本关键词）、`kw_mode`（`role_only`/`both`/`either`）、`exclude_kw`。
- **structural 经验**：`kinds`（该经验针对的结构异常种类，见下）、`structural_condition`（人读的阈值说明；真正阈值由 `graph_structure.py` 检测器 + `configs/default.yaml` 掌握）。

## 三种 scope 与对应的旧格式

| scope | 适配器 | 旧种子文件 | 作用 |
|---|---|---|---|
| `node` | `Experience.from_node_memory` | `nodes.json` 的 `memory[]` | 节点技术界定 / 漏抽提示 |
| `edge` | `Experience.from_rule` / `from_case` | `rules.json` / `cases.json` | 边去伪 / 补漏 / 人工核对 |
| `structural` | `Experience.from_structural` | `structural.json`（本次新增） | 图级结构异常的修复经验 |

## 结构异常种类 (`STRUCTURAL_KINDS`)

由 `src/graph_structure.py` 的 `analyze_structure` 产出，structural 经验用 `trigger.kinds` 认领：

- `isolated` —— 孤立节点（入度=出度=0）
- `illegitimate_root` —— 非法起点（入度=0 但战术阶段过晚，如防御规避/C2/外传）
- `disconnected_component` —— 图分裂成多个弱连通分量

## 如何新增一条经验

1. **新增一类边去伪/补漏经验**：在 `rules.json`（规则）或 `cases.json`（案例）追加一条五元组记录（`src/dst/src_kw/dst_kw/kw_mode/polarity/hint`）。无需改代码。
2. **新增一类节点界定/漏抽经验**：在 `nodes.json` 的 `memory[]` 追加 `{tech, when_kw, hint, find_node?}`。无需改代码。
3. **新增一类结构修复经验**：在 `structural.json` 的 `experiences[]` 追加 `{id, scope:"structural", trigger:{kinds:[...]}, action, hint}`。若引入**新的结构异常种类**，则需同时在 `graph_structure.analyze_structure` 里检测该 `kind` 并加入 `STRUCTURAL_KINDS`。
4. **学习型经验**（运行中由验证/裁决沉淀）：以 `provenance.source="learned"` 写入，受 Memory Gate（置信度/次数门限）约束后晋升为长期经验；写入前按 `Experience.dedup_key()` 去重。

## 检索与索引

- `match_experiences(nodes, edges, scope=None)` —— 把当前图命中的所有经验统一成 `Experience` 列表。
- `get_structural_experiences(nodes, edges, structural_report)` —— 按检测到的异常 `kind` 取相关结构经验，注入 Agent C 结构修复 prompt。
- `experience_index()` —— 每条经验一行的可扫描索引（类比 harness 的 `MEMORY.md`），运行后落盘到 `outputs/_mem/experience_index.json`。
