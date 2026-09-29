# HTSim workload generator

这个目录提供一个独立的、声明式的 GOAL workload 生成器。输入可以是：

- collective 级 DAG：当前支持 direct All-to-All、ring AllReduce、tree AllReduce；先 lowering 成成对的 p2p send/recv，再写 GOAL binary，并输出完整 flow metadata。
- p2p flow 级 DAG：直接指定 flow、compute 与依赖关系，直接写 GOAL binary；不生成 collective metadata。

生成器不依赖 Atlas 或第三方 Python 包。它直接实现 Atlas/LogGOPSim 使用的 GOAL binary 格式，也保留一份可读的 `workload.goal` 便于检查。

## 快速开始

在本目录执行：

```bash
python3 generate.py templates/collective_dag.json
python3 generate.py templates/flow_dag.json
```

也可以使用 uv；项目没有外部运行时依赖：

```bash
uv run generate.py templates/collective_dag.json
```

默认输出为：

```text
outputs/
├── collective_three_branches/
│   ├── workload.bin
│   ├── workload.goal
│   └── workload.meta.json
└── flow_three_branches/
    ├── workload.bin
    └── workload.goal
```

指定输出根目录：

```bash
python3 generate.py templates/flow_dag.json --output-root /path/to/outputs
```

验证本地 writer 与 Atlas `txt2bin` 是否逐字节一致：

```bash
python3 generate.py templates/collective_dag.json --verify-with-txt2bin
```

不保留文本 GOAL：

```bash
python3 generate.py templates/flow_dag.json --no-goal-text
```

## 输入模型

两个模板都使用 JSON，以保证工具零依赖。共同字段：

- `schema_version`: 当前固定为 `1`。
- `kind`: `collective` 或 `flow`。
- `name`: 输出子目录名。
- `world_size`: GOAL rank 数。
- `defaults`: 可选的 `cpu`、`nic`、`chunk_bytes` 等默认值。
- `nodes`: 高层 DAG 节点；`depends_on` 指定前驱。

GOAL 的依赖是 rank-local 的。因此一条高层依赖边的两端必须至少共享一个 rank；编译器会拒绝无法投影到任何 rank 的边。

### Flow DAG

`flow` 节点字段：

- `src`, `dst`, `bytes`: 源、目的和字节数。
- `tag`: 可选；省略时自动生成。显式 tag 加分块时，第 `i` 块使用 `tag + i`。
- `chunk_bytes`: 可选；flow 大于该值时拆为多个有序消息。
- `cpu`, `nic`: 可选，范围 `0..255`。

`compute`、`sink` 和 `barrier` 都是本地计算节点，使用 `ranks` 与 `duration_ns`。`ranks` 可以是 `"all"` 或 rank 数组。`duration_ns` 可为一个整数，也可为按 rank 给值的对象。

### Collective DAG

All-to-All 使用：

```json
{
  "op": "alltoall",
  "algorithm": "direct",
  "ranks": "all",
  "bytes_per_peer": 1048576,
  "chunk_bytes": 262144
}
```

Ring AllReduce 使用总 payload `bytes`，可指定 `channels` 和每个 channel 的 `ring_orders`。每个 channel 执行 Atlas/NCCL 风格的 reduce-scatter 与 allgather：

```json
{
  "op": "allreduce",
  "algorithm": "ring",
  "ranks": [0, 1, 2, 3],
  "bytes": 4194304,
  "channels": 1,
  "ring_orders": [[0, 1, 2, 3]]
}
```

Tree AllReduce 默认按 `ranks` 顺序构造平衡二叉树，也可显式指定每个 rank 的 parent：

```json
{
  "op": "allreduce",
  "algorithm": "tree",
  "ranks": [0, 1, 2, 3],
  "bytes": 2097152,
  "tree_parent": {"0": null, "1": 0, "2": 0, "3": 1}
}
```

Collective 可设置 `collective_id`、`context` 和 `ordering_domain`。自动 tag 遵循 Atlas 的编码习惯：

```text
tag = per_(src,dst,context)_message_sequence * 100000 + context
```

`workload.meta.json` 的每个 flow 保存 `collective_id`、collective type、algorithm、phase、chunk、step、channel、ordering domain、src/dst、bytes、tag/message sequence，以及 send/recv 在各 rank 中的 task ID。

`reduction_ns_per_byte`、`reduction_overhead_ns`、`copy_ns_per_byte`、`copy_overhead_ns` 可用于给 lowering 中的本地 reduction/copy 建模；默认数据率和 overhead 都是 0。模板展示了固定 overhead。

## 验证

运行全部测试：

```bash
python3 -m unittest discover -s tests -v
```

测试包括：

- binary round-trip、依赖计数、DAG 无环性与 send/recv 完整匹配；
- 原生 Python writer 与 Atlas `txt2bin` 的逐字节比较；
- direct All-to-All、ring AllReduce、tree AllReduce 的 p2p transfer multiset 与 `atlahs/goal_gen/ai/nccl_generator_v2` 参考实现比较。

也可单独运行 lowering 对照：

```bash
python3 tools/compare_atlas_lowering.py --atlas-root ../atlahs
```

## HTSim

产出的 `workload.bin` 可直接作为现有 HTSim 命令的 `-goal` 参数：

```bash
../HTSIM/htsim/sim/datacenter/htsim_uec \
  -topo <topology-file> \
  -goal outputs/collective_three_branches/workload.bin \
  -nodes 4 <other-htsim-options>
```

`world_size`/`-nodes` 和 topology 可容纳的 host 数需要彼此一致。

