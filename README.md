# HTSim workload generator

这个目录提供一个独立的、声明式的 GOAL workload 生成器。输入可以是：

- collective 级 DAG：可在 collective 之间混合 compute/collective_delay/sink/barrier 节点，并覆盖 Atlas 当前实现的全部 collective/operator/algorithm 组合；collective 先 lowering 成成对的 p2p send/recv，再与本地节点一起写入 GOAL binary，并输出完整 flow metadata 和等价 flow DAG template。
- p2p flow 级 DAG：直接指定 flow、compute、flow_delay 与依赖关系，再写 GOAL binary。`flow_delay` 只按唯一 `flow_id` 选择目标。

生成器不依赖 Atlas 或第三方 Python 包。它直接实现 Atlas/LogGOPSim 使用的 GOAL binary 格式，也保留一份可读的 `workload.goal` 便于检查。

## 快速开始

在本目录执行：

```bash
python3 generate.py templates/collective_dag.json
python3 generate.py templates/collective_all_supported.json
python3 generate.py templates/flow_dag.json
python3 generate.py templates/delay_dag.json
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
│   ├── workload.meta.json
│   └── collective_three_branches.flow.json
├── flow_three_branches/
│   ├── workload.bin
│   ├── workload.goal
│   └── workload.meta.json
└── collective_with_delay/
    ├── workload.bin
    ├── workload.goal
    └── workload.meta.json
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

模板都使用 JSON，以保证工具零依赖。共同字段：

- `schema_version`: 当前固定为 `1`。
- `kind`: `collective` 或 `flow`。
- `name`: 输出子目录名。
- `world_size`: GOAL rank 数。
- `defaults`: 可选的 `cpu`、`nic`、`chunk_bytes` 等默认值。
- `nodes`: 高层 DAG 节点；`depends_on` 指定前驱。
- `extends`: 可选的相对路径；继承基础模板并把当前 `nodes` 追加到基础 DAG。用于从同一 flow baseline 派生单一 perturbation template。

GOAL 的依赖是 rank-local 的。因此一条高层依赖边的两端必须至少共享一个 rank；编译器会拒绝无法投影到任何 rank 的边。

### Flow DAG

`flow` 节点字段：

- `src`, `dst`, `bytes`: 源、目的和字节数。
- `flow_id`: 可选的显式唯一 ID；`flow_delay` 必须通过该 ID 引用目标 flow。
- `tag`: 可选；省略时自动生成。显式 tag 加分块时，第 `i` 块使用 `tag + i`。
- `chunk_bytes`: 可选；flow 大于该值时拆为多个有序消息。
- `cpu`, `nic`: 可选，范围 `0..255`。

`compute`、`sink` 和 `barrier` 是普通本地计算节点，使用 `ranks` 与 `duration_ns`。

`flow_delay` 是 flow DAG 专用节点，不允许出现在 collective DAG 中。它直接按
`flow_id` 定位目标，将 destination rank 上原本从 recv 指向后继的依赖改为
`recv -> flow_delay -> successors`：

```json
{
  "id": "flow_35_delay_2us",
  "op": "flow_delay",
  "flow_id": 35,
  "duration_ns": 2000
}
```

### Collective DAG

Collective DAG 不只能包含通信算子，也支持 `compute`、`collective_delay`、`sink` 和 `barrier` 本地节点。这四种节点的字段相同：

- `ranks`: 执行该节点的 GOAL rank，可为 `"all"` 或非空 rank 数组，例如 `[0, 2]`。
- `duration_ns`: 本地计算时间，可为所有 rank 共用的非负整数，也可为按 rank 设置的对象，例如 `{"0": 1000, "1": 1500}`。
- `cpu`: 可选 GOAL CPU ID；省略时使用 `defaults.cpu` 或 `0`。
- `depends_on`: 前驱节点 ID 数组。依赖按共有 rank 投影：每个本地节点会等待同一 rank 上的前驱 collective lowering 完成。

`compute`、`collective_delay`、`sink` 和 `barrier` 都会生成 GOAL `calc` 任务，不会 lowering 成网络 flow。`sink` 通常使用 `duration_ns: 0` 来汇合多个分支。`barrier` 是用于表达依赖的 rank-local anchor；它不会隐式增加跨 rank 同步，所有前驱都需要在 `depends_on` 中显式列出。

#### Delay 节点

`collective_delay` 专门表达 collective DAG 节点之间的等待、软件开销或其他不产生通信的中间延迟。它与 `compute` 使用相同的 GOAL `calc` 表示，但不能按 collective 内部的 src/dst 或 flow 定位。

```json
{
  "id": "between_stages",
  "op": "collective_delay",
  "ranks": "all",
  "duration_ns": 5000,
  "depends_on": ["allreduce"]
}
```

`duration_ns` 也可以按 rank 指定，例如 `{"0": 5000, "1": 6500}`。`collective_delay` 只等待与自己共享 rank 的前驱；如果一条边两端没有任何共同 rank，编译器会报错。

完整的 `collective -> collective_delay -> collective -> sink` 可运行示例见
`templates/delay_dag.json`。需要对内部通信注入 `flow_delay` 时，必须先使用 lowering
输出的 flow template，再从该 flow template 派生 perturbation；两种 delay 不允许跨
DAG 粒度混用。

### Collective → flow template

输入为 collective DAG 时，每次正常生成都会额外在输出目录写入同逻辑的
`<name>.flow.json`。也可以用 `--flow-template-output` 把它写到指定模板路径：

```bash
python3 generate.py templates/exp2/exp2.a.1.collective.json \
  --flow-template-output templates/exp2/exp2.a.1.flow.json
```

collective lowering 为生成的 P2P flow 分配 `flow_id`，导出的 flow DAG 直接使用这些
ID，不进行第二次编号；同时用 endpoint-specific `rank_depends_on` 精确保留 lowering
后 send、recv 和本地 calc 的依赖。推荐文件名统一使用
`<experiment>.collective.json` 和 `<experiment>.flow.json` 表示 DAG 粒度。

一个 `collective -> compute -> sink` 示例：

```json
{
  "schema_version": 1,
  "kind": "collective",
  "name": "allreduce_with_compute",
  "world_size": 4,
  "nodes": [
    {
      "id": "allreduce",
      "op": "allreduce",
      "algorithm": "ring",
      "protocol": "simple",
      "ranks": "all",
      "bytes": 4194304
    },
    {
      "id": "post_compute",
      "op": "compute",
      "ranks": "all",
      "duration_ns": 2500,
      "depends_on": ["allreduce"]
    },
    {
      "id": "sink",
      "op": "sink",
      "ranks": "all",
      "duration_ns": 0,
      "depends_on": ["post_compute"]
    }
  ]
}
```

多分支写法可直接参考 `templates/collective_dag.json`，其结构为：

```text
AllToAll       -> compute --+
AllReduce Ring -> compute --+-> sink
AllReduce Tree -> compute --+
```

支持矩阵与 Atlas `nccl_generator_v2/nccl_comm.py` 保持一致：

| Operator | Algorithm | Protocol |
|---|---|---|
| AllToAll | direct | Simple（Atlas 在该类中固定调用 Simple） |
| AllReduce | ring | Simple / LL / LL128 |
| AllReduce | tree | Simple / LL / LL128 |
| AllGather | ring | Simple / LL / LL128 |
| ReduceScatter | ring | Simple / LL / LL128 |
| Reduce | ring | Simple / LL / LL128 |
| Broadcast | ring | Simple / LL / LL128 |

Atlas 没有实现 tree AllGather、tree ReduceScatter、tree Reduce 或 tree Broadcast；输入这些组合会得到明确错误，而不是静默换算法。`templates/collective_all_supported.json` 包含上表所有 operator/algorithm，可作为完整模板。

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

其余 ring 算子：

```json
{"op": "allgather", "algorithm": "ring", "bytes_per_rank": 1025, "in_place": true}
{"op": "reduce_scatter", "algorithm": "ring", "bytes": 4100}
{"op": "reduce", "algorithm": "ring", "bytes": 2049, "root": 0}
{"op": "broadcast", "algorithm": "ring", "bytes": 2049, "root": 0}
```

Collective 的通用可选字段：

- `protocol`: `simple`、`ll` 或 `ll128`；默认 `simple`。与 Atlas 一致，LL/LL128 的每个 primitive wire size 向上对齐到 8 bytes。
- `channels`: channel 数；默认 1。
- `ring_orders`: 每个 channel 一份 rank permutation；不指定时使用 `ranks` 顺序。
- `chunk_bytes`: AllToAll、tree AllReduce、AllGather、Reduce、Broadcast 的最大 payload chunk。
- `root`: Reduce/Broadcast 的 GOAL rank；默认 `ranks` 中第一个。
- `in_place`: AllGather 是否 in-place；out-of-place 时在初始 send 前插入 Atlas `CopySend` 对应的 copy calc。

Collective 可设置 `collective_id`、`context` 和 `ordering_domain`。自动 tag 遵循 Atlas 的编码习惯：

```text
tag = per_(src,dst,context)_message_sequence * 100000 + context
```

`workload.meta.json` 的每个 flow 保存 `collective_id`、collective type、algorithm、phase、chunk、step、channel、ordering domain、src/dst、bytes、tag/message sequence，以及 send/recv 在各 rank 中的 task ID。

协议 lowering 后还会记录 `protocol`、原始 `payload_bytes` 和实际 `wire_bytes`。

`reduction_ns_per_byte`、`reduction_overhead_ns`、`copy_ns_per_byte`、`copy_overhead_ns` 可用于给 lowering 中的本地 reduction/copy 建模；默认数据率和 overhead 都是 0。模板展示了固定 overhead。

## 验证

运行全部测试：

```bash
python3 -m unittest discover -s tests -v
```

测试包括：

- binary round-trip、依赖计数、DAG 无环性与 send/recv 完整匹配；
- `collective_delay` 与 `flow_delay` 的粒度隔离、rank-local lowering、时长和依赖；
- collective DAG 到等价 flow DAG template 的 task/edge/flow 一致性；
- 原生 Python writer 与 Atlas `txt2bin` 的逐字节比较；
- 全部 7 个 operator/algorithm 组合的 p2p transfer multiset 与 `atlahs/goal_gen/ai/nccl_generator_v2` 参考实现比较；
- 除固定 Simple 的 AllToAll 外，对每个组合分别验证 Simple、LL、LL128，共 19 个 Atlas 对照项。

Atlas 对照脚本会导入 Atlas 自己的 `nccl_primitives.py`，因此该可选测试需要 numpy/scipy；生成器本身仍然零第三方依赖。

也可单独运行 lowering 对照：

```bash
python3 tools/compare_atlas_lowering.py --atlas-root ../atlahs
```

## HTSim

产出的 `workload.bin` 可直接作为现有 HTSim 命令的 `-goal` 参数：

```bash
../HTSIM/htsim/sim/build/datacenter/htsim_uec \
  -topo <topology-file> \
  -goal outputs/collective_three_branches/workload.bin \
  -nodes 4 <other-htsim-options>
```

`world_size`/`-nodes` 和 topology 可容纳的 host 数需要彼此一致。

Dragonfly/DF 后端也可直接消费同一个 binary；以下命令已用于验证 delay 示例：

```bash
../HTSIM/htsim/sim/build/datacenter/htsim_uec_df \
  -basepath ../HTSIM/htsim/sim/datacenter/topologies/dragonfly/p2a4h2 \
  -goal outputs/collective_with_delay/workload.bin \
  -routing MINIMAL -q 88
```

