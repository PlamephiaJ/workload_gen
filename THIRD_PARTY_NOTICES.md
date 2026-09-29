# Reference implementation notes

The GOAL binary layout in `workload_gen/goal_io.py` follows the serialization
layout used by `atlahs/sim/LogGOPSim/Parser.hpp` (`MAGIC_COOKIE`, jump table,
packed node records, and dependency appendix).

The collective traffic patterns in `workload_gen/compiler.py` were implemented
from the algorithms in
`atlahs/goal_gen/ai/nccl_generator_v2/nccl_comm.py`: direct All-to-All, ring
AllReduce (reduce-scatter plus allgather), and tree AllReduce (upward reduction
plus downward broadcast).  No Atlas source file is imported at runtime and no
reference source file is modified.  `tools/compare_atlas_lowering.py` imports
the sibling Atlas checkout only as an optional validation oracle.

