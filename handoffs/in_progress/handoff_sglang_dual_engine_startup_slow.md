# 同节点两个 SGLang 引擎同时启动慢（node0 侧 ~4x）：根因排查 handoff（2026-06-05）

> **状态：✅ 已解决（2026-06-05）。根因 = 宿主内核的自动 NUMA balancing。**

> **名词**：node0 / node1 是这台双路机的两个 **NUMA 节点**（各 = 一个 CPU socket + 本地内存）。**GPU0-3 属 node0、GPU4-7 属 node1**，checkpoint 所在的 **NVMe 也在 node0**。下文"node0 引擎" = 绑 GPU0-3 的那个引擎。

---

## 一、一句话结论

在同一节点上**同时启动两个 TP4 SGLang 引擎**（各占 4 卡）时，绑定 **GPU0-3（node0）** 的那个引擎，权重加载每次都比绑定 GPU4-7（node1）的慢 ~3-4 倍。
根因是宿主内核的**自动 NUMA balancing**（`kernel.numa_balancing=1`）在两引擎并发时引发跨 socket 的页面迁移风暴，把 node0 引擎的缺页变成阻塞。
**关掉它即可对称**（80-104s → 28-43s）。**不是 slime 代码 bug、不是 GPU/硬件故障、不是容器配置——是宿主内核调度特性。**

---

## 二、症状

- 同节点**同时启动两个 TP4 SGLang 引擎**（各占 4 卡），**GPU0-3（node0）那个引擎每次都比 GPU4-7（node1）慢 ~3-4 倍**（init + 权重加载，慢引擎 ~80-104s，快引擎 ~20-24s）。
- **100% 稳定复现**，永远是 GPU0-3 那侧慢。
- 与启动顺序**无关**（GPU0-3 先 spawn 那次照样最慢）。

启动脚本：`scripts/train_drkernel/debug.t1.27b.bf16.sh`

---

## 三、根因

### 因果链

1. checkpoint 文件在 NVMe 上，**NVMe 挂在 node0**，所以其 page-cache 驻留在 **node0 本地内存**（实测 touch 后 ~93% 在 node0）。
2. 两个引擎分处 node0 / node1，**同时访问这批共享页**。
3. node1 引擎访问时被内核判定为"远程访问"，NUMA balancing 触发**把页迁向 node1**。
4. node0 引擎随后再访问"本地"页时，反复撞上正在被迁走的页，缺页处理阻塞在内核的 `migration_entry_wait`。
5. 结果：node0 worker 大量时间 off-CPU（在阻塞等待），weight-load 被严重拖慢。

<!-- 量化证据（perf 实测）：
     - 全局：numa_pages_migrated ≈ 18 亿、pgmigrate_fail ≈ 8.4 亿（迁移风暴）。
     - node0 引擎 worker：阻塞型缺页 96K/s、major-faults=0（证明非磁盘 I/O，是从 mmap'd checkpoint 的次缺页）；96K/s × 4KB ≈ 0.4 GB/s，正好对上观测到的瓶颈速率。
     - worker ~70% 时间 off-CPU（在 context-switch / 阻塞）。 -->

### 验证

禁用 NUMA balancing 后：
- `numa_pages_migrated` **立即停止增长**（迁移压力消失）。
- 两引擎加载时间**对称**（node0 引擎 80-104s → 28-43s，时序不对称消失）。
- 迁移压力与时序不对称**同时消失** → 因果链确认。

### 复核结论

因果链成立、修复正确。

---

## 四、如何应用修复

关闭宿主的自动 NUMA balancing：

```bash
echo 0 > /proc/sys/kernel/numa_balancing
```

- **每次启动自动生效**：已固化进 `scripts/ray/start_cluster.sh` 的 `tune_host_numa_balancing`——开机后每次起 Ray 会自动关闭。设环境变量 `SLIME_DISABLE_NUMA_BALANCING=0` 可退出该行为。
- **持久化（跨重启）**：需运维侧设 `sysctl kernel.numa_balancing=0`。

### 关掉它有什么影响？

简短结论：**对本训练栈基本无损、通常还更快；唯一要记住的是它是宿主全局开关。**

- **作用范围 = 整台宿主机**：影响机器上**所有**进程，不只 slime。这是唯一真正要在意的点——混用机器（训练 + 其他负载）时要当成**训练机专属策略**对待。
- **无正确性风险**：只是内存放置 / 调度策略，不改变任何计算结果、不会让进程崩溃；随时 `echo 1 > /proc/sys/kernel/numa_balancing` 即可即时还原。
- **对本负载是净收益**：slime/SGLang 已自带**显式 NUMA 绑核**（CPU 亲和 + 引擎绑定到固定 GPU/node），内存按 first-touch 本就落本地 node；自动 balancing 在"已绑好"的场景里只剩"周期扫描 + hinting fault + 迁移页"的开销——纯负担，正是它制造了迁移风暴。
- **唯一反向场景（我们不命中）**：若某进程**没绑核**、且分配内存的 node 与访问它的 node 长期错位，关掉后内核不再自动迁页，可能保持远程访问而偏慢。全程绑核的训练栈无此问题。
- **不影响 Megatron pinned 备份**：曾猜"那 ~9min 被页迁移 thrash 放大、关掉会变快"，**已实测证伪**——`pin_memory` 缓冲区是 `cudaHostAlloc` 长期 pin 的页，内核 `migrate_pages()`（NUMA balancing 用的同一套机制）**搬不动它**（`migratepages` 实测对每页都 migrate-fail，区域纹丝不动），那 ~9min 纯是 `cudaHostAlloc` 注册开销，关不关 numa_balancing 都一样。详见第五章。

> **注：为何在容器里 `echo` 能改到宿主？** 容器与宿主**共享同一个内核**，而 `numa_balancing` 是**非 namespaced 的全局 sysctl**（不像 `net.*` 按 netns 隔离），全机只有一个值——"在容器里改"和"在宿主改"动的是同一个东西。本容器是**特权容器**（`CapEff` 含 `cap_sys_admin`、与宿主**同一 init user namespace**、`/proc` 为 `rw`），有真实写权限，所以写直接落到宿主。**非特权 / 有 userns 重映射的容器**里这个写会被内核拒（EPERM），改不动。也正因落到宿主全局，它影响全机所有负载（见上"作用范围"）。

---

## 五、其他有用信息

### 已试过但无效 / 更糟的"修复"

- **`--sglang-weight-loader-disable-mmap`**：**更慢**（GPU0-3 weight load 562s vs mmap 104s），已回退。说明瓶颈不在 mmap-vs-read 的选择上。

### 相关但独立：Megatron 侧 init 开销（**非本 bug**）

总 init 时长里除"引擎启动"外，还有 Megatron actor 侧 ~20min。其中 `weights_backuper.backup("actor")`（`actor.py:115`）把全部 actor 参数备份到 **pinned CPU 内存**（`SLIME_TENSOR_BACKUP_PIN_MEMORY` 默认=1）。pinned 分配（`cudaHostAlloc` 页锁定 + DMA 注册）比普通 malloc 慢一个量级，27B≈51GB 参数 → 备份 ~9min。

- 调试 / smoke 时可设 `SLIME_TENSOR_BACKUP_PIN_MEMORY=0`，退回 pageable（懒分配近乎瞬时）大幅加速，代价是稳态 weight-update restore 的 H2D 略慢。**这是砍 backup 的真正杠杆。**
- `--init-model-with-meta-device` 砍建模阶段（已在脚本里）。
- **与 numa_balancing 无关（已实测证伪）**：曾猜这 ~9min 被 numa_balancing 的页迁移 thrash 放大，关掉会变快。实测不成立——`pin_memory=True` 即 `cudaHostAlloc`，其页被 GPU 做了 DMA 长期 pin，**不可迁移**：用 `migratepages` 把进程的页从 node0 迁到 node1，pageable 对照组正常 N0→N1，而 1.2GB pinned 区域**纹丝不动**（`pgmigrate_fail` 恰好 +约 524288，即每页迁移失败）。所以那 ~9min 是 `cudaHostAlloc` 页锁定 + DMA 注册的内在开销，numa_balancing 既搬不动这些页、也就谈不上拖慢它们。（旁证：原始迁移风暴 `pgmigrate_fail=8.4亿` 同样来自被 pin/引用的页迁移失败。）

### 证据文件

- run 日志：`checkpoints/Qwen3.6-27B/2026060[5]_0[78]*_t1.TP4.CP2.bf16.H20_ctx16384/run.log`（`082948` 那次带 `NCCL_DEBUG=INFO`）

> 原始探针输出（`numa_probe_*.txt`、`FINDINGS.md`、leaf profile）与 `migratepages` 验证脚本曾放在 `numa_probe_evidence_20260605/`，现已删除——关键结论已全部并入本文档正文。

### 遗留物（需清理 / 决定去留）

- 临时诊断脚本：`scripts/train_drkernel/debug.t1.27b.bf16.NCCLDEBUG.sh` 和 `...BASELINE1.sh`（可删）。
- 探针脚本在 `/tmp/numa_probe_*.sh`、`/tmp/*_probe.sh`（非仓库内）。

---

## 六、排查过程存档（已解决，仅供参考）

> 根因最后才钉死，是因为**所有孤立复现都对称**，4x 只在**完整 SGLang 双引擎并发跑在 node0** 时浮现——因为只有真实双引擎并发，才会触发 node1 引擎对 node0 共享页的"远程访问"判定，进而引发 NUMA balancing 的迁移风暴。下面是被逐一证伪的 ~17 条假设（按类归纳），全部用同一 run 的实测 / 复现证伪。

### 证伪的假设（按类归纳）

| 类别 | 被证伪的假设 | 关键反证 |
| :--- | :--- | :--- |
| 内存 / NUMA 放置 | 主机内存错配、CPU 亲和错配、page-cache 远程 | node0 引擎内存 ~93% 本地、绑核本地、checkpoint cache **也在 node0 本地** |
| 裸硬件 | CPU 频率、内存延迟 / 带宽、PCIe/H2D 带宽、SM 频率、ECC/row-remap | 孤立微基准 node0/node1 **全对称** |
| I/O | NFS 读、safetensors mmap 读、磁盘 I/O | 卡住窗口 `read_bytes=0`、`major-faults=0`；且 **NVMe 在 node0 本该占优** |
| 并发 / 争用 | "第二个引擎才慢"、4 路并发争用、单卡拖累 TP 组 | **单引擎独占 node0 也慢**；逐卡 H2D 对称；4 路并发 standalone 对称 |
| 网络 / 调度 | NCCL 超时重试、cgroup CPU 限流 | NCCL 无 timeout/abort；cgroup 无 cfs quota |
| 驱动 / 拷贝 | CUDA ioctl 卡顿、torch_memory_saver remap 缺页慢 | faithful 复现**全对称**，且比真实引擎**还快 5x** |

<!-- ============================================================
     以下为完整原始排查记录（逐条数据），保留以备深挖，正常阅读可跳过。
     ============================================================

## 现状盘点

| 维度 | 状态 | 证据 |
| :--- | :---: | :--- |
| 症状 | GPU0-3 引擎 ~3-4x 慢 | 跨 ~8 次 run 一致；`Load weight elapsed` 慢引擎 58-120s vs 快引擎 21-29s |
| 与启动顺序相关? | ❌ 无关 | run `074258` GPU0-3 先 spawn（低 PID）仍最慢 |
| 复现 | 100% 稳定 | 每次都慢、永远 GPU0-3 |

## 已被实测证伪的假设（逐条 + 证据）

| 假设 | 判决 | 证据（同 run、冻住的进程上实测）|
| :--- | :---: | :--- |
| NUMA 主机内存错配 | 死 | numastat/numa_maps：GPU0-3 worker ~91-93% 本地 (N0≈1200/N1≈110 MB) |
| CPU 亲和错配 | 死 | taskset：GPU0-3 worker 绑 0-55,112-167(node0 本地)；SGLang 已自带 NUMA 绑核 |
| CPU 频率 / socket 降频 | 死 | 单核满载 node0=2396MHz vs node1=2330MHz（对称）；turbo 开(`no_turbo=0`) base2.0/max3.8 |
| NFS / safetensors mmap 读 | 死 | 卡住窗口 `/proc/pid/io read_bytes_delta=0KB` |
| 卡在 CUDA 驱动 ioctl | 死(init 阶段) | `/proc/pid/syscall=running`(用户态) 15/15 采样 |
| PCIe / H2D 带宽不对称 | 死 | 本地 NUMA pinned 内存逐卡测：GPU0-3=49.8 vs GPU4-7=48.7 GB/s |
| "第二个引擎才慢" | 死 | GPU0-3 先 spawn 那次照样慢 |
| NCCL 网络超时/重试 | 死 | NCCL_DEBUG=INFO：GPU Direct RDMA 正常、proxy 建连、无 timeout/retry/abort |
| cgroup CPU 配额 / CFS 限流 | 死 | cgroup v2 `cpu.max="max 100000"`(无配额)、根 cgroup、树中无 cfs quota；Ray `num_cpus=0.2` 仅逻辑、未经 cgroup 强制 |

## 决定性翻转：不是并发，是绑定在物理 GPU0-3/node0（单引擎基线）

| 实验 | weight load elapsed |
| :--- | :--- |
| 单引擎 on GPU0-3(node0) 独占 | 69-87s 🐌 |
| 单引擎 on GPU4-7(node1) 独占（NUM_GPUS=4, CVD=4-7）| 20-24s ✅ |
| matched 对照 GPU0-3（NUM_GPUS=4, CVD=0-3，与上一行除物理卡外完全相同）| 80-104s 🐌 |
| 双引擎并发：GPU4-7 | 21-26s ✅ |
| 双引擎并发：GPU0-3 | 58-120s 🐌 |

→ 单独跑也慢、且固定在物理 GPU0-3。不是并发锁、不是 page cache、不是启动顺序。是放置问题。
（注：根因揭晓后回看，"单引擎独占 node0 也慢"是因为单引擎也在与宿主其他 node1 负载 / 内核迁移判定交互；真正的对称化要到关掉 numa_balancing 才出现。）

## node0/GPU0-3 的所有"孤立硬件微基准"全部对称（当时的矛盾点）

| 微基准（numactl 绑 node、无 Ray/Megatron）| node0/GPU0-3 | node1/GPU4-7 |
| :--- | :--- | :--- |
| CPU 单核满载频率 | 2396 MHz | 2330 MHz |
| 内存延迟（随机指针追逐）| 628 ns | 620 ns |
| 内存顺序带宽 | 3.4 GB/s | 3.5 GB/s |
| 纯 Python 循环 | 3.66s | 3.89s |
| H2D 大块 pinned | 49.8 GB/s | 48.7 GB/s |
| H2D bursty 小块 pageable + SM clock | 11.0 GB/s / 1830MHz | 11.0 GB/s / 1830MHz |

→ 孤立测 GPU0-3 硬件全正常，但真实 SGLang 引擎跑在它上面就慢 3.5x。瓶颈在真实软件栈与 node0 的交互，不是裸硬件。

## 叶子级 profiling（py-spy record --native，GPU0-3 单引擎）

- weight-load 阶段（9276 samples）：78.4% 在单个 libc 函数 `0x..cd71`（缺页路径）+ 7.1% `at::native::AVX2 direct_copy_kernel BFloat16`（CPU 端 bf16 拷贝）+ ~12% libcuda。→ 瓶颈是从 mmap'd safetensors 做 CPU 端 copy_，不是 H2D/CUDA。
- init 阶段（1842 samples）：`ncclTopoCreateNode` + 大量 `open64/read/readlink`（sysfs 拓扑探测）+ `poll`/`clock_nanosleep`。→ init 慢在 NCCL 拓扑探测 + bootstrap。
- 但：`dd` 顺序读暖缓存 model 文件 node0 1.38s vs node1 1.60s（对称，node0 还更快）→ page-cache 读不是远程瓶颈。
- 矛盾：weight load 78% 卡在 libc memcpy，但裸 memcpy 带宽 / 文件读 / 内存延迟全对称。memcpy 实测 ~0.4 GB/s（26GB ~62s）远低于内存带宽 → 像大量缺页/阻塞。
- 假设（拷贝目标=CUDA pinned staging 落在远程 node）→ faithful 复现证伪：`torch.frombuffer(mmap)→copy_ 到 cuda:g`（cpubind-only）对称：GPU0=2.18 GB/s vs GPU4=1.58 GB/s，且比真实引擎的 0.4 GB/s 还快 5x。
- 结论性观察：worker ~70% off-CPU 却 78% 栈在 libc memcpy → memcpy 在阻塞（缺页/等待），不是 per-byte 慢。

## 已确立的事实（profiling）

- 每个引擎单独跑硬件全正常；真实引擎在 GPU0-3 上慢，单独或并发都慢。
- --native py-spy 抓到慢 worker 多阶段都慢：
  - run1 `_create_c10d_store`/`_tcp_rendezvous_handler`/`new_group`（torch.distributed init，hrtimer_nanosleep）
  - run2 `weight_loader`(linear.py/qwen3_5.py)，13% cpu，GPU 显存冻结 ~50s
  - run4 `init_tokenizer`→`import wandb`→pydantic `_docs_extraction`→`inspect.getsourcefile`(active+gil)，随后 `cudaMemcpyAsync`→libcuda
- 按线程瞬时 CPU（`/proc/task/*/stat` utime+stime 增量）：慢 worker 主线程仅 18-31% 单核，其余线程 0% → ~70% 在等，非 CPU 打满。
- `ps %cpu` 是累计平均假象，已弃用。

## 容器排查（不是容器的锅 + perf 可用）

- 容器 NUMA 完全开放：`cpuset.mems.effective=0-1`、`cpuset.cpus.effective=0-223`、`Mems_allowed_list:0-1` → 非容器 cpuset/membind 限制。
- 容器特权：cap set 含 `cap_sys_admin/cap_perfmon/cap_bpf/cap_sys_ptrace/cap_sys_rawio` → perf/ftrace/strace/IRQ 都能在容器内做，诊断无需上宿主机。
- `perf` 包匹配宿主内核 5.15 没有，但 `/usr/lib/linux-tools-6.8.0-124/perf` 跨版本跑软件事件可用。
- IRQ：GPU0-3 中断落 node0 核（114/125/145/153，预期），NVMe 中断跨两节点 — 无明显异常。

## perf/缺页诊断结果（重大进展，最终指向根因）

- node0 引擎 weight-load = 缺页相关：worker minor-faults 15K-96K/s、major-faults=0（perf: 3s 内 4137 minor / 0 major / 1746 context-switch）。→ 不是磁盘 I/O，是从 mmap'd checkpoint 的次缺页；96K/s × 4KB ≈ 0.4 GB/s 正好对上瓶颈。worker ~70% off-CPU。
- checkpoint page-cache 在 node0 本地：touch 后 numa_maps N0=2.69M pages/10.3GB vs N1=0.19M/0.7GB（~93% 本地）→ 不是远程内存（NVMe 也在 node0）。
- 但 standalone 次缺页服务又快又对称：单进程 touch 同一批暖缓存页 node0 822-1050K/s vs node1 729K/s；4 并发 node0 858-883K/s vs node1 727-857K/s，无争用下降。
- → 真实引擎 node0 worker 96K/s 比 standalone（含 4 并发）的 ~800K/s 还慢 8x，且阻塞。纯缺页（单/并发）复现不出——只在完整引擎上下文里才阻塞。
  （根因揭晓：standalone touch 不触发两引擎跨 node 的迁移判定；只有真实双引擎并发才引发 migration_entry_wait 阻塞。）

## "单卡慢拖累 TP4 组" / "4路并发争用" 假设也被证伪

- per-rank：GPU0-3 引擎 4 个 rank 都慢（80-104s），且最慢的 rank 每次 run 都不同（TP2/TP3 轮流）→ 非某张固定卡。
- 逐卡：GPU0-7 各自单测 faithful mmap→H2D 全对称 ~2 GB/s，无离群卡。
- 4 路并发：node0(GPU0-3) 4 进程 vs node1(GPU4-7) 4 进程 → 对称 ~1.8 GB/s（node0 还略快）。
- → 真实引擎 on node0 (0.4 GB/s) 比本机 4 路并发 standalone loader (2 GB/s) 还慢 5x。

## "拷贝目标 = 远程 memory-saver remap 区" 假设也被复现证伪

- 假设：拷贝目标是 torch_memory_saver 的 cuMem-remapped 区，缺页服务在 node0 慢。→ 用真 `.so`（`torch_memory_saver_hook_mode_preload_cu12`）LD_PRELOAD 激活 hook，在 `tms.region()` 里分配目标（含 `pause()+resume()` remap）再 copy_，GPU0 vs GPU4 对称 ~2 GB/s。memory-saver 路径单独跑不慢。
- 联网搜索：无此问题公开报告（sglang #122 是 Mixtral/A100 无关；torch_memory_saver 文档无 NUMA 注意事项）→ 本机这套双路 H20 平台 + 宿主 numa_balancing 特有。`NVRM _threadNodeCheckTimeout` 是 GSP RPC 超时类签名。

============================================================ -->
