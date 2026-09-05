# Kernel Agent Full-Async 503 风暴根因与修复

## 工程结论

full-async rollout 开始 ~60s 后 router 全面返回 `503 no_available_workers`、engine 大量
`Request is disconnected from the client side (type 1)` 的根因是:**容器的 egress 代理
(clash,`http://10.11.2.164:7893`)对静默 60 秒的 HTTP 请求返回 502 并拆链,而 slime 启动的
sglang-router(Rust/reqwest)默认继承 `http_proxy`/`HTTPS_PROXY` 等环境变量,把所有
router→engine 的 `/generate` 都送进了代理**

长的非流式生成在响应返回前 TCP 完全静默,60s 一到被代理统一掐断;router 把代理的 502 当
worker 失败计入 circuit breaker(阈值 10,两个 engine 很快全开)→ 后续请求统一 503
engine 侧的 "client disconnected" 与 abort 都是下游症状

slime 的 httpx 客户端因 `trust_env=False` 不受影响,雷区只在 router→engine 这一跳
`no_proxy` 此前只在 ray runtime env 里设了小写、且不含 worker 节点 IP(reqwest 优先读大写
`NO_PROXY`,容器大写只有 localhost),所以本机和远端 engine 全部走代理

## 修复

| 文件 | 改动 |
| --- | --- |
| `slime/utils/http_utils.py` | 新增 `scrub_proxy_env()`,在 `run_router`(router 子进程)入口清除大小写 `http_proxy`/`https_proxy`/`all_proxy`;router 只与集群内 worker 通信,不应走代理。`SLIME_SCRUB_PROXY=0` 可关闭(默认开启) |
| `examples/kernel_agent/run.t1.qwen3.8.27B.fasync.sh` | runtime env 同时设置大写 `NO_PROXY` 和小写 `no_proxy`,覆盖全部节点 IP(master + REMOTE_HOSTS),兜底其他使用系统代理设置的 HTTP 客户端 |

两层防御独立生效:`scrub_proxy_env` 兜底任何 no_proxy 配置失误;NO_PROXY 列表兜底其他组件
(如 worker 注册用的 `requests`)。注意 NO_PROXY 列表来自脚本静态节点清单,扩节点需同步

## 验证

| 实验 | 结果 |
| --- | --- |
| 同一 sleep-90s worker(绑 `10.11.2.164`),单请求走代理 vs 直连 | 走代理 60.1s 整收到 `502 Bad Gateway`(worker 侧 t=60.0s 看到断连);直连 90.1s 返回 200。根因最小复现 |
| 真 engine(27B TP4,与失败 run 同参数)+ 继承代理 env 的 router + 48 并发 12k-token 非流式 /generate | 提交后 ~61s engine 一次性 abort 49 个请求,客户端 ~63s 起收 `503 no_available_workers`;router debug 日志显式打出 `proxy(http://10.11.2.164:7893/) intercepts ...` 与 CB `closed -> open`。与线上 7 次失败(首 abort 全部在 rollout 开始后 63–69s)同签名 |
| 同一 engine + 修复后 router(经 `slime.utils.http_utils.run_router` 启动,自动 scrub)+ 同样 48 并发 | **48/48 全部 200**(最长 249.3s,远超 60s),engine 零新增 disconnect |

排除项(隔离实验证实与根因无关):slime http 客户端与事件循环结构、router 重试与 circuit
breaker 配置本身、sglang engine 的 is_disconnected 轮询(只是如实上报代理拆链)、KernelGym、
数据与 thinking 配置、3000s timeout、colocate。早前 run 的 502 风暴(engine 在远端节点)与
后期 503 风暴(engine 在本机)是同一根因的两种表象

codex review(xhigh)结论:修复正确(reqwest 代理变量族覆盖完整、scrub 时机在 reqwest client
构建之前、父进程 env 不受影响、保留 NO_PROXY 无害);其建议的逃生开关已采纳

## 因果链

1. rollout 发出 ~64 个非流式 `/generate`,router 的 reqwest 因环境变量走 clash 代理转发到 engine
2. 长生成期间连接静默;代理 60s 超时,对 router 返回 502、对 engine 侧拆链
3. engine 的 4s `is_disconnected` 轮询发现断连,abort 请求并写 400(写进已死的代理连接,客户端永远看不到 400)
4. router 把 502 当 worker 失败记入 circuit breaker(reqwest 无错误日志;CB 状态迁移是 info 级,线上 warn 级被吞,故 router 日志静默)
5. 两个 worker 失败数各 ≥10 → circuit 全开 → 新请求与重试统一 `503 no_available_workers`
6. 60s 后 circuit half-open,部分请求恢复(线上日志中 engine 并发已回升,但 job 恰在此时被人工停止)


## 可复核证据

历史实验目录（experiments/FAsync.TVM.Qwen3.6-27B.CTX16384/router_repro/,2026-06-12):
- probe_proxy_vs_direct.log + worker_29440.log:代理 60.1s 502 vs 直连 90.1s 200 的最小复现
- router_debug.log:`proxy(http://10.11.2.164:7893/) intercepts 'http://10.11.2.164:29430/'`(reqwest debug),
  以及 48 × 503(latency≈62.9s)
- router.log:CB `closed -> open` / `open -> half_open` / `half_open -> open` 序列
- client_slime_stack.log / client_testD.log / client_real_router48.log:slime 栈与裸 httpx 在 48 并发下
  ~63s 全军覆没;client_real_router.log(32 并发 4k tokens,多数 <60s 完成)全部 200
- client_testFIX.log:修复后 48/48 全 200,engine disconnect 计数不变
- conns_testC.log:48 条 router→engine 连接同一秒由 ESTAB 全转 TIME-WAIT
- 线上失败 log 时间差:7 个 run 首 abort 距 rollout start 63/63/64/64/69/63/64s
  (logs/20260611.log 与 logs/20260612.*.log;052722 run 首 prefill 05:35:45 → abort 05:36:45 = 60.0s)
- 反证:fake worker 绑 127.0.0.1(在 no_proxy 豁免里)时 66/66 全过 —— 曾导致"网络层干净"的误判
- 环境:`http_proxy=https_proxy=HTTP(S)_PROXY=http://10.11.2.164:7893`,`NO_PROXY=127.0.0.1,localhost,::1`,
  clash 监听 0.0.0.0:7893,`CLASH_PROXY_PROFILE=flower`
