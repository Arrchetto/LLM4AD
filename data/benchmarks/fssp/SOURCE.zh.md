# 数据集来源与验证记录

## 问题族

流水车间调度问题（Flow Shop Scheduling Problem, FSSP）

## LLM4AD 任务

- fssp_eoh_full（后续：fssp_llamea_class）

## 数据集

Taillard 流水车间调度实例，通过 CO-Bench 基准分发。

## 来源页面

https://huggingface.co/datasets/CO-Bench/CO-Bench

## 参考文献

Sun, W., Feng, S., Li, S., & Yang, Y. (2025). Co-bench: Benchmarking language model agents in algorithm search for combinatorial optimization. arXiv preprint arXiv:2504.04310.

Taillard, É. (1993). Benchmarks for basic scheduling problems. European Journal of Operational Research, 64(2), 278-285.

## 下载日期

2026-07-01

## 许可/条款

仅限学术研究使用；保留原始署名。

## 最优已知值

最优已知 makespan 取自 CO-Bench 文件中的 `upper_bound` 字段，状态标记为 `BEST_KNOWN`。这些是上界/可行解，并非已证明的最优值。
