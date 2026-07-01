# CVRP 数据集来源与验证记录
## 问题族与任务
- **数学问题族**：CVRP
- **对应 LLM4AD 任务目录**：cvrp_construct、cvrpf
- **数据集名称**：CVRPLIB
- **实例系列**：A, B, P, X

## 来源与下载信息
- **来源页面**：https://galgos.inf.puc-rio.br/cvrplib/en/instances
- **实际下载 URL**：
  - https://galgos.inf.puc-rio.br/cvrplib/en/download/instance-set/4
  - https://galgos.inf.puc-rio.br/cvrplib/en/download/instance-set/5
  - https://galgos.inf.puc-rio.br/cvrplib/en/download/instance-set/12
  - https://galgos.inf.puc-rio.br/cvrplib/en/download/instance-set/17
- **下载日期**：2026-06-29

## 作者、维护机构与引用
- **维护机构**：Galgos/PUC-Rio 研究团队
- **A、B、P 集引用**：Augerat et al. (1995), “Computational results with a branch and cut code for the capacitated vehicle routing problem.”
- **X 集引用**：Uchoa et al. (2017), “New benchmark instances for the Capacitated Vehicle Routing Problem”, *European Journal of Operational Research*, 257(3), 845–858, DOI: 10.1016/j.ejor.2016.08.012。

## 原始文件清单
| 文件名 | 大小 | SHA-256 |
|---|---|---|
| `A.7z` | 9.0 KiB | `92f6378b4e52ceef4d676d78d5fd0a70828b721be64457c8280cec4bcf9797e2` |
| `B.7z` | 8.3 KiB | `ee60e03fd6bf03df98e1ec1aada8d40e57fbb056d8cc4648249eb54956ff1696` |
| `P.7z` | 4.2 KiB | `aebb86994726a960b5445805e4ede075bc08d539c54b807176ce525c97bdbd9b` |
| `X.7z` | 290.5 KiB | `141b45603580ab88c79b830c697c95348393f9105be14b0d70a9b9334f569fcb` |

## 实例规模与格式
- **格式说明**：174 .vrp instances found across sets ['A', 'B', 'P', 'X']
- **兼容性结论**：`CONVERSION_REQUIRED`
- **评价器契约验证**：MATCH after deterministic conversion
- **Smoke test**：PASS 

## 许可与使用限制
CVRPLIB 使用条款说明平台与材料仅供研究使用；各个数据集的再分发可能受附加条款约束，不要假设可无限制再分发。

## 备注
- CVRPLIB 使用 TSPLIB/VRPLIB 风格的 .vrp 格式与 1-based 节点编号；需要转换为 0-based 并重新计算距离矩阵。
- 当前评价器不惩罚车辆数，目标是在容量可行前提下最小化总行驶距离。
