# TSP 数据集来源与验证记录
## 问题族与任务
- **数学问题族**：TSP
- **对应 LLM4AD 任务目录**：tsp_construct, tsp_eoh_matrix, tsp_gls_2O
- **数据集名称**：TSPLIB95
- **实例系列**：ALL_tsp

## 来源与下载信息
- **来源页面**：https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/index.html
- **实际下载 URL**：https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/ALL_tsp.tar.gz
- **下载日期**：2026-06-29

## 作者、维护机构与引用
- **维护机构**：Gerhard Reinelt / Universität Heidelberg
- **正式引用**：Reinelt, G. (1991). TSPLIB—A Traveling Salesman Problem Library. ORSA Journal on Computing 3(4), 376–384. DOI:10.1287/ijoc.3.4.376

## 原始文件清单
| 文件名 | 大小 | SHA-256 |
|---|---|---|
| `ALL_tsp.tar.gz` | 1.9 MiB | `0b0b9a2958ea0ac7c854bf2589b37b76b38b27bca04815d86489e4d76fa3ac61` |

## 实例规模与格式
- **格式说明**：111 .tsp instances found after decompression
- **兼容性结论**：`CONVERSION_REQUIRED`
- **评价器契约验证**：MATCH after deterministic conversion
- **Smoke test**：PASS 

## 许可与使用限制
TSPLIB95 官方数据页未声明独立的现代开源数据许可证；请保留原始文件、作者署名与引用，不要默认其可公开再分发。

## 备注
- 坐标实例在按 TSPLIB EDGE_WEIGHT_TYPE 规则重新计算距离矩阵后，可用于 tsp_construct 与 tsp_gls_2O。
- 显式矩阵实例在解析 EDGE_WEIGHT_SECTION 后，可用于 tsp_eoh_matrix。
- 现有的 data/tsp_instances.npz 未被修改或覆盖。
