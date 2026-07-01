# QAP 数据集来源与验证记录
## 问题族与任务
- **数学问题族**：QAP
- **对应 LLM4AD 任务目录**：qap_construct
- **数据集名称**：QAPLIB
- **实例系列**：Nugent, Taillard, Skorin-Kapov, Burkard, Hadley, Lipa, others

## 来源与下载信息
- **来源页面**：https://qaplib.mgi.polymtl.ca/
- **实际下载 URL**：
  - https://qaplib.mgi.polymtl.ca/data.d/qapdata.tar.gz
  - https://qaplib.mgi.polymtl.ca/soln.d/qapsoln.tar.gz
- **下载日期**：2026-06-29

## 作者、维护机构与引用
- **维护机构**：QAPLIB 维护者（École Polytechnique de Montréal 镜像）
- **正式引用**：Burkard, Karisch & Rendl (1997). QAPLIB—A Quadratic Assignment Problem Library. Journal of Global Optimization 10, 391–403.

## 原始文件清单
| 文件名 | 大小 | SHA-256 |
|---|---|---|
| `qapdata.tar.gz` | 416.9 KiB | `c6bf0e277156d097df21124f53ca5a5d0fa5344481ee401e13acb67694e190b3` |
| `qapsoln.tar.gz` | 13.8 KiB | `002243dea9452000ec7ee5e43bd0aa651a639eb61039c4f35679531e67d71744` |

## 实例规模与格式
- **格式说明**：136 .dat problem files and 128 .sln solution files found
- **兼容性结论**：`DIRECT`
- **评价器契约验证**：MATCH
- **Smoke test**：PASS 

## 许可与使用限制
QAPLIB 页面未声明统一的现代数据许可证；各实例来自不同贡献者，请按系列保留署名与引用，不要默认其可公开再分发。

## 备注
- QAPLIB .dat 文件给出 n 以及 flow/distance 矩阵；解文件使用 1-based 排列，需转换为 0-based 后供评价器使用。
