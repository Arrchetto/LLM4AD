# Offline 1D Bin Packing 数据集来源与验证记录
## 问题族与任务
- **数学问题族**：Offline 1D Bin Packing
- **对应 LLM4AD 任务目录**：bp_1d_construct
- **数据集名称**：BPPLIB
- **实例系列**：Falkenauer U/T, Scholl, Wäscher, Schwerin, Hard28

## 来源与下载信息
- **来源页面**：https://site.unibo.it/operations-research/en/research/bpplib-a-bin-packing-problem-library
- **实际下载 URL**：
  - https://site.unibo.it/operations-research/en/research/bpplib-a-bin-packing-problem-library/falkenauer.rar/@@download/file/Falkenauer.rar
  - https://site.unibo.it/operations-research/en/research/bpplib-a-bin-packing-problem-library/scholl.rar/@@download/file/Scholl.rar
  - https://site.unibo.it/operations-research/en/research/bpplib-a-bin-packing-problem-library/wascher.rar/@@download/file/W%C3%A4scher.rar
  - https://site.unibo.it/operations-research/en/research/bpplib-a-bin-packing-problem-library/schwerin.rar/@@download/file/Schwerin.rar
  - https://site.unibo.it/operations-research/en/research/bpplib-a-bin-packing-problem-library/hard28.rar/@@download/file/Hard28.rar
- **下载日期**：2026-06-29

## 作者、维护机构与引用
- **维护机构**：M. Delorme, M. Iori, S. Martello / University of Bologna
- **正式引用**：Delorme, Iori & Martello (2018). BPPLIB: a library for bin packing and cutting stock problems. Optimization Letters 12, 235–250. DOI:10.1007/s11590-017-1192-z

## 原始文件清单
| 文件名 | 大小 | SHA-256 |
|---|---|---|
| `Falkenauer.rar` | 42.3 KiB | `e09f6ed02addc00f21db98229c17e1bdc0e42d201a3a799598594057a5a85316` |
| `Scholl.rar` | 264.1 KiB | `b1b0dd2c4f641256a5a5638fc5a5eb0819ac3f2da9baf1be0cad87176e5d1971` |
| `Wäscher.rar` | 4.0 KiB | `61c703fb5a76fa3f74e4be5baad60a1ace57a11a73833843fba33f9575cfeb88` |
| `Schwerin.rar` | 34.2 KiB | `3c7d0a5d291cf842669cbca5d0b5249e6c0ec22476c4e0c9d9dace35571b7d59` |
| `Hard28.rar` | 10.2 KiB | `cee6c3417a5fd34d69da1142898e70720f32a4b37c2e555e9cc07aae83104fec` |

## 实例规模与格式
- **格式说明**：1615 .txt BPP-format instances found
- **兼容性结论**：`DIRECT`
- **评价器契约验证**：MATCH
- **Smoke test**：PASS 

## 许可与使用限制
BPPLIB 仓库本身有代码许可证，但其中的历史实例系列来自多位作者，未对每个系列统一声明再分发许可；请保留原始署名与引用。

## 备注
- 仅直接使用 BPP 格式文件；CSP 格式需要按需求展开，不作为直接匹配使用。
