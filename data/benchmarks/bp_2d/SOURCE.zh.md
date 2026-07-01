# 2D Bin Packing 数据集来源与验证记录
## 问题族与任务
- **数学问题族**：2D Bin Packing
- **对应 LLM4AD 任务目录**：bp_2d_construct
- **数据集名称**：2DPackLib
- **实例系列**：CLASS, BENG

## 来源与下载信息
- **来源页面**：https://site.unibo.it/operations-research/en/research/2dpacklib
- **实际下载 URL**：
  - https://site.unibo.it/operations-research/en/research/2dpacklib/class.zip/@@download/file/CLASS.zip
  - https://site.unibo.it/operations-research/en/research/2dpacklib/beng.zip/@@download/file/BENG.zip
- **下载日期**：2026-06-29

## 作者、维护机构与引用
- **维护机构**：University of Bologna / University of Modena and Reggio Emilia
- **正式引用**：Iori, de Lima, Martello & Monaci (2022). 2DPackLib: a two-dimensional cutting and packing library. Optimization Letters 16, 471–480. DOI:10.1007/s11590-021-01808-y

## 原始文件清单
| 文件名 | 大小 | SHA-256 |
|---|---|---|
| `CLASS.zip` | 234.7 KiB | `9f5ca94d6a9ee5c9273302b8fbddf028c3f6a1d595e80622aafe79ec55589f9c` |
| `BENG.zip` | 5.7 KiB | `90536b052787ed7d04bb58c4b03ab6308247685e3890facc3f2773867830816c` |

## 实例规模与格式
- **格式说明**：510 .ins2D instances found
- **兼容性结论**：`CONVERSION_REQUIRED`
- **评价器契约验证**：MATCH after deterministic conversion
- **Smoke test**：PASS 

## 许可与使用限制
2DPackLib 论文按 CC BY 4.0 许可，但各历史实例系列未单独声明独立许可证；请保留原始署名与引用。

## 备注
- 文件包含每类物品的 demand；转换时需要将每类矩形按需求数量展开为独立物品。
- 当前评价器不旋转物品、不要求 guillotine 切割，因此仅使用 oriented、free-cutting 的 2D-BPP 实例。
