# Online 1D Bin Packing 数据集来源与验证记录
## 问题族与任务
- **数学问题族**：Online 1D Bin Packing
- **对应 LLM4AD 任务目录**：online_bin_packing, online_bin_packing_2O
- **数据集名称**：FunSearch Weibull protocol
- **实例系列**：Weibull(45,3) clipped to [1,100], seed 2024

## 来源与下载信息
- **来源页面**：https://github.com/google-deepmind/funsearch
- **实际下载 URL**：generated from LLM4AD/llm4ad/task/optimization/online_bin_packing/generate_weibull_instances.py
- **下载日期**：2026-06-29

## 作者、维护机构与引用
- **维护机构**：Google DeepMind FunSearch 团队
- **正式引用**：Romera-Paredes et al. (2023). Mathematical discoveries from program search with large language models. Nature. DOI:10.1038/s41586-023-06924-6

## 原始文件清单
| 文件名 | 大小 | SHA-256 |
|---|---|---|
| `generated/test_10000.json` | 488.3 KiB | `0f91def5fdf4e2cfc5cd026b2d15558689ecb404120ba43468ab4aec333ee18f` |
| `generated/test_100000.json` | 975.8 KiB | `d5001ab7baad1a1bb799930f0d46301c4233cf19a9578e1ecf1f0714ec33bdfa` |
| `generated/test_5000.json` | 244.4 KiB | `683cd8559fc3721e4af7850ebb1a3bb283ddd4c371247f78707d5658d32ca809` |
| `generated/train_5000.json` | 244.4 KiB | `f5c972b14e91f857bdaf67b73a4f432af880fad554c91831af7f98c7d7befebd` |
| `generated/val_5000.json` | 244.3 KiB | `7ce6669376f1039b917b59d892499b62d1ca5123e07a630954259f8022ffc006` |

## 实例规模与格式
- **格式说明**：21 JSON instance files with capacity, num_items and items array
- **兼容性结论**：`DIRECT`
- **评价器契约验证**：MATCH
- **Smoke test**：PASS 

## 许可与使用限制
FunSearch 软件按 Apache-2.0 许可，其他材料按 CC BY 4.0 许可；使用时请引用 Romera-Paredes et al., Nature, DOI:10.1038/s41586-023-06924-6。

## 备注
- 不存在外部静态压缩包；协议、随机种子与生成器版本即为可追溯来源。
- 实例在生成后被冻结，重复运行可得到完全相同的文件。
