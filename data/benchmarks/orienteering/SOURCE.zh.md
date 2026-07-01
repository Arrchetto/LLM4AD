# Orienteering Problem 数据集来源与验证记录
## 问题族与任务
- **数学问题族**：Orienteering Problem
- **对应 LLM4AD 任务目录**：orienteering_construct, orienteering_class
- **数据集名称**：KU Leuven Orienteering Problem Library
- **实例系列**：TsiligiridesOP1, TsiligiridesOP2, TsiligiridesOP3, ChaoOP64, ChaoOP66

## 来源与下载信息
- **来源页面**：https://www.mech.kuleuven.be/en/mim/op
- **实际下载 URL**：
  - https://www.mech.kuleuven.be/en/mim/op/instances/TsiligiridesOP1
  - https://www.mech.kuleuven.be/en/mim/op/instances/TsiligiridesOP2
  - https://www.mech.kuleuven.be/en/mim/op/instances/TsiligiridesOP3
  - https://www.mech.kuleuven.be/en/mim/op/instances/ChaoOP64
  - https://www.mech.kuleuven.be/en/mim/op/instances/ChaoOP66
- **下载日期**：2026-06-29

## 作者、维护机构与引用
- **维护机构**：KU Leuven Centre for Industrial Management / Traffic and Infrastructure
- **正式引用**：Tsiligirides (1984). Heuristic Methods Applied to Orienteering; Chao, Golden & Wasil (1996). A fast and effective heuristic for the Orienteering Problem. EJOR 88, 475–489. DOI:10.1016/0377-2217(95)00035-6

## 原始文件清单
| 文件名 | 大小 | SHA-256 |
|---|---|---|
| `TsiligiridesOP1.zip` | 6.2 KiB | `973ef6a5e82012fff6ebaa68096615be68111ab90b230e42dcf787db941deef7` |
| `TsiligiridesOP2.zip` | 3.4 KiB | `1fca710dd4b9ba644403f2d2d855bc85969bd97f32b8860ab7c1b48cbb78547c` |
| `TsiligiridesOP3.zip` | 7.4 KiB | `546b3cd571544e9cb5b1989c8731b97cb01219894f65b277652ff5291ce55a4f` |
| `ChaoOP64.zip` | 4.8 KiB | `0b4d7b014f4223836ca1cdcb628a825825cba671b6a931a023fa85f5a063407c` |
| `ChaoOP66.zip` | 7.8 KiB | `2b95b2d46a9a58e54e87d04389a1c96b94c2ebd59fbcd936d3bb1521e54e2ed6` |

## 实例规模与格式
- **格式说明**：89 .txt instances found
- **兼容性结论**：`CONVERSION_REQUIRED`
- **评价器契约验证**：MATCH after deterministic conversion
- **Smoke test**：PASS 

## 许可与使用限制
KU Leuven OP 下载页未声明独立的数据许可证；请保留原始文件与引用，不要默认其可公开再分发。

## 备注
- 官方格式给出独立的起点与终点；当前评价器允许任意 start_node/end_node，因此可映射为 start=第一个点、end=第二个点。
- 转换时需要计算欧氏距离矩阵与 Tmax 预算。
