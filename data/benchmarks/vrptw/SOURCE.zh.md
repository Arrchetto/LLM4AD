# VRPTW 数据集来源与验证记录
## 问题族与任务
- **数学问题族**：VRPTW
- **对应 LLM4AD 任务目录**：vrptw_construct
- **数据集名称**：SINTEF Solomon + Gehring–Homberger
- **实例系列**：Solomon-100, Homberger-200, Homberger-400, Homberger-600, Homberger-800, Homberger-1000

## 来源与下载信息
- **来源页面**：https://www.sintef.no/projectweb/top/vrptw/
- **实际下载 URL**：
  - https://www.sintef.no/globalassets/project/top/vrptw/solomon/solomon-100.zip
  - https://www.sintef.no/globalassets/project/top/vrptw/homberger/200/homberger_200_customer_instances.zip
  - https://www.sintef.no/globalassets/project/top/vrptw/homberger/400/homberger_400_customer_instances.zip
  - https://www.sintef.no/globalassets/project/top/vrptw/homberger/600/homberger_600_customer_instances.zip
  - https://www.sintef.no/globalassets/project/top/vrptw/homberger/800/homberger_800_customer_instances.zip
  - https://www.sintef.no/globalassets/project/top/vrptw/homberger/1000/homberger_1000_customer_instances.zip
- **下载日期**：2026-06-29

## 作者、维护机构与引用
- **维护机构**：SINTEF Transportation Optimization Portal
- **正式引用**：Solomon, M. M. (1987). Algorithms for the Vehicle Routing and Scheduling Problems with Time Window Constraints. Operations Research 35(2), 254–265. DOI:10.1287/opre.35.2.254

## 原始文件清单
| 文件名 | 大小 | SHA-256 |
|---|---|---|
| `Solomon-100.zip` | 81.6 KiB | `8a0a72cbe6b7f8f9988ace4ebde0378ec34943acaaac47f2c408915e41887747` |
| `Homberger-200.zip` | 168.1 KiB | `79092cc627135f370a6381b0c64afc8403e4d4ff74afa8808d28d208ac784571` |
| `Homberger-400.zip` | 322.4 KiB | `669e9e7c6fe3513b2c62028d17b67aa5829d94cf2b379f362c8c71236e6a366c` |
| `Homberger-600.zip` | 485.6 KiB | `df4c40d2191f7e854c46f5422a89055bae93fb1f880692181ee8ee7bf629636b` |
| `Homberger-800.zip` | 649.8 KiB | `19cad076841f8bf6e8d05ee4d5c56ef2925dacfce64495e003d65bfc6d8545aa` |
| `Homberger-1000.zip` | 816.0 KiB | `2e96845ee870ef21f833a216a79866e812232d3d922db3e14c30790f06780a81` |

## 实例规模与格式
- **格式说明**：356 .txt instances found
- **兼容性结论**：`CONVERSION_REQUIRED`
- **评价器契约验证**：MATCH after deterministic conversion
- **Smoke test**：PASS 

## 许可与使用限制
SINTEF VRPTW 页面未提供独立的数据许可证；请保留原始文件与引用，不要默认其可公开再分发。

## 备注
- Solomon/Homberger 文件使用自定义文本格式与 1-based 节点编号；需要转换为 0-based 并固定距离/目标口径。
- 经典 Solomon 比较常采用层级目标（先车辆数、后距离），而当前评价器以总距离为目标；实验前必须明确记录所采用的目标规则。
