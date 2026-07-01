# Set Cover 数据集来源与验证记录
## 问题族与任务
- **数学问题族**：Set Cover
- **对应 LLM4AD 任务目录**：set_cover_construct
- **数据集名称**：OR-Library unicost CYC/CLR
- **实例系列**：CYC06–CYC11, CLR10–CLR13

## 来源与下载信息
- **来源页面**：https://people.brunel.ac.uk/~mastjjb/jeb/orlib/scpinfo.html
- **实际下载 URL**：
  - http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/scpcyc06.txt
  - http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/scpcyc07.txt
  - http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/scpcyc08.txt
  - http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/scpcyc09.txt
  - http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/scpcyc10.txt
  - http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/scpcyc11.txt
  - http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/scpclr10.txt
  - http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/scpclr11.txt
  - http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/scpclr12.txt
  - http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/scpclr13.txt
- **下载日期**：2026-06-29

## 作者、维护机构与引用
- **维护机构**：J. E. Beasley / Brunel University OR-Library
- **正式引用**：Beasley, J. E. (1987). An algorithm for set covering problems. EJOR 31, 85–93; OR-Library overview: Beasley (1990). DOI:10.1057/jors.1990.166

## 原始文件清单
| 文件名 | 大小 | SHA-256 |
|---|---|---|
| `scpcyc06.txt` | 5.0 KiB | `d92472193b3eb69ce675941da0a4a6a4c5a7d31f6d2b4e1ff7815d4d298011d5` |
| `scpcyc07.txt` | 14.8 KiB | `0b358ecba9227bb2e942f9b4edf9a6c557483df6157b5aa2b73e4e4882f020e3` |
| `scpcyc08.txt` | 40.1 KiB | `89a8a298d01cb8a87597cc15b7498ed2e0066ca0a8bdb87a37f897488adda504` |
| `scpcyc09.txt` | 113.2 KiB | `7a1e374bcf61ab77bf3913738f0fcd01a4cf0d13912a3a6005bc9db85fc774b0` |
| `scpcyc10.txt` | 293.6 KiB | `0f53dc9cd09c16ee3e11e01313be0ea209b81832658770a1f2d35e267b5d46e7` |
| `scpcyc11.txt` | 740.4 KiB | `77f9b41b62caa047b1d99dac1cb87f77c11ac0acd8bafb59a45fc0fe65ee80d4` |
| `scpclr10.txt` | 50.5 KiB | `7ae9fa82a59ca6d411cf179ef3541183c4c18c03bac0a18fb81814536563a441` |
| `scpclr11.txt` | 163.3 KiB | `4de5c8fa47124988b35839eeaee3b67a6f28b24159a3db854826e29298673847` |
| `scpclr12.txt` | 499.9 KiB | `10f228f4419bdc4b2a698289b58509606a92c222ec2612234970e39e02fa528d` |
| `scpclr13.txt` | 1.4 MiB | `d46def0ee4e3d8e161051d5122ef597bfb855de48b4ead3d07c89d186b9b9402` |

## 实例规模与格式
- **格式说明**：10 .txt instances collected
- **兼容性结论**：`DIRECT`
- **评价器契约验证**：MATCH
- **Smoke test**：PASS 

## 许可与使用限制
OR-Library 数据页未声明独立的开源许可证；请保留 Beasley/原作者署名与引用，不要默认其可公开再分发。

## 备注
- 仅下载 unicost CYC/CLR 实例，因为当前评价器忽略列成本、以最小化选中子集数量为目标。
- OR-Library 的普通 SCP/rail 系列为 PARTIAL，此处未下载。
