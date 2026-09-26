# FALCON++ 表格复现与完整流程

论文：**FALCON++: Shorter Signatures without NTRU Smoothing Estimates**  
作者：**Hao Yan、Nicholas Zhao**，**Imperial College London（帝国理工学院）**  
邮箱：`{h.yan22,n.zhao22}@imperial.ac.uk` · [English](README.md)

精简版保留两项功能：**完整复现正文 6 张表**，以及运行 **KeyGen → Sign → Verify**。对应参数为 I-1245（n=512，q=509）和 V-117（n=1024，q=1949）。

## 运行

使用已验证的 **Python 3.13**，在仓库根目录执行：

```sh
python -m pip install -r requirements.txt
python reproduce_paper.py
python demo_falconpp.py --seed example-run
```

仅依赖 NumPy 和 mpmath，版本已固定；建议安装在 Python 虚拟环境中。不需要 LaTeX 或绘图库。

## 完整复现表格

`python reproduce_paper.py` 重新计算、导出 CSV/Markdown，并检查 **全部 88 个数据单元格**；任何不一致都会报错。

| 表格 | 内容 | 输出 |
|---|---|---|
| 表 1 | 相同宽度下的采样次数 | `results/tables/table1.csv` |
| 表 2 | 公钥/签名大小及代价估计 | `results/tables/table2.csv` |
| 表 3 | 选定参数 | `results/tables/table3.csv` |
| 表 4 | 模数与长度对比 | `results/tables/table4.csv` |
| 表 5 | 矩阶与接受率界 | `results/tables/table5.csv` |
| 表 6 | 启发式代价与 chi-BDD 模型 | `results/tables/table6.csv` |

`results/table_verification.json` 记录逐项核对及来源：38 个计算结果、26 个设计参数、24 个文献输入。`data/paper_tables.tex` 只保留论文中 6 张表的原文及原论文源文件的 SHA-256，用作核对基准，不作为计算答案。文献输入及出处保存在 `data/literature_baselines.json`。

未舍入数值在 `results/selected_parameters.json`，熵区间在 `results/entropy.json`。表 2、表 4 中的 **419/921 字节按论文熵估计公式及界重新计算**；签名示例另外报告实际编码长度。

## 完整签名流程

```sh
python demo_falconpp.py --parameter I-1245
python demo_falconpp.py --parameter V-117
python demo_falconpp.py --wire-format padded --seed example-run
```

默认运行两组参数，使用未填充的规范 rANS；`--wire-format padded` 可查看已有的保守定长填充模式。`--seed` 用于重复实验，不指定时使用系统播种随机性。

KeyGen 生成 NTRU 陷门和采样树；Sign 每次生成新盐、哈希为 syndrome、进行 Klein–GPV 采样、截断校正和范数检查，再编码；Verify 解码、重新哈希并检查重构向量。程序检查正确签名通过、修改消息后验签拒绝，汇总结果写入 `results/workflow.json`，不保存私钥。

## 文件与上传

- 根目录 5 个 Python 文件：表格计算、熵计算、表格导出、逐项核对和完整流程示例。
- `implement/falconpp/`：实际运行所需的密钥生成、算术、采样、签名及编码模块。
- `implement/security/`：表格所需的数值模型。
- `data/`：两个小型输入/核对文件。

精简包不包含论文、图片、历史实验驱动、开发测试集或预生成结果。结果在运行时本地生成。算法原有出处保留在源码中，文献对比数据的出处在 JSON 中；本次打包没有新增许可证授权。

手动上传 GitHub 时，解压 ZIP，打开 `FALCONpp-github`，将**里面的全部内容**拖入网页上传区，保留 `.gitignore` 和子目录。精简后可以一次上传，无需分批。
