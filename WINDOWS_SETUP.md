# LLM4AD Windows 安装与运行指南

> 本分支 `上传windows` 已经把 `data/benchmarks/` 目录放进仓库，Windows 上无需额外下载数据集。  
> 但 1.7 GB 的 knapsack 数据（当前任务未使用）已被 `.gitignore` 排除。

---

## 一、安装前置软件

### 1. Git for Windows
1. 访问 https://git-scm.com/download/win
2. 下载并安装，建议一路默认即可。
3. 安装完成后，在 PowerShell 中验证：
   ```powershell
   git --version
   ```

### 2. Python 3.11
1. 访问 https://www.python.org/downloads/release/python-3119/
2. 下载 **Windows installer (64-bit)**。
3. 安装时**务必勾选**：
   - ✅ **Add python.exe to PATH**
   - ✅ **Use admin privileges when installing py.exe**
   - ✅ 在 *Optional Features* 里勾选 **tcl/tk and IDLE**（GUI 需要 tkinter）
4. 验证：
   ```powershell
   python --version
   ```
   应显示 `Python 3.11.x`。

> 建议不要安装 Python 3.13，因为 `setup.py` 要求 `>=3.9,<3.13`。

---

## 二、启用 Windows 长路径支持（重要）

本项目的 benchmark 文件路径较深，例如：
```
data/benchmarks/bp_1d/extracted/Falkenauer/Falkenauer/Falkenauer U/Falkenauer_u1000_00.txt
```
Windows 默认 260 字符路径限制会导致文件无法读取。

### 步骤 1：启用系统长路径
以**管理员身份**打开 PowerShell，执行：
```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```
执行后**重启电脑**。

### 步骤 2：让 Git 支持长路径
```powershell
git config --global core.longpaths true
```

---

## 三、下载代码

打开 PowerShell，进入你想放项目的目录（例如 `D:\Projects`）：
```powershell
cd D:\Projects
git clone https://github.com/Arrchetto/LLM4AD.git
cd LLM4AD
git checkout 上传windows
```

---

## 四、创建虚拟环境

虚拟环境是为了把项目依赖和系统 Python 隔离开，避免冲突。

### 1. 创建 venv
```powershell
python -m venv .venv
```
这会在 `LLM4AD\.venv\` 目录下创建一个新的 Python 环境。

### 2. 激活 venv
```powershell
.venv\Scripts\Activate.ps1
```
激活成功后，命令行前面会出现 `(.venv)` 标识：
```powershell
(.venv) PS D:\Projects\LLM4AD>
```

> 如果提示“无法加载脚本 `Activate.ps1`”，执行：
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> 然后重新激活。

### 3. 升级 pip
```powershell
python -m pip install --upgrade pip
```

### 4. 安装项目依赖
```powershell
pip install -e .
```
`-e .` 表示以“可编辑模式”安装本项目，并自动读取 `setup.py` 中的依赖列表。

> 在 Windows 上，`llamea` 需要从 Git 安装，请确保 Git 已经在 PATH 中。  
> 如果 `torch` 下载过慢，可以先安装 CPU 版：
> ```powershell
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

---

## 五、VS Code 中选择哪个解释器

如果你用 VS Code 打开项目，必须确保它使用虚拟环境中的 Python，而不是系统 Python。

### 方法一：通过命令面板选择
1. 在 VS Code 中打开 `LLM4AD` 文件夹。
2. 按 `Ctrl+Shift+P`（或 `Cmd+Shift+P`）。
3. 输入并选择：`Python: Select Interpreter`。
4. 选择：
   ```
   Python 3.11.x ('.venv': venv)  .\.venv\Scripts\python.exe
   ```
   如果没有自动出现，点击 **Enter interpreter path...**，然后找到：
   ```
   D:\Projects\LLM4AD\.venv\Scripts\python.exe
   ```

### 方法二：通过状态栏选择
1. 打开任意 `.py` 文件。
2. 看右下角状态栏，会显示当前解释器，例如 `Python 3.11.x`。
3. 点击它，然后选择 `.venv` 中的解释器。

### 验证解释器是否正确
在 VS Code 终端中执行：
```powershell
where python
```
应该输出：
```
D:\Projects\LLM4AD\.venv\Scripts\python.exe
...
```

---

## 六、验证安装

### 验证 1：cvrpf 数据能否加载
```powershell
python -c "from llm4ad.task.optimization.cvrpf import CVRPFEvaluation; e = CVRPFEvaluation(); print(f'OK: {len(e.training_instances)} instances loaded')"
```
期望输出：
```
OK: 50 instances loaded
```

### 验证 2：运行 cvrpf 单元测试
```powershell
python -m unittest tests.test_cvrpf -v
```
期望结果：
```
Ran 51 tests in ...
OK
```

### 验证 3：任务自动发现
```powershell
python -c "from llm4ad.task import CVRPFEvaluation; print('OK:', CVRPFEvaluation.__name__)"
```

---

## 七、设置 API 密钥

在运行 LLM 实验前，必须设置你的 LLM API key。

### 方式一：仅当前终端有效
```powershell
$env:LLM_API_KEY = "sk-你的真实密钥"
```

### 方式二：永久设置用户级环境变量
```powershell
[System.Environment]::SetEnvironmentVariable("LLM_API_KEY", "sk-你的真实密钥", "User")
```
设置后需要**重启 PowerShell 或 VS Code**。

---

## 八、运行示例

### 运行 cvrpf EoH 示例
```powershell
cd example/tasks/cvrp_construct
python run_eoh.py
```

### 启动 GUI
```powershell
cd GUI
python run_gui.py
```

---

## 九、重新安装虚拟环境

如果你换电脑、venv 损坏或依赖出错，可以删掉重新创建：

```powershell
cd D:\Projects\LLM4AD

# 1. 如果 venv 已激活，先退出
deactivate

# 2. 删除旧 venv（PowerShell）
Remove-Item -Recurse -Force .venv

# 3. 重新创建
python -m venv .venv

# 4. 激活
.venv\Scripts\Activate.ps1

# 5. 升级 pip 并安装依赖
python -m pip install --upgrade pip
pip install -e .
```

> 如果 `Remove-Item` 因路径太长失败，先启用长路径支持（见第二节），或用 `rmdir /s /q .venv`。

---

## 十、常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `python` 命令找不到 | 安装 Python 时未勾选 Add to PATH | 重装 Python 并勾选，或手动添加 PATH |
| `Activate.ps1` 无法执行 | PowerShell 执行策略限制 | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `No module named 'tkinter'` | 未安装 tkinter | 重装 Python 并勾选 tcl/tk |
| 数据文件读取失败 | 路径超过 260 字符 | 启用 Windows 长路径支持 |
| `git clone` 后文件缺失 | Git 长路径限制 | `git config --global core.longpaths true` 后重新 clone |
| `pip install -e .` 安装 llamea 失败 | Git 未加入 PATH | 确保 Git for Windows 已安装并重启终端 |
| GUI 打不开或显示异常 | ttkbootstrap / matplotlib 版本问题 | `pip install --upgrade ttkbootstrap matplotlib` |

---

## 十一、数据目录说明

- 代码中 `cvrpf` 任务会默认查找：`LLM4AD/data/benchmarks/cvrp/extracted/`
- 你也可以通过环境变量覆盖：
  ```powershell
  $env:LLM4AD_DATA_ROOT = "D:\你的数据路径\data\benchmarks\cvrp\extracted"
  ```
- 1.7 GB 的 `knapsack` 数据未被任何当前任务使用，已排除在 git 之外。
