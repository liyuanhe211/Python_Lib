# Python_Lib 安装流程（含新项目引用本库）

## 〇、torch 与重型依赖：默认不装，按需启用 extras

本库的默认安装**不含 torch**，也不含会连带拉入 torch 的重型库
（docling / marker-pdf / FlagEmbedding / sentence-transformers / peft）。
torch 体积巨大而只有少数模块用到，**只有当用户明确说明项目需要时**才启用
对应 extra，多个 extra 可用逗号组合（如 `[rag,pdf]`）：

| extra | 内容 | 什么时候需要 |
|---|---|---|
| `[torch]` | torch + torchvision | Machine_Learning_Lib 训练、直接 `import torch` 的代码 |
| `[rag]` | FlagEmbedding / sentence-transformers / peft / datasets / fsspec（连带拉入 torch） | LLM_Lib 的 RAG 嵌入与重排 |
| `[pdf]` | docling / marker-pdf / pymupdf / pypdf / pytesseract / markitdown（连带拉入 torch） | PDF 转换与识别（Marker / Docling 管线） |

## 一、新项目引用本库：Windows 本机（uv 与 pip 二选一，推荐 uv）

依次执行：`git init` 建立新项目自身的版本控制，`uv init` 生成 pyproject.toml，`uv venv` 创建 .venv 虚拟环境，然后以可编辑方式安装本库：

--------------------------

```
setx UV_CACHE_DIR "E:\My_Program\.uv_cache"
set UV_CACHE_DIR=E:\My_Program\.uv_cache
git init
uv init
uv venv
uv add --editable "E:\My_Program\Python_Lib" --config-settings editable_mode=strict
```

--------------------------

项目明确需要 torch / RAG / PDF 功能时，把路径改写成带 extra 的形式，
按需选一行（extras 见开头表格，可逗号组合）：

--------------------------

```
uv add --editable "E:\My_Program\Python_Lib[torch]" --config-settings editable_mode=strict
uv add --editable "E:\My_Program\Python_Lib[rag]" --config-settings editable_mode=strict
uv add --editable "E:\My_Program\Python_Lib[pdf]" --config-settings editable_mode=strict
uv add --editable "E:\My_Program\Python_Lib[torch,rag,pdf]" --config-settings editable_mode=strict
```

--------------------------

装了 torch 且需要 GPU 时，再手动换 CUDA 版 torch（下游项目经 PyPI 装到的是
CPU 版；本仓库 `[tool.uv.sources]` 的 CUDA 源不会传导给下游项目。cu128 对应
本机 RTX 5080）：

--------------------------

```
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

--------------------------

pip 方式（项目不用 uv 管理时），第一行为最小安装，需要 extras 时按需选
带 extra 的行、需要 GPU 时再接上面的 cu128 命令：

--------------------------

```
pip install --editable "E:\My_Program\Python_Lib" --config-settings editable_mode=strict
pip install --editable "E:\My_Program\Python_Lib[torch]" --config-settings editable_mode=strict
pip install --editable "E:\My_Program\Python_Lib[rag]" --config-settings editable_mode=strict
pip install --editable "E:\My_Program\Python_Lib[pdf]" --config-settings editable_mode=strict
pip install --editable "E:\My_Program\Python_Lib[torch,rag,pdf]" --config-settings editable_mode=strict
```

--------------------------

说明：`--editable` 让本库的代码修改即时生效、不用重装；`editable_mode=strict`
生成静态文件映射，Pylance / PyCharm 才能正确解析 import（代价：本库新增
文件后要重跑一次安装命令才可见）。

## 二、新项目引用本库：Linux / HPC

先查 glibc 版本，按结果选下面两条路线之一：

--------------------------

```
getconf GNU_LIBC_VERSION
```

--------------------------

**glibc >= 2.28**（Rocky/Alma 8+、Ubuntu 20.04+ 等），uv 方式：

--------------------------

```
uv add --editable ~/My_Program/Python_Lib --config-settings editable_mode=strict
```

--------------------------

pip 方式：

--------------------------

```
pip install --editable ~/My_Program/Python_Lib --config-settings editable_mode=strict
```

--------------------------

Linux 上默认同样不装 torch，也不装 PDF 读取 / 识别相关库（docling / pymupdf /
pypdf / pytesseract / markitdown[all]），避免拉入大量重型依赖。
需要这些功能时把路径改写成 extra 形式，按需选一行（extras 见开头表格，
可逗号组合）：

--------------------------

```
uv add --editable "~/My_Program/Python_Lib[torch]" --config-settings editable_mode=strict
uv add --editable "~/My_Program/Python_Lib[rag]" --config-settings editable_mode=strict
uv add --editable "~/My_Program/Python_Lib[pdf]" --config-settings editable_mode=strict
uv add --editable "~/My_Program/Python_Lib[torch,rag,pdf]" --config-settings editable_mode=strict
```

--------------------------

```
pip install --editable "~/My_Program/Python_Lib[torch]" --config-settings editable_mode=strict
pip install --editable "~/My_Program/Python_Lib[rag]" --config-settings editable_mode=strict
pip install --editable "~/My_Program/Python_Lib[pdf]" --config-settings editable_mode=strict
pip install --editable "~/My_Program/Python_Lib[torch,rag,pdf]" --config-settings editable_mode=strict
```

--------------------------

**glibc < 2.28**（CentOS 7 系 HPC 登录节点 = glibc 2.17）：上面的命令直接跑会
失败，原因和对策如下。

pip 和 uv 安装时本来就会检测本机实际的 glibc 版本，只考虑标签不高于它的
manylinux wheel（glibc 2.17 对应 `manylinux_2_17`，旧名 `manylinux2014`），
所以**不存在也不需要**一个"安装 glibc 2.17 版本"的专用参数——旧版本 wheel
的选择是自动的。旧机器上装不上的原因有两类：

1. 包的新版本只发 `manylinux_2_28` wheel 但还发了源码包（sdist）——这时
   解析器不退版本、转而现场编译，然后编译失败。对策是把版本压回最后一个
   还发旧 glibc wheel 的版本；本库的 `[legacy]` extra 就是这份现成的上限
   清单（2026-07 在新机器上用
   `uv pip compile --python-platform x86_64-manylinux_2_17` 替旧平台探测
   得到，见 pyproject.toml）。
2. 任何版本都没有旧 glibc wheel、源码现场编译也不可行的包——PyQt6 /
   onnxruntime / docling-parse / sentencepiece——只能跳过不装，对应功能
   （Qt 图形界面等）在这台机器上不可用，其余功能不受影响。

因此旧 glibc 机器上引用本库，uv 方式改成两步：`uv add` 带上 `[legacy]` extra
和 `--no-sync`（隐式 sync 会在 PyQt6 上失败），再手动 `uv sync` 并跳过无解包。
注意 `[legacy]` 本身显式含 CPU 版 torch / torchvision（旧 glibc HPC 上的
转换 / 索引管线需要）：

--------------------------

```
uv add --no-sync --editable "~/My_Program/Python_Lib[legacy]" --config-settings editable_mode=strict
uv sync --no-install-package PyQt6 --no-install-package PyQt6-Qt6 --no-install-package PyQt6-sip --no-install-package onnxruntime
```

--------------------------

需要 PDF / RAG 功能时把第一行换成组合 extra 形式，按需选一行（第二行的
`uv sync` 不变；用了 `[pdf]` 且 Python >= 3.11 时 sync 再加
`--no-install-package docling-parse`，用了 `[rag]` 且 Python >= 3.13 时再加
`--no-install-package sentencepiece`）：

--------------------------

```
uv add --no-sync --editable "~/My_Program/Python_Lib[legacy,pdf]" --config-settings editable_mode=strict
uv add --no-sync --editable "~/My_Program/Python_Lib[legacy,rag]" --config-settings editable_mode=strict
uv add --no-sync --editable "~/My_Program/Python_Lib[legacy,pdf,rag]" --config-settings editable_mode=strict
```

--------------------------

装好之后，这台机器上运行程序必须带 `--no-sync`（否则 `uv run` 的隐式 sync
会再去装 PyQt6 又失败），或者 `export UV_NO_SYNC=1` 一劳永逸：

--------------------------

```
uv run --no-sync python xxx.py
```

--------------------------

**pip 方式（旧 glibc 机器，不带 pdf / rag 时可用）**：PyQt6 与 markitdown
在 pyproject 里已限定 `sys_platform == 'win32'`，Linux 的默认依赖集不含
它们，所以默认集与 `[legacy]` 组合可以用 pip 安装。要点有三：

1. **必须带 `--prefer-binary`，不要用一刀切的 `--only-binary=:all:`**。
   pip 本来就会排除标签高于本机 glibc 的 wheel，但对"新版本只发
   `manylinux_2_28` wheel、又发了 sdist"的编译型包（numpy / scipy 等），
   默认行为是转而现场编译 sdist 然后失败；`--prefer-binary` 让 pip 在
   "新版本的 sdist"与"旧版本的 wheel"之间选后者，自动退到最后一个还发
   旧 glibc wheel 的版本。不能用 `--only-binary=:all:` 的原因：本库依赖
   里有 jieba 这类**从头到尾只发 sdist 的纯 Python 包**（pyperclip 等
   长期无 wheel 的同类也算），会被它直接排除成 "versions: none" 报错
   （2026-08-09 在 BSCC-A 集群实测踩到 jieba）；这类包的现场构建只是
   纯 Python 打包、秒级完成，`--prefer-binary` 会正确放行。
2. **torch 要先从 CPU 专用索引按版本钉死装好**。`[legacy]` 在 Linux 上
   显式含 torch / torchvision，但 pip 读不到本库 `[tool.uv.sources]` 的
   按平台选索引机制，直接解析会从 PyPI 拿到带全套 nvidia 依赖的 CUDA
   构建（好几个 GB，CPU 集群纯属浪费）。先按 uv.lock legacy 分叉锁定的
   版本预装 CPU 构建，后续解析看到 `torch<2.7` 已满足就不会再动它。
3. **版本上限直接复用 `[legacy]` extra**——pip 同样认得 extras，上限清单
   与 uv 路线共用一处、不另行维护。

--------------------------

```
pip install --only-binary=:all: torch==2.6.0+cpu torchvision==0.21.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install --prefer-binary --editable "~/My_Program/Python_Lib[legacy]" --config-settings editable_mode=strict
```

--------------------------

pip 方式的边界：`[pdf]` / `[rag]` 组合在旧 glibc 上 pip **走不通**——pip
没有"跳过某个依赖"的机制，而 `[pdf]` 经 docling-parse（Python >= 3.11 无
旧 glibc wheel）、`[rag]` 经 chromadb → onnxruntime（同样无解）都会撞死，
这两种组合一律用上面的 uv 两步法（`--no-install-package` 跳过无解包）。

pip 与 uv 两条路线怎么选：**uv 仍是首选**——uv.lock 的 legacy 分叉给出
逐版本锁定、可复现的环境；pip 是安装时求解，装出的版本会随时间漂移，
装完应 `pip freeze` 留档。pip 路线适合两种场景：项目本身不用 uv 管理；
或者只想要一个与本库全家桶解耦的最小环境（见下）。另注：pip 路线的解析
可行性有 2026-07 那次 `uv pip compile --python-platform x86_64-manylinux_2_17`
全闭包探测背书；2026-08-09 在 BSCC-A（glibc 2.17）首次实测，当场修正了
一处（sdist-only 的 jieba 被 `--only-binary=:all:` 误杀，已改为
`--prefer-binary`）。再撞到个别包时按同样思路处理：编译型包给 `[legacy]`
补上限，纯 Python 包确认 `--prefer-binary` 会放行。

**最小环境的锁定做法（可选）**：想"只装某个小闭包（例如 NN 训练那十来个
包）、又要锁定可复现"时，在任意新机器（含 Windows）上用 uv 替旧平台预
编译一份锁定的 requirements 文件，再拿到旧机器上安装：

--------------------------

```
uv pip compile --python-platform x86_64-manylinux_2_17 --python-version 3.13 --only-binary :all: requirements.in -o requirements_glibc217.txt
pip install --only-binary=:all: -r requirements_glibc217.txt
```

--------------------------

（torch 仍需按上面的方式先从 CPU 索引预装，或在 requirements.in 里写明
索引。这份文件与 pyproject 是两份声明、会漂移，只在确有最小环境需求时
使用。2026-07 探测 `[legacy]` 上限用的正是这一机制。闭包里若含 jieba /
pyperclip 这类 sdist-only 纯 Python 包，`--only-binary` 别写 `:all:`，
改为逐包列出编译型包。）

## 三、Python_Lib 仓库自身装环境

**Windows**：本机日常的全功能环境（PDF 管线 + RAG + 训练）用：

--------------------------

```
uv sync --extra pdf --extra torch --extra rag
```

--------------------------

torch 按 pyproject 的 `[tool.uv.sources]` 自动装 CUDA 版。注意：纯 `uv sync`
得到的是**不含 torch 的最小环境**，并且会把环境里已装的 torch / docling /
FlagEmbedding 等卸载掉——在本机维护全功能环境时不要裸跑 `uv sync`。

**Linux**：先查 glibc 版本，再按情况选命令。

--------------------------

```
getconf GNU_LIBC_VERSION
```

--------------------------

glibc >= 2.28（Rocky/Alma 8+、Ubuntu 20.04+ 等），最小安装（不含 torch）：

--------------------------

```
uv sync
```

--------------------------

同样是 glibc >= 2.28，需要 PDF 读取 / 识别（docling 转换）、RAG 索引 / 检索、
torch 训练时按需选一行加 extras，可任意组合（docling / FlagEmbedding 会连带
拉入 CPU 版 torch）：

--------------------------

```
uv sync --extra pdf
uv sync --extra rag
uv sync --extra torch
uv sync --extra pdf --extra rag
uv sync --extra pdf --extra rag --extra torch
```

--------------------------

glibc < 2.28（CentOS 7 系 HPC 登录节点）：启用 legacy extra（torch 回退 2.6.0 等），
并跳过 PyQt6 / onnxruntime 等无解的包。注意 legacy 本身仍显式带 torch——老 glibc
HPC 上的转换 / 索引管线需要它。需要 PDF / RAG 功能时加 `--extra pdf` /
`--extra rag` 透传：

--------------------------

```
bash uv_sync_old_glibc.sh
bash uv_sync_old_glibc.sh --extra pdf
bash uv_sync_old_glibc.sh --extra rag
bash uv_sync_old_glibc.sh --extra pdf --extra rag
```

--------------------------

旧 glibc 机器上运行程序必须带 `--no-sync`，否则 `uv run` 隐式 sync 会回装装不上的新版本：

--------------------------

```
uv run --no-sync python xxx.py
```

--------------------------

（细节见 pyproject.toml 的 `[project.optional-dependencies]` 注释）

## 四、注册本仓库自带的 MCP server（用户级，一次性）

本仓库自带的 MCP server **一律注册到用户级** `~/.claude.json`（任何项目的
Claude Code 会话都能用），用 `.venv/Scripts/python.exe` 直接运行 server
文件（不经 `uv run`，避免隐式 sync 卸掉 torch extras）。装好本仓库环境
（上面第三节）之后，依次执行：

--------------------------

```
claude mcp add --scope user comp-chem-mcp-lyh -- E:/My_Program/Python_Lib/.venv/Scripts/python.exe E:/My_Program/Python_Lib/src/Chem_Lib/Comp_Chem_MCP_Server.py
claude mcp add --scope user hpc-mcp-lyh -- E:/My_Program/Python_Lib/.venv/Scripts/python.exe E:/My_Program/Python_Lib/src/HPC_Lib/HPC_MCP_Server.py
```

--------------------------

确认注册成功（应显示 Connected；已在运行的会话看不到新工具，开新会话即可）：

--------------------------

```
claude mcp list
```

--------------------------

当前的 server 清单：

| server 名 | 实现文件 | 功能 |
|---|---|---|
| `comp-chem-mcp-lyh` | `src/Chem_Lib/Comp_Chem_MCP_Server.py` | Gaussian 输入构建 / 几何提取 / 路由区段编辑 / 输入输出检查 / 本机 Multiwfn 会话 |
| `hpc-mcp-lyh` | `src/HPC_Lib/HPC_MCP_Server.py` | HPC 远程操作：文件传输（SFTP 或集群自带的专用高速通道，自动选择）、远端权限收紧、远程命令、SLURM 队列 / 提交 / 取消 / 作业与节点信息 |

**约定**：以后在本仓库新增任何 MCP server，都必须注册到用户级，并把注册
命令补进本节、同步更新仓库根 `CLAUDE.md` 的"MCP server 注册约定"清单。
Claude 会在会话启动时用 `claude mcp list` 核对该注册的是否都注册了，
发现缺失会提示补注册。
