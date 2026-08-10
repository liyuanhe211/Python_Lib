# Lit_Retrieval_0 文献检索流水线手册

不依赖 Zotero 的完整文献处理流程：**引用目录解析 → 文献信息搜索补全 →
文献下载 → 标准化重命名 → PDF 文本化**。核心检索与下载逻辑移植自
Zotero 插件项目 `Zotero_MultiFetcher`（`E:\My_Program\Zotero_MultiFetcher`，
仅作参考，未被修改）。

---

## 一、需求整理（原始要求的存档）

以下是本流水线立项时的需求叙述（整理版，2026-07-26）：

1. **引用目录解析**（`Lit_Retrieval_1_Reference_List_Parsing.py`）：
   使用 Zotero_MultiFetcher 项目中的提示词，结合 `LLM_Lib/LLM.py` 的语言
   模型访问工具解析参考文献列表；默认解析模型为 claude-sonnet-4-6，
   本机有 Claude 订阅，不需要 API key。
2. **文献信息搜索补全**（`Lit_Retrieval_2_Metadata_Completion.py`）：
   利用项目中的方法（CrossRef 检索）把解析结果中缺失的文献信息补全——
   最重要的是 DOI，也包括其他引用信息；补全后按标准 ACS 格式列出识别到的
   引文列表供人工核对；每个文献各存一份容易导入 Zotero 等文献管理软件的
   RIS（不是把所有条目合并成一个文件），文件名与该文献的 PDF 文件名一致。
   已经下载并改名的，RIS 归档到「PDF texts」子文件夹（目录结构参考
   `Lit_Retrieval_5_PDF_to_Markdown.py`）；还没下载到的，RIS 留在下载目标
   文件夹下，供将来重新尝试下载。
3. **文献下载**（`Lit_Retrieval_3_Download_PDF.py`）：
   汇集 Zotero 官方软件本身的 PDF 下载方法和插件提供的所有下载方法，
   轮流尝试直到下载成功；要显示进度。**并行的是多篇文献，每篇文献名下的
   多个下载渠道必须串行**——否则会产生大量无意义的访问。下载前在目标
   存储位置的母文件夹（例如目标是 `E:\My_Program\Knowledge_Base_Chemistry\0 New Download`，
   就在 `E:\My_Program\Knowledge_Base_Chemistry\` 中）通过文件名列表查重——
   可以假设既有文件已经过 `Lit_Retrieval_4_Rename_Ref.py` 标准命名（注意文件名结尾的标题
   不一定完整）。已经下载过的不重复下载，直接进入下一步。
4. **标准化重命名**：下载后用 `Lit_Retrieval_4_Rename_Ref.py` 处理。`Lit_Retrieval_4_Rename_Ref.py` 被改造
   为也可以接受一个 RIS 文件——既然前面已经生成了 RIS，信息足够（包括
   期刊缩写）时直接改名，不需要再做语言模型访问（避免浪费）。
5. **文本化**：改名后用 `Lit_Retrieval_5_PDF_to_Markdown.py` 处理。
6. 所有产物（PDF、PDF 识别结果、RIS 等）存储在用户指定的位置（可参数
   传入，未指定时交互式询问），目录结构参考 `Lit_Retrieval_5_PDF_to_Markdown.py`。
7. 每个脚本既可以接收上一层脚本提供的信息自动化运行，也可以交互式询问
   所需信息运行；多行输入以某一行等于 `end` 结束。
8. 总驱动脚本 `Lit_Retrieval_0.py` 串联全流程，同样支持交互；已经
   下载完成的文献不重复下载。

---

## 二、流程总览

```
参考文献文本（粘贴 / 文件）
        │
        ▼
[1] 解析  Lit_Retrieval_1_Reference_List_Parsing.py
        │   语言模型（claude-sonnet-4-6，本机订阅）→ 结构化条目
        │   纯 DOI 列表输入直接构建条目，不访问语言模型
        ▼
[2] 补全  Lit_Retrieval_2_Metadata_Completion.py
        │   CrossRef 检索补 DOI + 标题 + 完整作者 + 期刊缩写等
        │   按 ACS 格式列出识别结果供人工核对
        │   每个条目各写一份 RIS：<workdir>/<将来的 PDF 文件名>.ris
        ▼
[3] 下载  Lit_Retrieval_3_Download_PDF.py
        │   先在库根目录（默认 workdir 的上一级）按文件名查重
        │   再按渠道轮询下载（详见下文）：文献之间并行、单篇内部串行
        │   书籍排在全部文章之后逐个处理
        ▼
[4] 改名  Lit_Retrieval_4_Rename_Ref.py（RIS 直读模式，免语言模型）
        │   单篇 RIS 从 <workdir>/ 迁入 PDF texts/<主干[:80]>/
        ▼
[5] 文本化  Lit_Retrieval_5_PDF_to_Markdown.py（Marker，幂等）
```

总驱动：`python -m Tools.Lit_Retrieval_0`。

全程状态记录在 `<workdir>/Lit_Retrieval_State.json`，重复运行自动
跳过已完成的条目与步骤，可安全断点续跑。

---

## 三、目录结构

以存储目录 `E:\...\Knowledge_Base_Chemistry\0 New Download` 为例：

```
0 New Download\
    Lit_Retrieval_State.json                 流水线状态（断点续跑依据）
    Reynolds「2014 - Macromolecules」Kerszulis, ....pdf   下载并标准命名后的 PDF
    Madsen「2000 - J. Anim. Ecol.」Silver Spoons ....ris  ← 还没下载到的那些文献
    PDF texts\
        Reynolds「2014 - Macromolecules」Kerszulis, ...\   ← 主干截断到 80 字符
            Reynolds「2014 - Macromolecules」....ris      单篇 RIS
            Reynolds「2014 - Macromolecules」....md       Lit_Retrieval_5_PDF_to_Markdown 产出
            _page_3_Figure_1.png 等                        提取的图片
            .all_pages_processed                           文本化完成标记
```

**RIS 一律「一个文献一份」，没有合并的 `References.ris`。** 每份 RIS 的文件名
与该文献的 PDF 文件名一致（同由 `expected_pdf_filename` 决定），存放位置随
下载状态迁移：

| 条目状态 | RIS 位置 |
|---|---|
| 尚未拿到 PDF（含下载失败） | `<workdir>/<标准名>.ris` |
| 已下载并改名 | `<workdir>/PDF texts/<主干[:80]>/<主干[:80]>.ris` |
| 库中已有既有副本 | 不留 RIS（本来就不用下载） |

下载失败的条目把 RIS 留在工作目录，正是为了**将来重跑时免去重新解析与
CrossRef 检索**——直接凭这份 RIS 继续尝试下载。迁移由
`Lit_Retrieval_Common.sync_pending_ris` 统一负责，在第 2、3、4 步末尾各调用
一次。

「PDF texts / 主干截断 80 字符 / 完成标记」这套目录契约的唯一定义处是
`LLM_Lib/RAG_Lib/Docling.py`，本流水线与 `Lit_Retrieval_5_PDF_to_Markdown.py`、RAG 流水线
互认产物，绝不重复转换。

---

## 四、各脚本用法

所有脚本都支持两种运行方式：**命令行参数（自动化）** 与 **交互式询问**。
多行输入（参考文献文本、DOI 列表）都以某一行输入 `end` 结束。

### 第 1 步：引用列表解析

```
python -m Tools.Lit_Retrieval_1_Reference_List_Parsing
python -m Tools.Lit_Retrieval_1_Reference_List_Parsing ^
    --workdir "E:\...\0 New Download" --input-file refs.txt ^
    [--model claude-sonnet-4-6] [--format-hint numbered] [--append]
```

- 提示词逐字移植自插件的 `src/modules/LLMPrompts.ts`（系统提示 +
  用户消息模板 + 每批 25 行的分批规则）；
- 语言模型访问走 `LLM_Lib/LLM.py` 的 `call_claude()`（本机 `claude` CLI，
  订阅额度，无需 API key；调用结果有 SQLite 缓存，相同输入不重复计费）；
- 长列表分批并发解析（默认 4 批并发）；输出被截断时自动做 JSON 修复；
- 纯 DOI 列表输入（每行一个 DOI）直接构建条目，完全跳过语言模型；
- 结果写入状态文件；交互模式下发现既有条目会询问「覆盖还是追加」。

### 第 2 步：元数据补全（CrossRef）+ RIS 输出

```
python -m Tools.Lit_Retrieval_2_Metadata_Completion
python -m Tools.Lit_Retrieval_2_Metadata_Completion ^
    --workdir "E:\...\0 New Download" [--concurrency 10]
```

检索与校验逻辑移植自插件的 `src/modules/CrossRefResolver.ts`：

- 条目已带 DOI → 直接取 CrossRef 记录回填全部字段；语言模型提取的 DOI
  在 CrossRef 查无记录时判为幻觉/笔误，标记 `doi_not_in_crossref` 并取消
  选中（不参与后续下载，需人工复核）；
- 条目无 DOI → 结构化查询（`query.author` / `query.container-title` /
  `query.bibliographic` + 出版年份 ±1 过滤）优先，不理想再退回自由文本
  查询；候选按**标题（Jaccard 相似度，权重 2）**/ 年份 / 卷 / 第一作者姓
  （Levenshtein ≤ 2 容错）/ 页码或文章号 / 期刊名 Jaccard 相似度打分，
  **得分 ≥ 0.5 才采纳**；此外还有两道闸：
  - **标题否决**：引文与候选双方标题都「像样」（规范化后各有 ≥ 4 个有效词）
    而相似度 < `TITLE_VETO_SIMILARITY`（0.3）时，该候选直接出局；
  - **最低可比字段数**：可比字段数 < `MIN_COMPARABLE_FACTORS`（2）时得 0 分。

  这两道闸是 2026-07-26 补的。原先的打分完全不看标题，而最终得分是
  「各项得分之和 ÷ 可比字段数」——候选记录字段稀疏时（技术报告往往只有
  年份和作者，没有期刊 / 卷 / 页码），可比字段数小到 1~2 个，个别字段的
  偶然吻合就被放大成高分。实际发生过的误匹配：引文
  「LOOP, M. S. 1974. …. Herpetologica 30:123–127.」被匹配到
  `10.2172/4327023`（美国能源部《Thermoelectric size effect in noble metals》
  进度报告）——仅凭年份同为 1974 就拿到 0.500 分，正好卡在采纳门槛上，而两者
  标题相似度只有 0.083。修复后该候选得 0.295 分且被标题否决；这条引文改为
  「未找到足够可信的匹配」，比给一个错误 DOI 要好。
- 采纳后做事后校验（年份、卷、文章号、页码首页、作者、期刊），问题记入
  `crossref_issues`，结束时集中列出供人工复核；
- 刊头 / 封面 / 目录等「可疑标题」候选被直接排除；
- **期刊缩写取自 CrossRef 的 `short-container-title`**，写入 RIS 的 `J2`
  标签——这正是第 4 步免语言模型改名的关键字段；
- 补全结束后**按 ACS 格式列出全部引文**（`format_acs_citation`），逐条附上
  「未找到 DOI」「校验问题」「已取消选中」等提示，供人工一眼核对识别是否
  正确；
- 为**每个条目各写一份 RIS**（不是合并成一个 `References.ris`），文件名就是
  这篇文献将来的 PDF 文件名，落在 `<workdir>/` 下，可直接拖入 Zotero。

### 第 3 步：多渠道 PDF 下载

```
python -m Tools.Lit_Retrieval_3_Download_PDF
python -m Tools.Lit_Retrieval_3_Download_PDF ^
    --workdir "E:\...\0 New Download" [--concurrency 3] ^
    [--openalex-key XXX] [--email you@example.org] ^
    [--proxy socks5h://127.0.0.1:1080] [--library-root E:\...\Knowledge_Base_Chemistry] ^
    [--no-dedup]
```

交互模式下如果目录里还没有状态文件，可以直接粘贴 DOI 列表（一行一个，
`end` 结束）现场构建条目。

**下载渠道与尝试顺序**（文章条目；每个渠道失败自动换下一个）：

| 顺序 | 渠道 | 出处 |
|---|---|---|
| 1 | OpenAlex（`best_oa_location.pdf_url`，合法开放获取） | 插件 `A0_OpenAlex.ts`，插件把它作为内置前置渠道 |
| 2 | Unpaywall API（合法开放获取） | Zotero 官方「查找可用 PDF」的开放获取解析途径查询的就是 Unpaywall 数据 |
| 3 | DOI 落地页抓取（`https://doi.org/{DOI}` 重定向后的文章页 → `citation_pdf_url` 等标记） | Zotero 官方的页面抓取途径 |
| 4 起 | Anna's Archive / LibGen / Sci-Hub 镜像交错轮询：`annas-archive.pk → libgen.la → sci-hub.ee → annas-archive.gl → libgen.vg → sci-hub.red → …` | 插件 `A1/A2/A3` 与 `default-source-config.txt`，顺序按实测可用性调整过 |

镜像顺序原先与插件默认配置完全一致，2026-07-26 按实测结果调整：
`sci-hub.su` / `.st` / `.ru` / `.box` 已整体被 DDoS-Guard 或 Cloudflare 挡在
门外（对任何 DOI 都返回 HTTP 403 的「Checking your browser」挑战页，请求
根本没进站），而 `sci-hub.ee` 能正常返回文章页并给出 PDF 直链，因此把
`.ee` 提到 Sci-Hub 系列最前面，实测同样可用的 `annas-archive.pk` 一并提前。
被挡住的镜像仍保留在清单里——反爬策略随时间变化，留着继续参与轮询没有坏处。

各镜像渠道的具体流程（与插件实现一致）：

- **Anna's Archive**：`/scidb/{DOI}/` 页面 → 提取含 `/d3/x/` 的直链
  （链接约 2.8 小时过期，每次现取）；`/scidb/` 只认它入库时用的那个 DOI，
  直接命中不了时再走一次别名桥接：`/search?index=journals&q={DOI}` →
  取 md5 详情页 → 从详情页读出它收录时用的 DOI → 重新 `/scidb/`。
  它的检索是按元数据建的索引，同一篇文献的两个 DOI 都能搜到同一条记录；
- **LibGen**：`index.php?req={DOI}&columns=doi` 搜索 → `edition.php` →
  `ads.php`（取动态 key）→ `get.php`（**必须带指向 ads.php 的 Referer 头**，
  否则服务器返回 500）→ 307 重定向到 CDN；
- **Sci-Hub**：镜像页 → `#pdf` 的 embed/iframe、`<object data=…>`、或
  `citation_pdf_url` meta；相对链接按页面 URL 解析并强制 https。

未命中原因分三类，日志里写得不一样，排查时先看是哪一类：

- **「被 DDoS-Guard / Cloudflare 反爬拦截，请求没能进站」**——请求连站点
  都没进到，换镜像或挂代理才有意义；算整站级故障，计入拉黑；
- **「未收录该文献」**——该站确实没有这一篇，镜像本身是好的，不计入故障；
- **「Sci-Hub 返回人机验证页」**——两种可能都有（多数情况是未收录，也可能
  是被限流），从页面上分不出来，因此**不**计入整站级故障。这一条是刻意的：
  Sci-Hub 对它没收录的 DOI 就返回这个页面，若按故障计，几篇没收录的文献就
  能把当前唯一可用的镜像误拉黑。

**没有 DOI 的条目：按标题检索（覆盖 JSTOR 上的旧刊）**

CrossRef 并非无所不包。实测 `Herpetologica` 1974 年那一卷在 CrossRef 里
**一条记录都没有**，所以第 2 步无论怎么检索都补不出
「LOOP, M. S. 1974. …. Herpetologica 30:123–127.」的 DOI，条目最终没有 DOI。
早先的第 3 步遇到没有 DOI 的条目直接放弃；现在会先按标题检索
（`find_doi_by_title()`）再走渠道。

检索走的是 **Anna's Archive 的期刊索引**（`/search?index=journals&q={标题}`），
不是 JSTOR 本身——JSTOR 没法程序化访问，见下。解析只用检索结果页：每条结果
由两个共享同一 md5 的块组成，一块是文件路径（形如 `scihub/10.2307/3892027.pdf`，
标识符直接写在路径里），另一块是「标题 | 作者 | 期刊、卷、页码、年份 | 摘要」，
所以不必逐条打开约 240 KB 的详情页。候选按标题 Jaccard 相似度排序，年份明确
冲突的直接排除，相似度低于 `TITLE_SEARCH_SIMILARITY`（0.6）时宁可返回空也不
给一个可能错的标识符。实测真正命中时相似度在 0.9 以上，不相干的结果落在 0.4
以下，中间安全带很宽。

**关于 JSTOR，有三点必须说清楚：**

1. **JSTOR 自己抓不了。** 文章页 `jstor.org/stable/<id>` 与检索页
   `jstor.org/action/doBasicSearch` 都返回 HTTP 403 的 reCAPTCHA
   「Access Check」页（首页和 `robots.txt` 倒是 200，但 `robots.txt` 明确
   `Disallow: /action` 和 `Disallow: /api`）。所以「直接在 JSTOR 里检索标题」
   这条路走不通，只能绕道 Anna's Archive 的索引。
2. **JSTOR 的 stable ID 与 `10.2307/<stable ID>` 一一对应**，粘贴
   `https://www.jstor.org/stable/3892027?seq=1` 这样的链接会被
   `match_dois()` 自动换算，与粘贴 DOI 等效（`is_pure_doi_input()` 也认，
   因此不会误触发语言模型解析）。
3. **换算出来的东西不一定是注册过的 DOI。** JSTOR 为它的很大一部分内容
   注册过 DOI，但并非全部。实测对比：

   | 标识符 | doi.org | CrossRef | DataCite | 注册机构查询 |
   |---|---|---|---|---|
   | `10.2307/1563325` | 302 重定向到 JSTOR | 有记录 | 无 | `"RA": "Crossref"` |
   | `10.2307/3892027` | 404 | 404 | 404 | `"status": "DOI does not exist"` |

   但 Sci-Hub / LibGen / Anna's Archive 一律按这个拼出来的标识符收录 JSTOR
   的内容（Anna's Archive 的 SciDB 页面上就明写着「DOI: 10.2307/3892027」），
   所以**拿它去下载可行，拿它去查元数据则可能一无所获**。因此第 2 步对
   `10.2307/<数字>` 这类标识符网开一面：查不到 CrossRef 记录时既不标记
   `doi_not_in_crossref` 也不取消选中（见 `is_jstor_identifier()`），否则
   这类条目会在下载前就被踢掉。

**同一篇文献的多个 DOI（别名回退）**：出版社更迭时，同一篇文献常在新旧两个
前缀下各注册过一条 CrossRef 记录。两条记录彼此独立、没有任何字段互相指向
（`relation` 为空，`alternative-id` 只有自己），OpenAlex 里也是两条独立的
work；而各下载渠道只收录其中一个 DOI。于是出现「DOI 完全正确、渠道也没坏、
却全渠道扑空」这种最难排查的失败。因此本条目自己的 DOI 走完全部渠道仍失败
时，`find_alias_dois()` 会再找一轮别名，并用别名把整条渠道链重跑：

- **(a) 前缀互换**：Wiley / Blackwell 系的 `10.1046` ↔ `10.1111` ↔ `10.1034`
  之间后缀原样保留（Blackwell / Munksgaard 的 `j.<ISSN>.<年>.<号>.x` 命名
  在并入 Wiley 后被整体沿用），换完再用 CrossRef 确认该 DOI 真实存在。
  只对 `j.` 开头的后缀尝试——别的命名风格换了前缀也不会存在，白费请求；
- **(b) 标题检索**：拿标题去 CrossRef `query.bibliographic` 检索，取标题
  Jaccard 相似度 ≥ `ALIAS_TITLE_SIMILARITY`（0.85）的其他 DOI。

两条途径互补：(a) 零成本但只覆盖 Wiley / Blackwell，(b) 通用。经别名下到的
条目把实际使用的 DOI 记进 `status.download_doi`，条目自己的 `doi` 字段不改
（RIS 已按原 DOI 写出，改了会不一致）。实测的两个例子：

| 引文 | CrossRef 给的 DOI | 各渠道收录的 DOI | 靠哪条途径救回 |
|---|---|---|---|
| Madsen & Shine 2000, J. Anim. Ecol. 69:952–958 | `10.1046/j.1365-2656.2000.00477.x` | `10.1111/j.1365-2656.2000.00477.x` | (a) 前缀互换 |
| King et al. 1999, J. Zool. 247:19–28 | `10.1017/s0952836999001028` | `10.1111/j.1469-7998.1999.tb00189.x` | (b) 标题检索 |

其他行为：

- **查重**：下载前在库根目录（默认 `workdir` 的上一级，可用
  `--library-root` 覆盖）递归收集全部 PDF 文件名，按 Lit_Retrieval_4_Rename_Ref 标准命名
  `{人名}「{年份} - {期刊缩写}」{其余}` 解析后与条目匹配。匹配规则：作者
  姓氏必须出现在文件名中；双方年份已知时必须一致；标题按词集合重叠度
  匹配（分母取文件名一侧——**文件名中的标题可能被截断**，不能要求覆盖
  全标题）；条目无标题时退而要求期刊缩写一致。命中则标记
  `already_exists` 并跳过下载。
- **并行的粒度是「文献」，不是「渠道」**：默认 3 篇文献同时下载，但同一篇
  文献名下的各个渠道**严格串行**——某个渠道拿到 PDF 就立刻收工，后面的渠道
  一次也不访问。
- **全部文献共用一份渠道顺序与一份渠道健康档案**（`SourceHealth`）：
  - 某镜像**成功过** → 排到镜像段的最前面，后续文献优先走这条已被证明
    可用的路。合法开放获取渠道（OpenAlex / Unpaywall / DOI 落地页）始终
    钉在更前面、不参与这个重排——它们快、便宜、给的是合法直链，某篇文献
    在某个镜像成功过，并不意味着下一篇就不该先问一遍有没有开放获取版本；
  - 某渠道累计 2 次**整站级故障**且从未成功过 → 本次运行内拉黑，后续文献
    直接跳过，并一次性播报「🚫 某某已连续 N 次整站级故障」；
  - 「渠道正常、只是没收录这一篇」（Anna's Archive 页面上没有下载链接、
    LibGen 搜不到这个 DOI、Sci-Hub 明说未收录）**不计入故障**——它对别的
    文献毫无预示作用，误拉黑会白白丢掉可用镜像；
  - `DOI 落地页`每次面对的是不同出版社的服务器，永不拉黑。
  这一套取代了早期「不同工作线程从镜像清单的不同起点轮询」的做法：起点
  旋转虽然分散了压力，却让**唯一能用的那个镜像在其余文献里被排到最后**，
  每篇文献都要先把十几个必然失败的镜像重试一遍。
- **书籍**（`book` / `bookSection`）：永不并行，放在全部文章之后逐个处理；
  走 LibGen 标题（+第一作者）搜索（无结果退回仅标题）→ `ads.php` →
  `get.php`；书籍下载一旦开始不设读超时。
- **内容校验**：下载首块必须含 `%PDF` 魔数、总大小 ≥ 4 KB，防止把报错
  页面存成 PDF；先写 `.part` 临时文件，校验通过才落盘。
- **进度**：每个条目每个渠道的尝试结果实时打印，并有「总进度 k/N」；
  每完成一条立即保存状态（中断后重跑不丢进度）。

### 第 4 步：标准化改名（Lit_Retrieval_4_Rename_Ref.py 的 RIS 直读模式）

`Lit_Retrieval_4_Rename_Ref.py` 新增能力（原有交互 / 右键菜单用法完全不变）：

- 对每个输入 PDF **自动发现 RIS**：同目录同名 `.ris`，或
  `PDF texts/<主干[:80]>/<主干[:80]>.ris`；
- RIS 中信息足够——第一作者姓（`AU`）、**期刊缩写（`J2`/`JA`；书籍类
  取出版社 `PB`）**、年份、标题——时直接生成标准文件名，
  **不访问语言模型**；信息不足自动退回原有语言模型流程；
- `--no-ris` 可禁用自动发现；
- 新增函数 `rename_pdf_from_ris(pdf_path, ris_path=None, info=None,
  assume_yes=True, allow_llm_fallback=True)` 供驱动脚本调用。

总驱动的做法：先把条目的 RIS 写成 PDF 同目录同名 `.ris` → 调
`rename_pdf_from_ris` 改名 → 成功后把单篇 RIS 归档到
`PDF texts/<新主干[:80]>/` 并删除临时 `.ris`。

### 第 5 步：文本化（Lit_Retrieval_5_PDF_to_Markdown.py）

改名后的 PDF 交给 `convert_multiple_pdf_to_markdown()`（Marker + Surya
OCR）。转换幂等：已有 `.all_pages_processed` 完成标记的自动跳过。
总驱动加 `--skip-markdown` 可跳过本步（Marker 依赖较重，首次运行会下载
模型权重）。

### 总驱动

```
python -m Tools.Lit_Retrieval_0
python -m Tools.Lit_Retrieval_0 --workdir "E:\...\0 New Download" --input-file refs.txt
python -m Tools.Lit_Retrieval_0 --workdir "E:\...\0 New Download" --resume
```

- 交互模式：询问存储目录；目录中已有状态文件时询问「继续处理既有条目 /
  输入新列表覆盖 / 输入新列表追加」；
- `--resume`：直接复用既有条目断点续跑；
- `--confirm-renames`：改名逐一人工确认（默认自动确认，但改名记录都会
  写入 `rename_history.csv` 与 `0_Renaming_Records.json`，可回溯）；
- 其余参数与第 3 步一致（`--proxy`、`--openalex-key`、`--no-dedup` 等）。

---

## 五、实现过程中的补充说明

以下内容是实现时确定的细节，原始需求中未明确，特此记录：

1. **共享模块**：新增 `Lit_Retrieval_Common.py` 承载各脚本共用的
   基础设施（reference 字典规范、状态文件读写、DOI 正则、RIS 读写、
   标准命名解析与查重、相似度函数、`end` 结尾的多行输入）。
   `Lit_Retrieval_4_Rename_Ref.py` 的 RIS 解析也从这里导入。
2. **第 2 步的 RIS 存放时机**：第 2 步运行时 PDF 还不存在，但**最终文件名
   是可以预先算出来的**——`Lit_Retrieval_Common.expected_pdf_filename` 直接
   复用第 4 步的命名实现（`Lit_Retrieval_4_Rename_Ref.standard_filename_from_ris_entry`）。
   因此第 2 步就为每个条目按最终文件名各写一份 RIS 放在 `<workdir>/` 下，
   下载成功并改名后再由 `sync_pending_ris` 迁到
   `PDF texts/<新主干[:80]>/`，下载失败的则留在原地。
3. **下载即用最终文件名**：既然文件名可以预先算出，下载阶段就直接以标准名
   落盘（与占位 RIS 同名），第 4 步改名通常只是确认一句「文件名无需更改」。
   RIS 直读所需的四个字段（第一作者姓 / 期刊缩写 / 年份 / 标题）有缺失时——
   此时第 4 步本来就要退回语言模型——退回「{第一作者姓} - {年份} -
   {标题前 80 字符}.pdf」的临时名（信息再不足则退回 DOI 命名）。
4. **Zotero 官方下载方法的对应关系**：Zotero「查找可用 PDF」有两条途径——
   开放获取解析器（官方经自家服务端查 Unpaywall 数据；本流水线直接调
   Unpaywall 公开 API，另外加了 OpenAlex）与 DOI 落地页抓取
   （`citation_pdf_url`；本流水线用同样的 meta 标记提取实现）。
5. **书籍支持是简化移植**：插件的书籍搜索（`A4_LibGenBook.ts` /
   `A5_AnnasArchiveBook.ts`）含多页抓取、模糊打分排名、镜像优先级、
   Anna's Archive 兜底等完整逻辑；本流水线目前实现的是 LibGen 标题搜索
   取第一个下载入口的简化版，未实现 Anna's Archive 书籍兜底与按页数 /
   文件大小选择版本。批量书籍建议仍用 Zotero 插件处理。
6. **HTML 解析用正则而非 DOM**：镜像页面结构简单（找特定 href 模式），
   用正则避免引入 BeautifulSoup 依赖；页面结构变化时需要同步更新正则。
7. **第三方依赖**：`requests`（第 2、3 步网络访问）；`--proxy` 用 SOCKS5
   时需要 `requests[socks]`（PySocks）；第 5 步需要 `marker-pdf`；第 1、
   4 步的语言模型走本机 `claude` CLI。
8. **联系邮箱**：CrossRef / Unpaywall / OpenAlex 请求中默认使用与参考插件
   一致的公开占位邮箱 `zotero-multifetcher@github.com`（可用 `--email`
   覆盖）。按惯例不把个人邮箱写进仓库代码。
9. **OpenAlex API key** 可选：`--openalex-key` 参数、环境变量
   `OPENALEX_API_KEY`、或 `E:\My_Program\LLM_API_KEYS_PRIVATE.py` 中的
   `OPENALEX_API_KEY` 变量；没有 key 时单条查询仍可用（限速更严）。
10. **失败重试**：下载失败的条目状态保持 `failed`，重新运行第 3 步或总
    驱动会自动重试（成功与已存在的条目不会重复下载）。镜像可用性随时间
    变化，隔天重试往往有效；镜像清单硬编码在
    `Lit_Retrieval_3_Download_PDF.py` 的 `MIRROR_SOURCES`，失效时
    在那里增删。
11. **查重的局限**：查重只看文件名（不读文件内容、不比对 DOI），依赖
    既有文件是标准命名；同一文献在库中用非标准名存放时会漏判；作者重名
    加标题高度相似时理论上可能误判——命中时会打印既有文件路径，必要时
    人工核对。
12. **`doi_not_in_crossref` 的条目**不参与下载（`selected=false`），总结
    时会列出，需要人工确认 DOI 后把状态文件中的 `doi` 改对并把
    `selected` 改回 `true` 再重跑。
