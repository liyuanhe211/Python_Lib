# -*- coding: utf-8 -*-
"""
Lit_Retrieval_3_Download_PDF.py — 多渠道轮询下载文献 PDF

══════════════════════════════════════════════════════════════════════════════
  文献检索流水线第 3 步：对每个已确定 DOI 的条目，轮流尝试多个下载渠道，
  直到下载成功。渠道分两类：

  ─ Zotero 官方「查找可用 PDF」使用的两条途径（移植其思路）─
  1. 开放获取解析器：Unpaywall API（Zotero 官方经由自家服务端查询的
     就是 Unpaywall 的数据）；本脚本另外加上 OpenAlex（参考插件的内置
     前置渠道），两者都返回合法的开放获取 PDF 直链；
  2. DOI 落地页抓取：加载 https://doi.org/{DOI} 重定向后的文章页面，
     从 <meta name="citation_pdf_url"> 等标记中提取出版社提供的 PDF 链接
     （Zotero 官方的页面抓取途径）。

  ─ Zotero_MultiFetcher 插件提供的全部渠道（逐一移植）─
  3. Anna's Archive：/scidb/{DOI}/ 页面 → "/d3/x/" 直链（A1_AnnasArchive.ts）；
  4. LibGen：index.php 搜 DOI → edition.php → ads.php → get.php → CDN
     （A2_LibGen.ts，get.php 需要带 Referer 头）；
  5. Sci-Hub：镜像页面 → #pdf embed / object / citation_pdf_url
     （A3_SciHub.ts）。
  镜像清单与尝试顺序与插件默认配置一致（Anna's / LibGen / Sci-Hub 交错，
  失败自动换下一个镜像）。

  其他要点：
  - 下载前在库根目录（默认取工作目录的上一级，如存储位置是
    「…\\Knowledge_Base_Chemistry\\0 New Download」则在
    「…\\Knowledge_Base_Chemistry\\」）递归收集既有 PDF 文件名查重；
    既有文件按 Lit_Retrieval_4_Rename_Ref.py 标准命名解析后与条目模糊匹配（文件名中的
    标题可能被截断），命中则跳过下载；
  - 并行的粒度是「文献」，不是「渠道」：多个文献同时下载（默认 3 并发），
    但同一个文献名下的各个渠道严格串行——某个渠道拿到 PDF 就立刻收工，
    后面的渠道一次也不访问；
  - 全部文献共用同一份渠道顺序与一份「渠道健康档案」（SourceHealth）：
    某个镜像一旦成功过就被提到镜像段的最前面优先尝试（合法开放获取渠道
    始终钉在更前面，不参与这个重排），某个镜像连续出现整站级故障
    （HTTP 403 / 超时 / 版式失效）就在本次运行内拉黑、后续文献直接跳过。
    这样避免每篇文献都把同一批必然失败的镜像重试一遍；
  - 书籍条目（book / bookSection）永不并行，放在全部文章之后逐个处理
    （LibGen 标题搜索）；
  - 下载直接落盘为该文献的标准文件名（与占位 RIS 同名），第 4 步改名通常
    只是确认一下；
  - 全程显示进度；下载内容校验 %PDF 魔数，防止把错误页面存成 PDF。

用法：
    # 交互模式（询问目录；无状态文件时可直接粘贴 DOI 列表，一行一个，end 结束）
    python -m Tools.Lit_Retrieval_3_Download_PDF

    # 自动化模式
    python -m Tools.Lit_Retrieval_3_Download_PDF ^
        --workdir "E:\\My_Program\\Knowledge_Base_Chemistry\\0 New Download"
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import concurrent.futures
import html as html_module
import os
import re
import sys
import threading
from pathlib import Path
from urllib.parse import quote, urljoin

import requests

from Tools.Lit_Retrieval_Common import (
    BROWSER_USER_AGENT,
    CONTACT_EMAIL,
    ask_workdir,
    build_library_index,
    expected_pdf_filename,
    find_existing_pdf_for_reference,
    is_book_reference,
    jaccard_similarity,
    load_state,
    match_dois,
    new_reference,
    new_state,
    normalize_doi,
    normalize_reference,
    read_multiline_until_end,
    ref_short_label,
    save_state,
    sync_pending_ris,
    unused_path,
)

# ── 镜像清单（与参考插件 default-source-config 的顺序一致）───────────────────

#: 交错排列的镜像渠道：(渠道类型, 域名)。前面是主力镜像，后面是备用镜像。
#:
#: 2026-07-26 调整过顺序：实测 sci-hub.su / .st / .ru 已整体被 DDoS-Guard 挡在
#: 门外（对任何 DOI 都返回 HTTP 403 的「Checking your browser」页），而
#: sci-hub.ee 能正常返回文章页并给出 PDF 直链，因此把 .ee 提到 Sci-Hub 系列的
#: 最前面；annas-archive.pk 同样实测可用，也一并提前。被挡住的镜像仍然保留在
#: 清单里——反爬策略随时间变化，留着让它们继续参与轮询没有坏处。
MIRROR_SOURCES: list[tuple[str, str]] = [
    ("annas-archive", "annas-archive.pk"),
    ("libgen", "libgen.la"),
    ("scihub", "sci-hub.ee"),
    ("annas-archive", "annas-archive.gl"),
    ("libgen", "libgen.vg"),
    ("scihub", "sci-hub.red"),
    ("annas-archive", "annas-archive.gd"),
    ("libgen", "libgen.bz"),
    ("scihub", "sci-hub.st"),
    ("libgen", "libgen.gl"),
    ("scihub", "sci-hub.su"),
    ("scihub", "sci-hub.ru"),
    ("scihub", "sci-hub.box"),
    ("scihub", "sci-hub.ren"),
]

PAGE_TIMEOUT = 30          # 页面 / API 请求超时（秒）
DOWNLOAD_TIMEOUT = 300     # PDF 下载读超时（秒）；书籍不设读超时（可能很慢）
MIN_PDF_BYTES = 4096       # 小于此大小的"PDF"视为错误页面
DEFAULT_ARTICLE_CONCURRENCY = 3

#: 一个渠道累计出现这么多次整站级故障后，本次运行内不再尝试它
SITE_FAILURE_BLACKLIST_THRESHOLD = 2

#: 这些渠道每次面对的是不同出版社的服务器，一次失败说明不了下次也失败，
#: 因此永不拉黑
_NEVER_BLACKLIST = {"DOI 落地页"}

#: 合法开放获取渠道，永远排在影子图书馆镜像之前，不参与「成功过的优先」重排。
#: 它们快、便宜、而且给的是合法直链，某篇文献在某个镜像成功过，并不意味着
#: 下一篇文献就不该先问一遍 OpenAlex / Unpaywall 有没有开放获取版本。
_PRIMARY_SOURCES = {"OpenAlex", "Unpaywall", "DOI 落地页"}

_MOBILE_USER_AGENT = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 11_3_1 like Mac OS X) "
    "AppleWebKit/603.1.30 (KHTML, like Gecko) Version/10.0 Mobile/14E304 Safari/602.1"
)

#: Sci-Hub「文献不存在」的页面提示（英文 / 俄文），移植自 A3_SciHub.ts
_SCIHUB_NOT_AVAILABLE = [
    re.compile(r"Please try to search again using DOI", re.IGNORECASE),
    re.compile(r"статья не найдена в базе", re.IGNORECASE),
]

#: 反爬中间层的挑战页特征。识别出来是为了把日志里的原因写准——原先这类页面
#: 一律报成「页面上没有 PDF 链接」，会让人误以为是解析正则失效或该站没收录，
#: 实际上请求根本没进到站点里面。
_ANTI_BOT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"DDoS-Guard", re.IGNORECASE), "被 DDoS-Guard 反爬拦截，请求没能进站"),
    (re.compile(r"Checking your browser", re.IGNORECASE), "被反爬中间层拦截，请求没能进站"),
    (re.compile(r"Just a moment", re.IGNORECASE), "被 Cloudflare 拦截，请求没能进站"),
    (re.compile(r"Enable JavaScript and cookies to continue", re.IGNORECASE),
     "被反爬中间层拦截，请求没能进站"),
]

#: Sci-Hub 的人机验证页。它既可能表示「本站没收录这一篇」，也可能表示「你被
#: 限流了」，从页面上分不出来，因此不当成整站故障——否则会把当前唯一能用的
#: 镜像因为几篇没收录的文献误拉黑。
_SCIHUB_VERIFICATION = [
    re.compile(r"<title>\s*Verification\s*[-–—]\s*Sci-Hub", re.IGNORECASE),
    re.compile(r"are you (?:are )?a?\s*robot", re.IGNORECASE),
]


def _anti_bot_reason(page: str) -> "str | None":
    """页面是反爬挑战页时返回原因描述，否则返回 None。"""
    for pattern, reason in _ANTI_BOT_PATTERNS:
        if pattern.search(page):
            return reason
    return None

# OpenAlex API key：可选，从 LLM_API_KEYS_PRIVATE.py 或环境变量读取。
# 没有 key 时 OpenAlex 单条查询仍可用，只是限速更严。
#
# 这里有两个都会导致密钥静默丢失的坑，缺一不可地都要处理：
#   1. 密钥文件所在目录不在 sys.path 上。只有 ``LLM_Lib.LLM`` 在导入时会顺手把它
#      加进去，而本模块的导入链走不到那里，所以必须自己找一遍。
#   2. 密钥文件里的变量名是驼峰的 ``OpenAlex_API_KEY``，不是全大写。
# 两个坑都表现为"导入失败 → 退回空字符串 → 整条流水线一直在无密钥状态下调用
# OpenAlex"，全程没有任何报错。

def _find_api_keys_dir() -> "str | None":
    """定位存放 ``LLM_API_KEYS_PRIVATE`` 的目录，规则与 ``LLM_Lib.LLM`` 一致。"""
    env_dir = os.environ.get("LLM_API_KEYS_DIR")
    if env_dir and Path(env_dir).is_dir():
        return env_dir
    for parent in Path(__file__).resolve().parents:
        if parent.name == "Python_Lib" and (parent / "src").is_dir():
            return str(parent.parent)
    return None


_keys_dir = _find_api_keys_dir()
if _keys_dir and _keys_dir not in sys.path:
    sys.path.insert(0, _keys_dir)

try:
    from LLM_API_KEYS_PRIVATE import OpenAlex_API_KEY as _OPENALEX_KEY  # type: ignore[import-untyped]
except ImportError:
    try:
        from LLM_API_KEYS_PRIVATE import OPENALEX_API_KEY as _OPENALEX_KEY  # type: ignore[import-untyped]
    except ImportError:
        _OPENALEX_KEY = ""
OPENALEX_API_KEY: str = _OPENALEX_KEY or os.environ.get("OPENALEX_API_KEY", "")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  HTTP 会话                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def make_session(proxy: "str | None" = None) -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": BROWSER_USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/pdf,*/*;q=0.8",
    })
    if proxy:
        session.proxies = {"http": proxy, "https": proxy}
    return session


def _normalize_base(url: str) -> str:
    url = url.strip()
    if not url.startswith("http"):
        url = f"https://{url}"
    if not url.endswith("/"):
        url += "/"
    return url


def _unescape_href(href: str) -> str:
    return html_module.unescape(href)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  各渠道的 PDF 直链获取（返回 dict(success, pdf_url, referer, error)）        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _ok(pdf_url: str, referer: "str | None" = None) -> dict:
    return {"success": True, "pdf_url": pdf_url, "referer": referer,
            "error": None, "site_down": False}


def _fail(error: str, site_down: bool = False) -> dict:
    """一次未命中。

    site_down=True 表示「这个渠道本身出问题了」（整站 403 / 超时 / 页面版式
    失效），对其他文献同样无望，可以据此拉黑；site_down=False 表示「渠道正常，
    只是没收录这一篇」，换一篇文献时仍应继续尝试。
    """
    return {"success": False, "pdf_url": None, "referer": None,
            "error": error, "site_down": site_down}


def fetch_from_openalex(doi: str, session: requests.Session,
                        api_key: str = "") -> dict:
    """OpenAlex：best_oa_location.pdf_url（移植自 A0_OpenAlex.ts）。"""
    try:
        url = (f"https://api.openalex.org/works/doi:{quote(doi, safe='')}"
               f"?select=best_oa_location")
        if api_key:
            url += f"&api_key={quote(api_key)}"
        response = session.get(url, timeout=PAGE_TIMEOUT,
                               headers={"Accept": "application/json"})
        if response.status_code == 404:
            return _fail("OpenAlex 无此 DOI 记录")
        if response.status_code != 200:
            return _fail(f"OpenAlex HTTP {response.status_code}",
                         site_down=response.status_code in (403, 429)
                         or response.status_code >= 500)
        location = (response.json() or {}).get("best_oa_location")
        if not location:
            return _fail("OpenAlex 无开放获取版本")
        pdf_url = location.get("pdf_url")
        if pdf_url:
            return _ok(pdf_url)
        return _fail("OpenAlex 有记录但无 PDF 直链")
    except (requests.RequestException, ValueError) as e:
        return _fail(f"OpenAlex 出错：{e}", site_down=True)


def fetch_from_unpaywall(doi: str, session: requests.Session,
                         email: str = CONTACT_EMAIL) -> dict:
    """Unpaywall：Zotero 官方开放获取解析途径查询的数据源。"""
    try:
        url = f"https://api.unpaywall.org/v2/{quote(doi, safe='/')}?email={quote(email)}"
        response = session.get(url, timeout=PAGE_TIMEOUT,
                               headers={"Accept": "application/json"})
        if response.status_code == 404:
            return _fail("Unpaywall 无此 DOI 记录")
        if response.status_code != 200:
            return _fail(f"Unpaywall HTTP {response.status_code}",
                         site_down=response.status_code in (403, 429)
                         or response.status_code >= 500)
        data = response.json() or {}
        locations = []
        if data.get("best_oa_location"):
            locations.append(data["best_oa_location"])
        locations.extend(data.get("oa_locations") or [])
        for location in locations:
            pdf_url = location.get("url_for_pdf")
            if pdf_url:
                return _ok(pdf_url)
        return _fail("Unpaywall 无开放获取 PDF 直链")
    except (requests.RequestException, ValueError) as e:
        return _fail(f"Unpaywall 出错：{e}", site_down=True)


_META_PDF_PATTERNS = [
    re.compile(r'<meta[^>]+name\s*=\s*["\']citation_pdf_url["\'][^>]+content\s*=\s*["\']([^"\']+)["\']',
               re.IGNORECASE),
    re.compile(r'<meta[^>]+content\s*=\s*["\']([^"\']+)["\'][^>]+name\s*=\s*["\']citation_pdf_url["\']',
               re.IGNORECASE),
    re.compile(r'<link[^>]+type\s*=\s*["\']application/pdf["\'][^>]+href\s*=\s*["\']([^"\']+)["\']',
               re.IGNORECASE),
]


def fetch_from_doi_landing_page(doi: str, session: requests.Session) -> dict:
    """DOI 落地页抓取（Zotero 官方「查找可用 PDF」的页面抓取途径）。

    加载 https://doi.org/{DOI}，跟随重定向到出版社文章页，从
    <meta name="citation_pdf_url"> 或 <link type="application/pdf">
    提取 PDF 链接（相对链接按最终页面 URL 解析）。
    """
    try:
        response = session.get(f"https://doi.org/{quote(doi, safe='/')}",
                               timeout=PAGE_TIMEOUT, allow_redirects=True)
        if response.status_code != 200:
            return _fail(f"DOI 落地页 HTTP {response.status_code}")
        page_url = str(response.url)
        for pattern in _META_PDF_PATTERNS:
            match = pattern.search(response.text)
            if match:
                pdf_url = urljoin(page_url, _unescape_href(match.group(1)))
                return _ok(pdf_url, referer=page_url)
        return _fail("落地页上未发现 citation_pdf_url 标记")
    except requests.RequestException as e:
        return _fail(f"DOI 落地页出错：{e}")


_ANNAS_D3_LINK = re.compile(r'href\s*=\s*["\'](https?://[^"\']*/d3/x/[^"\']+)["\']')
_ANNAS_MD5_LINK = re.compile(r'/md5/([a-f0-9]{32})', re.IGNORECASE)


def _annas_scidb_link(doi: str, base: str, session: requests.Session) -> "dict | None":
    """访问 /scidb/{DOI}/ 并提取直链；页面上没有可下载的东西时返回 None。

    出错（网络异常 / 非 200）时返回一个 _fail 字典，调用方直接透传。
    """
    try:
        response = session.get(f"{base}scidb/{doi}/", timeout=PAGE_TIMEOUT)
    except requests.RequestException as e:
        return _fail(f"Anna's Archive 出错：{e}", site_down=True)
    if response.status_code != 200:
        blocked = _anti_bot_reason(response.text or "")
        detail = f"（{blocked}）" if blocked else ""
        return _fail(f"Anna's Archive HTTP {response.status_code}{detail}",
                     site_down=True)
    page = response.text
    match = _ANNAS_D3_LINK.search(page)
    if match:
        return _ok(_unescape_href(match.group(1)))
    # 兜底：文字为 Download 的链接
    match = re.search(
        r'<a[^>]+href\s*=\s*["\'](https?://[^"\']+)["\'][^>]*>\s*Download\s*</a>',
        page, re.IGNORECASE)
    if match:
        return _ok(_unescape_href(match.group(1)))
    return None


def _annas_doi_via_search(doi: str, base: str,
                          session: requests.Session) -> "str | None":
    """用 Anna's Archive 的期刊检索把 DOI 换成它自己收录时用的那个 DOI。

    存在的理由：出版社更迭后同一篇文献会有两个 DOI（例如 Blackwell 时期的
    10.1046/… 与 Wiley 时期的 10.1111/…），而 /scidb/ 只认它入库时用的那一个。
    Anna's Archive 的检索是按元数据建的索引，两个 DOI 都能搜到同一条记录，
    因此「检索 → 取 md5 详情页 → 从详情页读出它收录时用的 DOI」这条路能把
    别名桥接过去。
    """
    try:
        search = session.get(
            f"{base}search?index=journals&q={quote(doi, safe='')}",
            timeout=PAGE_TIMEOUT)
        if search.status_code != 200:
            return None
        md5s = _ANNAS_MD5_LINK.findall(search.text)
        if not md5s:
            return None
        detail = session.get(f"{base}md5/{md5s[0]}", timeout=PAGE_TIMEOUT)
        if detail.status_code != 200:
            return None
    except requests.RequestException:
        return None

    for candidate in match_dois(detail.text):
        candidate = normalize_doi(candidate).rstrip(".")
        # 详情页里 DOI 常以 ".pdf" 结尾（来自文件名），去掉后再比
        candidate = re.sub(r"\.pdf$", "", candidate, flags=re.IGNORECASE)
        if candidate and candidate.lower() != doi.lower():
            return candidate
    return None


def fetch_from_annas_archive(doi: str, base_url: str,
                             session: requests.Session) -> dict:
    """Anna's Archive：/scidb/{DOI}/ 页面提取 "/d3/x/" 直链。

    /scidb/ 直接命中不了时，再走一次「期刊检索 → md5 详情页 → 换回它收录时
    用的 DOI → 重新 /scidb/」，以跨过同一篇文献的 DOI 别名。
    """
    base = _normalize_base(base_url)
    result = _annas_scidb_link(doi, base, session)
    if result is not None:
        return result

    alias = _annas_doi_via_search(doi, base, session)
    if alias:
        result = _annas_scidb_link(alias, base, session)
        if result is not None and result["success"]:
            return result

    # 页面能正常打开、只是没有这一篇——镜像本身是好的，不拉黑
    return _fail("Anna's Archive 页面上没有下载链接")


def fetch_from_libgen(doi: str, base_url: str,
                      session: requests.Session) -> dict:
    """LibGen：搜 DOI → edition.php → ads.php → get.php（带动态 key）。"""
    base = _normalize_base(base_url)
    try:
        search = session.get(
            f"{base}index.php?req={quote(doi, safe='')}&columns=doi",
            timeout=PAGE_TIMEOUT)
        if search.status_code != 200:
            return _fail(f"LibGen HTTP {search.status_code}", site_down=True)

        edition = re.search(r'href\s*=\s*["\']([^"\']*edition\.php\?id=\d+[^"\']*)["\']',
                            search.text)
        if not edition:
            # 搜索页正常返回、只是没收录这个 DOI——镜像本身是好的，不拉黑
            return _fail("LibGen 搜索结果中没有该 DOI")

        edition_url = urljoin(base, _unescape_href(edition.group(1)))
        edition_page = session.get(edition_url, timeout=PAGE_TIMEOUT)
        ads = re.search(r'href\s*=\s*["\']([^"\']*ads\.php\?md5=[a-fA-F0-9]+[^"\']*)["\']',
                        edition_page.text)
        if not ads:
            # 搜到了版本页却没有下载入口，说明这个镜像的 scimag 部分已失效
            # （或页面版式已变），对其他文献同样无望
            return _fail("LibGen 版本页上没有下载入口", site_down=True)

        ads_url = urljoin(base, _unescape_href(ads.group(1)))
        ads_page = session.get(ads_url, timeout=PAGE_TIMEOUT)
        get_link = re.search(r'href\s*=\s*["\']([^"\']*get\.php\?md5=[a-fA-F0-9]+[^"\']*)["\']',
                             ads_page.text)
        if not get_link:
            return _fail("LibGen 下载页上没有动态 key", site_down=True)

        get_url = urljoin(base, _unescape_href(get_link.group(1)))
        # get.php 会检查 Referer（指向生成 key 的 ads.php 页面），否则返回 500
        return _ok(get_url, referer=ads_url)
    except requests.RequestException as e:
        return _fail(f"LibGen 出错：{e}", site_down=True)


_SCIHUB_PDF_PATTERNS = [
    re.compile(r'id\s*=\s*["\']pdf["\'][^>]*src\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE),
    re.compile(r'src\s*=\s*["\']([^"\']+)["\'][^>]*id\s*=\s*["\']pdf["\']', re.IGNORECASE),
    re.compile(r'<object[^>]+data\s*=\s*["\']([^"\']+\.pdf[^"\']*)["\']', re.IGNORECASE),
    re.compile(r'<meta[^>]+name\s*=\s*["\']citation_pdf_url["\'][^>]+content\s*=\s*["\']([^"\']+)["\']',
               re.IGNORECASE),
]


def fetch_from_scihub(doi: str, base_url: str,
                      session: requests.Session) -> dict:
    """Sci-Hub：镜像页面提取 #pdf embed / object / citation_pdf_url。"""
    base = _normalize_base(base_url)
    page_url = f"{base}{doi}"
    try:
        response = session.get(page_url, timeout=PAGE_TIMEOUT,
                               headers={"User-Agent": _MOBILE_USER_AGENT})
        page = response.text or ""
        if response.status_code == 200:
            for pattern in _SCIHUB_PDF_PATTERNS:
                match = pattern.search(page)
                if match:
                    raw = _unescape_href(match.group(1))
                    pdf_url = urljoin(page_url, raw)
                    pdf_url = re.sub(r"^http://", "https://", pdf_url)
                    return _ok(pdf_url, referer=page_url)
            if not page.strip() or any(p.search(page) for p in _SCIHUB_NOT_AVAILABLE):
                # 镜像明确回答「没这一篇」，镜像本身是好的，不拉黑
                return _fail("Sci-Hub 未收录该文献")
            if any(p.search(page) for p in _SCIHUB_VERIFICATION):
                # 人机验证页：分不清是「没收录」还是「被限流」，一律不拉黑，
                # 免得把当前唯一可用的镜像因为几篇没收录的文献误伤
                return _fail("Sci-Hub 返回人机验证页（通常意味着未收录该文献，"
                             "也可能是被限流）")
        blocked = _anti_bot_reason(page)
        if blocked:
            return _fail(f"Sci-Hub {blocked}（HTTP {response.status_code}）",
                         site_down=True)
        # 其余情况（停靠广告页 / 版式失效）都算镜像本身有问题
        return _fail(f"Sci-Hub 页面上没有 PDF 链接（HTTP {response.status_code}）",
                     site_down=True)
    except requests.RequestException as e:
        return _fail(f"Sci-Hub 出错：{e}", site_down=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  下载与校验                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def download_url_to_file(
    session: requests.Session,
    url: str,
    dest: Path,
    referer: "str | None" = None,
    read_timeout: "int | None" = DOWNLOAD_TIMEOUT,
    log=print,
) -> "Path | None":
    """流式下载到 dest（先写 .part 临时文件），校验 %PDF 魔数后落盘。

    - LibGen get.php 链接自动补 Referer 指向 ads.php（服务器校验此头）；
    - 首块内容不含 %PDF 魔数、或总大小过小，判为失败并清理临时文件。
    """
    headers = {"Accept": "application/pdf,application/octet-stream,*/*;q=0.8"}
    if referer:
        headers["Referer"] = referer
    elif "get.php" in url:
        md5_match = re.search(r"[?&]md5=([a-fA-F0-9]+)", url)
        if md5_match:
            origin = re.match(r"https?://[^/]+", url)
            if origin:
                headers["Referer"] = f"{origin.group(0)}/ads.php?md5={md5_match.group(1)}"

    part_path = dest.with_suffix(dest.suffix + ".part")
    try:
        with session.get(url, stream=True, headers=headers,
                         timeout=(PAGE_TIMEOUT, read_timeout)) as response:
            if response.status_code != 200:
                return None
            first_chunk = b""
            total = 0
            with open(part_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=65536):
                    if not chunk:
                        continue
                    if not first_chunk:
                        first_chunk = chunk[:1024]
                        if b"%PDF" not in first_chunk:
                            return None  # 非 PDF 内容（HTML 报错页等）
                    f.write(chunk)
                    total += len(chunk)
        if total < MIN_PDF_BYTES:
            return None
        part_path.replace(dest)
        return dest
    except requests.RequestException:
        return None
    finally:
        if part_path.exists():
            try:
                part_path.unlink()
            except OSError:
                pass


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  单条目下载（文章：多渠道轮询）                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class SourceHealth:
    """全部文献共用的渠道健康档案（线程安全）。

    存在的理由：并行的是文献，各文献却走同一份渠道清单，如果每篇都把整个
    清单从头试到尾，一个已经整站 403 的镜像会被反复访问 N 次——这正是要
    消除的无意义访问。因此：

    - 某渠道成功过 → 排到最前面，后续文献优先走这条已被证明可用的路；
    - 某渠道累计出现 SITE_FAILURE_BLACKLIST_THRESHOLD 次整站级故障且从未
      成功过 → 本次运行内拉黑，后续文献直接跳过；
    - 「渠道正常、只是没这一篇」不计入故障，因为它对别的文献毫无预示作用。
    """

    def __init__(self, threshold: int = SITE_FAILURE_BLACKLIST_THRESHOLD) -> None:
        self._lock = threading.Lock()
        self._threshold = threshold
        self._successes: dict[str, int] = {}
        self._site_failures: dict[str, int] = {}
        self._announced: set[str] = set()

    def order(self, chain: list) -> list:
        """重排渠道清单：合法开放获取渠道原样留在最前面，其后的镜像里
        成功次数多的优先，同样成功次数的保持原顺序。"""
        with self._lock:
            successes = dict(self._successes)

        def sort_key(pair: tuple) -> tuple:
            index, (source_name, _fetcher) = pair
            if source_name in _PRIMARY_SOURCES:
                return (0, 0, index)
            return (1, -successes.get(source_name, 0), index)

        return [item for _, item in sorted(enumerate(chain), key=sort_key)]

    def is_blacklisted(self, source_name: str) -> bool:
        if source_name in _NEVER_BLACKLIST:
            return False
        with self._lock:
            return (self._successes.get(source_name, 0) == 0
                    and self._site_failures.get(source_name, 0) >= self._threshold)

    def record_success(self, source_name: str) -> None:
        with self._lock:
            self._successes[source_name] = self._successes.get(source_name, 0) + 1

    def record_failure(self, source_name: str, site_down: bool) -> "str | None":
        """记一次未命中；本次调用刚好触发拉黑时返回一句提示，否则返回 None。"""
        if not site_down or source_name in _NEVER_BLACKLIST:
            return None
        with self._lock:
            count = self._site_failures.get(source_name, 0) + 1
            self._site_failures[source_name] = count
            if (count < self._threshold or self._successes.get(source_name, 0)
                    or source_name in self._announced):
                return None
            self._announced.add(source_name)
        return (f"  🚫 {source_name} 已连续 {count} 次整站级故障，"
                f"本次运行内不再尝试该渠道。")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  DOI 别名发现（同一篇文献被注册过不止一个 DOI）                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

#: 出版社更迭时，同一篇文献往往在新旧两个前缀下各注册过一次 DOI，两个都是
#: 有效的、都能在 CrossRef 查到，但各家下载渠道只认自己入库时用的那一个。
#: 这些前缀之间后缀可以原样互换（Blackwell / Munksgaard 的 "j.<ISSN>.<年>.<号>.x"
#: 命名在并入 Wiley 后被整体沿用），因此可以零成本构造出候选别名。
_ALIAS_PREFIX_FAMILIES: list[set[str]] = [
    {"10.1046", "10.1111", "10.1034"},   # Blackwell Science / Munksgaard → Wiley
]

#: 判定「候选 DOI 与本条目是同一篇文献」所需的标题相似度
ALIAS_TITLE_SIMILARITY = 0.85

_CROSSREF_WORKS_URL = "https://api.crossref.org/works"
_CROSSREF_HEADERS = {"User-Agent": f"LitRetrieval-Python/1.0 (mailto:{CONTACT_EMAIL})",
                     "Accept": "application/json"}


def _prefix_swap_candidates(doi: str) -> list[str]:
    """把 DOI 前缀换成同一迁移家族里的其他前缀，后缀原样保留。"""
    prefix, _, suffix = doi.partition("/")
    # 只对 Wiley / Blackwell 系的 "j.…" 后缀做互换，别的命名风格换了也不会存在
    if not suffix.lower().startswith("j."):
        return []
    for family in _ALIAS_PREFIX_FAMILIES:
        if prefix in family:
            return [f"{other}/{suffix}" for other in sorted(family) if other != prefix]
    return []


def _crossref_work_title(doi: str, session: requests.Session) -> "str | None":
    """取 CrossRef 中该 DOI 的标题；DOI 不存在或请求失败时返回 None。"""
    try:
        response = session.get(f"{_CROSSREF_WORKS_URL}/{quote(doi, safe='/')}",
                               timeout=PAGE_TIMEOUT, headers=_CROSSREF_HEADERS)
        if response.status_code != 200:
            return None
        titles = ((response.json() or {}).get("message") or {}).get("title") or []
    except (requests.RequestException, ValueError):
        return None
    return titles[0] if titles else ""


def _crossref_title_search(title: str,
                           session: requests.Session) -> list[tuple[str, str]]:
    """按标题检索 CrossRef，返回 [(DOI, 标题)]。"""
    try:
        response = session.get(
            f"{_CROSSREF_WORKS_URL}?rows=10&select=DOI,title"
            f"&query.bibliographic={quote(title)}",
            timeout=PAGE_TIMEOUT, headers=_CROSSREF_HEADERS)
        if response.status_code != 200:
            return []
        items = (((response.json() or {}).get("message") or {}).get("items")) or []
    except (requests.RequestException, ValueError):
        return []
    return [(item.get("DOI") or "", (item.get("title") or [""])[0])
            for item in items if item.get("DOI")]


def find_alias_dois(ref: dict, session: requests.Session, log=print) -> list[str]:
    """找出与本条目是同一篇文献、但 DOI 不同的其他注册记录。

    动机（2026-07-26 修复）：CrossRef 对同一篇文献可能存有两条彼此独立的记录，
    两条记录之间没有任何字段互相指向（relation 为空，alternative-id 只有自己），
    OpenAlex 也是两条独立的 work。下载渠道只收录其中一个 DOI，于是「元数据补全
    拿到的 DOI 完全正确、下载却全渠道扑空」。实际发生过的例子：

      Madsen & Shine 2000, J. Anim. Ecol. 69:952–958
        CrossRef 返回 10.1046/j.1365-2656.2000.00477.x（Blackwell 时期）
        各下载渠道收录的却是 10.1111/j.1365-2656.2000.00477.x（Wiley 时期）

      King et al. 1999, J. Zool. 247:19–28
        CrossRef 返回 10.1017/s0952836999001028（Cambridge 时期）
        各下载渠道收录的却是 10.1111/j.1469-7998.1999.tb00189.x（Wiley 时期）

    两条途径互补，前者零成本但只覆盖 Wiley / Blackwell，后者通用：
      (a) 前缀互换后用 CrossRef 确认该 DOI 真实存在（覆盖上面第一例）；
      (b) 拿标题去 CrossRef 检索，取标题几乎一致的其他 DOI（覆盖第二例）。
    """
    doi = normalize_doi(ref.get("doi") or "")
    if not doi:
        return []
    title = str(ref.get("title") or "").strip()
    label = ref_short_label(ref)
    aliases: list[str] = []

    def consider(candidate: str, candidate_title: str, how: str) -> None:
        candidate = normalize_doi(candidate)
        if not candidate or candidate.lower() == doi.lower():
            return
        if any(candidate.lower() == existing.lower() for existing in aliases):
            return
        if title and candidate_title:
            similarity = jaccard_similarity(title.lower(), candidate_title.lower())
            if similarity < ALIAS_TITLE_SIMILARITY:
                return
        aliases.append(candidate)
        log(f"     {label}：发现同一篇文献的另一个 DOI {candidate}（{how}）")

    for candidate in _prefix_swap_candidates(doi):
        candidate_title = _crossref_work_title(candidate, session)
        if candidate_title is None:      # CrossRef 里没有这个 DOI
            continue
        consider(candidate, candidate_title, "前缀互换后经 CrossRef 确认")

    if title:
        for candidate, candidate_title in _crossref_title_search(title, session):
            consider(candidate, candidate_title, "CrossRef 标题检索")

    return aliases


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  按标题发现 DOI（补 CrossRef 覆盖不到的旧刊，含 JSTOR）                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

#: 认定「检索到的就是这一篇」所需的标题相似度。取 0.6 是因为 Anna's Archive
#: 收录的标题偶尔带副标题或大小写差异，真正命中时相似度通常在 0.9 以上，
#: 而不相干的结果一般落在 0.4 以下，中间有很宽的安全带。
TITLE_SEARCH_SIMILARITY = 0.6

_MD5_ANCHOR = re.compile(r'(?=<a[^>]+href="/md5/[a-f0-9]{32}")')
_MD5_HREF = re.compile(r'href="/md5/([a-f0-9]{32})"')
#: Anna's Archive 检索结果里文件路径形如 `scihub/10.2307/3892027.pdf`，
#: DOI 直接写在路径里，因此不必逐条打开详情页
_ANNAS_PATH_DOI = re.compile(r'\bscihub/(10\.\d{4,9}/[^\s"\'<>|]+?)\.pdf\b',
                             re.IGNORECASE)
_YEAR_IN_TEXT = re.compile(r"\b(1[6-9]\d{2}|20\d{2})\b")


def _annas_block_text(block: str) -> str:
    text = re.sub(r"<[^>]+>", " | ", block)
    text = re.sub(r"\s*\|\s*(\|\s*)+", " | ", text)
    return " ".join(text.split())


def parse_annas_search_page(page: str) -> list[dict]:
    """把 Anna's Archive 的期刊检索结果页解析成 [{md5, doi, title, meta}]。

    一条结果由两个共享同一 md5 的块组成：一块是文件路径（DOI 在里面），
    另一块是「标题 | 作者 | 出版方、期刊、期、卷、页码、年份 | 摘要」。
    部分结果放在 HTML 注释里由前端展开，所以要把注释一并纳入。
    """
    haystack = page + "\n".join(re.findall(r"<!--(.*?)-->", page, re.S))
    records: dict[str, dict] = {}
    order: list[str] = []
    for block in _MD5_ANCHOR.split(haystack):
        found = _MD5_HREF.search(block)
        if not found:
            continue
        md5 = found.group(1)
        record = records.setdefault(
            md5, {"md5": md5, "doi": None, "title": "", "meta": ""})
        if md5 not in order:
            order.append(md5)
        doi_match = _ANNAS_PATH_DOI.search(block)
        if doi_match and not record["doi"]:
            record["doi"] = normalize_doi(doi_match.group(1))
        parts = [p.strip() for p in _annas_block_text(block).split("|") if p.strip()]
        if len(parts) >= 3 and not record["title"]:
            record["title"] = parts[0]
            record["meta"] = " | ".join(parts[1:3])
    return [records[md5] for md5 in order]


def find_doi_by_title(ref: dict, session: requests.Session,
                      base_url: str = "annas-archive.pk", log=print) -> "str | None":
    """拿标题去 Anna's Archive 的期刊索引里找这篇文献的 DOI。

    动机（2026-07-26 新增）：CrossRef 并非无所不包——很多期刊的旧刊压根
    没在 CrossRef 注册过。实测 `Herpetologica` 1974 年那一卷在 CrossRef 里
    一条记录都没有，所以第 2 步无论怎么检索都找不到
    「LOOP, M. S. 1974. …. Herpetologica 30:123–127.」这条引文，条目最终
    没有 DOI，下载环节直接跳过。但这篇文章在 JSTOR 上（stable ID 3892027，
    对应 DOI `10.2307/3892027`），Sci-Hub 与 Anna's Archive 都按这个 DOI
    收录了它。

    JSTOR 本身没法程序化访问（文章页与检索页都返回 HTTP 403 的 reCAPTCHA
    验证页），所以这里走 Anna's Archive 的期刊索引——它按元数据建索引，
    标题检索能命中，而结果里的文件路径直接写着 DOI。

    候选按标题相似度排序，并要求年份不冲突；相似度低于
    TITLE_SEARCH_SIMILARITY 时宁可返回 None，也不给一个可能错的 DOI。
    """
    title = str(ref.get("title") or "").strip()
    if not title:
        return None
    label = ref_short_label(ref)
    base = _normalize_base(base_url)
    try:
        response = session.get(f"{base}search?index=journals&q={quote(title)}",
                               timeout=PAGE_TIMEOUT)
        if response.status_code != 200:
            log(f"     {label}：标题检索未能进行（Anna's Archive 检索页 "
                f"HTTP {response.status_code}）")
            return None
    except requests.RequestException as e:
        log(f"     {label}：标题检索出错（{e}）")
        return None

    year = ref.get("year")
    scored: list[tuple[float, dict]] = []
    for candidate in parse_annas_search_page(response.text):
        if not candidate["doi"] or not candidate["title"]:
            continue
        if year:
            years = {int(y) for y in _YEAR_IN_TEXT.findall(candidate["meta"])}
            if years and int(year) not in years:
                continue          # 年份明确冲突，直接排除
        similarity = jaccard_similarity(title.lower(), candidate["title"].lower())
        scored.append((similarity, candidate))

    if not scored:
        return None
    similarity, best = max(scored, key=lambda pair: pair[0])
    if similarity < TITLE_SEARCH_SIMILARITY:
        log(f"     {label}：标题检索没有足够可信的结果"
            f"（最高相似度 {similarity:.2f}，门槛 {TITLE_SEARCH_SIMILARITY}）")
        return None
    log(f"     {label}：标题检索找到 {best['doi']}"
        f"（标题相似度 {similarity:.2f}）← {best['title'][:60]}")
    return best["doi"]


def _article_source_chain(openalex_key: str, email: str) -> list[tuple[str, "callable"]]:
    """构建文章条目的渠道尝试顺序（全部文献共用同一份顺序）。

    开放获取 API（OpenAlex / Unpaywall / DOI 落地页）总是最先尝试，镜像渠道
    随后，顺序与参考插件的默认配置一致。运行期的实际顺序还会被 SourceHealth
    按「成功过的优先」调整。
    """
    chain: list[tuple[str, "callable"]] = [
        ("OpenAlex", lambda doi, s: fetch_from_openalex(doi, s, openalex_key)),
        ("Unpaywall", lambda doi, s: fetch_from_unpaywall(doi, s, email)),
        ("DOI 落地页", fetch_from_doi_landing_page),
    ]
    dispatch = {
        "annas-archive": fetch_from_annas_archive,
        "libgen": fetch_from_libgen,
        "scihub": fetch_from_scihub,
    }
    for source_type, domain in MIRROR_SOURCES:
        fetcher = dispatch[source_type]
        chain.append((
            domain,
            (lambda d, s, _f=fetcher, _dom=domain: _f(d, _dom, s)),
        ))
    return chain


def _try_source_chain(doi: str, ref: dict, dest: Path, session: requests.Session,
                      chain: list, health: "SourceHealth", label: str,
                      log=print) -> bool:
    """用给定的 DOI 串行轮询各渠道；下载成功并落盘时返回 True。"""
    skipped: list[str] = []
    for source_name, fetcher in chain:
        if health.is_blacklisted(source_name):
            skipped.append(source_name)
            continue
        result = fetcher(doi, session)
        if not result["success"]:
            log(f"     {label}：{source_name} 未命中（{result['error']}）")
            notice = health.record_failure(source_name, result.get("site_down", False))
            if notice:
                log(notice)
            continue
        log(f"     {label}：{source_name} 找到直链，开始下载……")
        saved = download_url_to_file(session, result["pdf_url"], dest,
                                     referer=result.get("referer"), log=log)
        if saved:
            size_mb = saved.stat().st_size / 1024 / 1024
            health.record_success(source_name)
            ref["status"]["download"] = "success"
            ref["status"]["download_source"] = source_name
            ref["status"]["pdf_path"] = str(saved)
            ref["status"]["error"] = None
            log(f"  ✅ {label}：下载成功（来源 {source_name}，{size_mb:.2f} MB）"
                f"→ {saved.name}")
            return True
        log(f"     {label}：{source_name} 的直链下载失败或内容不是 PDF，换下一个渠道。")
        notice = health.record_failure(source_name, site_down=True)
        if notice:
            log(notice)

    if skipped:
        log(f"     {label}：已跳过本次运行中被拉黑的 {len(skipped)} 个渠道"
            f"（{'、'.join(skipped)}）。")
    return False


def download_article(ref: dict, workdir: Path, session: requests.Session,
                     openalex_key: str, email: str,
                     health: "SourceHealth | None" = None, log=print) -> None:
    """对单个文章条目串行轮询各渠道，任一渠道拿到 PDF 就立即收工。

    有两级兜底：

    1. 条目**根本没有 DOI**（CrossRef 覆盖不到的旧刊，例如 JSTOR 上的
       Herpetologica 1974 年那一卷）→ 先按标题检索出 DOI 再走渠道，
       而不是像早先那样直接放弃；
    2. 本条目自己的 DOI 走完全部渠道仍然扑空 → 查「同一篇文献的其他 DOI」
       （见 find_alias_dois），拿别名把各渠道重跑一轮；别名也没救回来时，
       再按标题检索一次（DOI 可能压根就匹配错了）。
    """
    label = ref_short_label(ref)
    health = health or SourceHealth()
    doi = ref.get("doi")

    if not doi:
        log(f"     {label}：条目没有 DOI，先按标题检索……")
        doi = find_doi_by_title(ref, session, log=log)
        if not doi:
            ref["status"]["download"] = "failed"
            ref["status"]["error"] = "无 DOI，且按标题也没检索到"
            log(f"  ❌ {label}：没有 DOI、按标题也没检索到，跳过下载。")
            return
        ref["doi"] = doi
        ref["doi_source"] = "annas-title-search"
        log(f"  ℹ️ {label}：按标题检索补上了 DOI {doi}。")

    dest = unused_path(workdir / expected_pdf_filename(ref))
    chain = health.order(_article_source_chain(openalex_key, email))

    if _try_source_chain(doi, ref, dest, session, chain, health, label, log=log):
        return

    log(f"     {label}：本条目的 DOI 全渠道扑空，查一下这篇文献有没有别的 DOI……")
    aliases = find_alias_dois(ref, session, log=log)
    title_doi = find_doi_by_title(ref, session, log=log)
    if title_doi and title_doi.lower() != doi.lower() \
            and not any(title_doi.lower() == a.lower() for a in aliases):
        aliases.append(title_doi)
    for alias in aliases:
        log(f"     {label}：改用 {alias} 重试各渠道……")
        # 别名重试时渠道顺序可能已被上一轮的成败改写，重新排一次
        alias_chain = health.order(_article_source_chain(openalex_key, email))
        if _try_source_chain(alias, ref, dest, session, alias_chain, health,
                             label, log=log):
            ref["status"]["download_doi"] = alias
            log(f"  ℹ️ {label}：这篇文献实际是用 {alias} 下到的"
                f"（条目记录的 DOI 是 {doi}，两个都指向同一篇文献）。")
            return

    ref["status"]["download"] = "failed"
    if aliases:
        ref["status"]["error"] = (
            f"所有渠道均未获取到 PDF（本条目 DOI 与别名 "
            f"{'、'.join(aliases)} 都试过了）")
    else:
        ref["status"]["error"] = "所有渠道均未获取到 PDF"
    log(f"  ❌ {label}：所有渠道都尝试过了，未能下载到 PDF。")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  书籍条目（LibGen 标题搜索，永不并行，最后处理）                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def download_book(ref: dict, workdir: Path, session: requests.Session,
                  log=print) -> None:
    """书籍条目：LibGen 标题（+第一作者）搜索 → ads.php → get.php → CDN。

    这是参考插件 A4_LibGenBook.ts 的简化移植：搜索
    index.php?req={标题 作者}&columns[]=t&columns[]=a&columns[]=s，
    无结果时退回仅标题搜索，然后取结果页上的第一个 ads.php 下载入口。
    书籍下载可能很慢，一旦开始下载不设读超时。
    """
    label = ref_short_label(ref)
    title = ref.get("title") or ""
    if not title:
        ref["status"]["download"] = "failed"
        ref["status"]["error"] = "书籍条目无标题，无法搜索"
        log(f"  ❌ {label}：书籍条目没有标题，无法搜索。")
        return

    names = ref.get("first_author_last_name") or []
    author = str(names[0]) if names else ""
    libgen_mirrors = [d for t, d in MIRROR_SOURCES if t == "libgen"]

    queries = [f"{title} {author}".strip(), title] if author else [title]
    for base_domain in libgen_mirrors:
        base = _normalize_base(base_domain)
        for query in queries:
            log(f"     {label}：在 {base_domain} 搜索「{query[:60]}」……")
            try:
                search = session.get(
                    f"{base}index.php?req={quote(query)}"
                    f"&columns%5B%5D=t&columns%5B%5D=a&columns%5B%5D=s",
                    timeout=PAGE_TIMEOUT)
            except requests.RequestException as e:
                log(f"     {label}：{base_domain} 访问失败（{e}），换下一个镜像。")
                break
            ads_links = re.findall(
                r'href\s*=\s*["\']([^"\']*ads\.php\?md5=[a-fA-F0-9]+[^"\']*)["\']',
                search.text)
            if not ads_links:
                continue
            ads_url = urljoin(base, _unescape_href(ads_links[0]))
            try:
                ads_page = session.get(ads_url, timeout=PAGE_TIMEOUT)
            except requests.RequestException:
                continue
            get_link = re.search(
                r'href\s*=\s*["\']([^"\']*get\.php\?md5=[a-fA-F0-9]+[^"\']*)["\']',
                ads_page.text)
            if not get_link:
                continue
            get_url = urljoin(base, _unescape_href(get_link.group(1)))
            dest = unused_path(workdir / expected_pdf_filename(ref))
            log(f"     {label}：找到下载入口，开始下载（书籍可能较慢，请耐心等待）……")
            saved = download_url_to_file(session, get_url, dest,
                                         referer=ads_url, read_timeout=None,
                                         log=log)
            if saved:
                size_mb = saved.stat().st_size / 1024 / 1024
                ref["status"]["download"] = "success"
                ref["status"]["download_source"] = f"LibGen 书籍搜索（{base_domain}）"
                ref["status"]["pdf_path"] = str(saved)
                log(f"  ✅ {label}：书籍下载成功（{size_mb:.2f} MB）→ {saved.name}")
                return

    ref["status"]["download"] = "failed"
    ref["status"]["error"] = "LibGen 书籍搜索未找到可下载的文件"
    log(f"  ❌ {label}：书籍搜索未能下载到文件。")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  查重 + 批量下载                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def dedup_against_library(refs: list[dict], workdir: Path,
                          library_root: "Path | None" = None,
                          log=print) -> int:
    """在库根目录中按文件名查重；命中的条目标记 already_exists 并跳过下载。"""
    root = library_root or workdir.parent
    index = build_library_index(root, log=log)
    hits = 0
    for ref in refs:
        if not ref.get("selected", True):
            continue
        if ref["status"]["download"] in ("success", "already_exists"):
            continue
        existing = find_existing_pdf_for_reference(ref, index)
        if existing:
            ref["status"]["download"] = "already_exists"
            ref["status"]["existing_path"] = str(existing)
            hits += 1
            log(f"  ⏭️ {ref_short_label(ref)}：库中已存在同一文献，跳过下载。")
            log(f"     既有文件：{existing}")
    if hits == 0:
        log("  ℹ️ 查重未发现库中已有的文献。")
    return hits


def _needs_download(ref: dict) -> bool:
    if not ref.get("selected", True):
        return False
    status = ref["status"]
    if status["download"] == "already_exists":
        return False
    if status["download"] == "success":
        # 之前下载过：确认文件还在（不在则重新下载）
        pdf_path = status.get("renamed_path") or status.get("pdf_path")
        if pdf_path and Path(pdf_path).exists():
            return False
    return True


def download_all(
    refs: list[dict],
    workdir: Path,
    concurrency: int = DEFAULT_ARTICLE_CONCURRENCY,
    openalex_key: str = "",
    email: str = CONTACT_EMAIL,
    proxy: "str | None" = None,
    state: "dict | None" = None,
    log=print,
) -> None:
    """批量下载：文章之间并行（每篇内部渠道串行），书籍随后逐个处理。"""
    todo = [r for r in refs if _needs_download(r)]
    articles = [r for r in todo if not is_book_reference(r)]
    books = [r for r in todo if is_book_reference(r)]

    skipped = len(refs) - len(todo)
    log(f"\n  📥 待下载 {len(todo)} 条（文章 {len(articles)} 条、书籍 {len(books)} 条）；"
        f"已完成或已存在 {skipped} 条将跳过。")
    if not todo:
        return

    openalex_key = openalex_key or OPENALEX_API_KEY
    progress_lock = threading.Lock()
    health = SourceHealth()
    completed = 0

    def save_progress() -> None:
        if state is not None:
            save_state(workdir, state)

    def article_worker(ref: dict) -> None:
        nonlocal completed
        lines: list[str] = []
        session = make_session(proxy)
        try:
            download_article(ref, workdir, session, openalex_key, email,
                             health=health, log=lines.append)
        except Exception as e:
            ref["status"]["download"] = "failed"
            ref["status"]["error"] = str(e)
            lines.append(f"  ❌ {ref_short_label(ref)}：下载过程出错：{e}")
        with progress_lock:
            completed += 1
            for line in lines:
                log(line)
            log(f"  📊 总进度：{completed}/{len(todo)}")
            save_progress()

    if articles:
        log(f"  🚀 开始下载文章：{min(concurrency, len(articles))} 篇同时进行，"
            f"每篇内部逐个渠道串行尝试……")
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(article_worker, ref) for ref in articles]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    if books:
        log(f"\n  📚 开始逐个下载书籍（书籍永不并行，共 {len(books)} 本）……")
        session = make_session(proxy)
        for ref in books:
            try:
                download_book(ref, workdir, session, log=log)
            except Exception as e:
                ref["status"]["download"] = "failed"
                ref["status"]["error"] = str(e)
                log(f"  ❌ {ref_short_label(ref)}：书籍下载出错：{e}")
            with progress_lock:
                completed += 1
                log(f"  📊 总进度：{completed}/{len(todo)}")
                save_progress()

    success = sum(1 for r in todo if r["status"]["download"] == "success")
    failed = [r for r in todo if r["status"]["download"] == "failed"]
    log(f"\n  📊 下载完成：成功 {success} 条、失败 {len(failed)} 条。")
    for ref in failed:
        log(f"     ❌ {ref_short_label(ref)}：{ref['status'].get('error') or '未知原因'}")
    if failed:
        log(f"     这些条目的 RIS 保留在 {workdir}，文件名与它们将来的 PDF 文件名"
            f"一致；重新运行本脚本即可继续尝试下载。")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  主流程                                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def run(
    workdir: Path,
    concurrency: int = DEFAULT_ARTICLE_CONCURRENCY,
    openalex_key: str = "",
    email: str = CONTACT_EMAIL,
    proxy: "str | None" = None,
    library_root: "Path | None" = None,
    dedup: bool = True,
    log=print,
) -> "list[dict] | None":
    """对状态文件中的条目查重并下载；返回条目列表（供驱动脚本调用）。"""
    state = load_state(workdir)
    if state is None or not state.get("references"):
        log(f"  ❌ 未在 {workdir} 找到条目。请先运行第 1、2 步。")
        return None

    refs = state["references"]
    if dedup:
        dedup_against_library(refs, workdir, library_root=library_root, log=log)
        save_state(workdir, state)

    download_all(refs, workdir, concurrency=concurrency,
                 openalex_key=openalex_key, email=email, proxy=proxy,
                 state=state, log=log)
    # 库中已有副本的条目不必再留占位 RIS；尚未拿到 PDF 的条目刷新一份
    sync_pending_ris(workdir, refs, log=log)
    save_state(workdir, state)
    return refs


def _interactive_build_state_from_dois(workdir: Path) -> "dict | None":
    """无状态文件时的交互兜底：直接粘贴 DOI 列表构建条目。"""
    print("  ℹ️ 该目录还没有条目信息。可以直接粘贴 DOI 列表（一行一个）。")
    text = read_multiline_until_end("\n  请输入 DOI 列表：")
    dois = match_dois(text)
    if not dois:
        print("  ⚠ 未识别到任何 DOI，退出。")
        return None
    state = new_state(workdir)
    for seq, doi in enumerate(dois, 1):
        ref = normalize_reference(new_reference(
            id=f"ref_{seq:03d}", doi=doi, doi_source="user", confidence=1.0,
        ))
        state["references"].append(ref)
    save_state(workdir, state)
    print(f"  ✅ 已从输入构建 {len(dois)} 个条目。")
    print("  💡 提示：这些条目还没有元数据，建议先运行第 2 步补全"
          "（否则查重与临时文件名只能基于 DOI）。")
    return state


def main() -> None:
    parser = argparse.ArgumentParser(
        description="文献检索流水线第 3 步：多渠道轮询下载 PDF。"
    )
    parser.add_argument("--workdir", help="文献存储目录（含状态文件，PDF 下载到这里）")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_ARTICLE_CONCURRENCY,
                        help=f"文章并发下载数（默认 {DEFAULT_ARTICLE_CONCURRENCY}）")
    parser.add_argument("--openalex-key", default="",
                        help="OpenAlex API key（可选；也可放在 LLM_API_KEYS_PRIVATE.py 的 OPENALEX_API_KEY）")
    parser.add_argument("--email", default=CONTACT_EMAIL,
                        help="Unpaywall 等 API 的联系邮箱（默认使用公开占位邮箱）")
    parser.add_argument("--proxy", default=None,
                        help="代理，如 socks5h://127.0.0.1:1080（需要 requests[socks]）")
    parser.add_argument("--library-root", default=None,
                        help="查重的库根目录（默认取工作目录的上一级）")
    parser.add_argument("--no-dedup", action="store_true", help="跳过库内查重")
    args = parser.parse_args()

    print("═" * 62)
    print("  📥 文献检索流水线 · 第 3 步 · 多渠道 PDF 下载")
    print("═" * 62)

    workdir = Path(args.workdir).resolve() if args.workdir else ask_workdir()

    if load_state(workdir) is None:
        if args.workdir:
            print(f"  ❌ 未在 {workdir} 找到状态文件，请先运行第 1、2 步。")
            return
        if _interactive_build_state_from_dois(workdir) is None:
            return

    run(
        workdir,
        concurrency=args.concurrency,
        openalex_key=args.openalex_key,
        email=args.email,
        proxy=args.proxy,
        library_root=Path(args.library_root) if args.library_root else None,
        dedup=not args.no_dedup,
    )
    print("\n  👋 第 3 步完成。下一步：用 Lit_Retrieval_4_Rename_Ref.py（RIS 模式）标准化改名，"
          "再用 Lit_Retrieval_5_PDF_to_Markdown.py 做文本化。总驱动脚本 Lit_Retrieval_0.py "
          "可以自动完成这些步骤。")


if __name__ == "__main__":
    main()
