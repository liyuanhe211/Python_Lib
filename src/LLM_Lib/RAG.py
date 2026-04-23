"""
Indexing.py — 向量数据库构建与检索工具库

══════════════════════════════════════════════════════════════════════════════
  提供跨项目可复用的向量数据库操作功能：

  基础工具函数:
    - get_device():              检测 CUDA / MPS / CPU
    - load_embedding_model():    加载嵌入模型
                                  对 BAAI/bge-m3 强制使用 FlagEmbedding/BGE-M3
                                  其他模型使用 SentenceTransformer
    - load_chromadb_collection(): 加载或创建 ChromaDB 集合
    - chunk_text():              语义感知分块（段落边界 + 多语言句子切割）
    - read_text_file():          多编码文本文件读取
    - extract_category():        从文件名方括号中提取分类标签
    - sanitize_dirname():        将文件名转为安全的目录名
    - search_collection():       混合检索（Dense + BGE-M3 Sparse / BM25 回退）
    - format_search_results():   格式化检索结果为可读文本
    - index_chunks():            将文本分块嵌入并写入 ChromaDB

  高层工作流函数:
    - index_single_file():       索引单个 txt 文件 → db_shards/ 分片
    - index_folder():            索引文件夹中所有匹配文件 → db_shards/ 分片
    - merge_shards():            将 db_shards/ 合并至统一 ChromaDB
    - search_from_indexed_db():  一站式加载数据库并执行语义检索
    - download_model():          预下载嵌入模型到本地缓存

  命令行用法 (python -m LLM_Lib.RAG):

    # 交互模式（不传参数直接运行）
    python -m LLM_Lib.RAG

        依赖安装（一次装齐，避免装一个再报另一个）：
        uv pip install torch chromadb jieba rank-bm25 sentence-transformers \
            "transformers==4.46.3" "pyarrow<=20.0.0" \
            datasets==3.2.0 fsspec==2024.9.0 peft FlagEmbedding==1.3.5

    # 预下载嵌入模型（并行索引前运行一次）
    python -m LLM_Lib.RAG download [--model BAAI/bge-m3]
        
    # 索引单个文件或整个文件夹 → 生成分片到 db_shards/
    python -m LLM_Lib.RAG index <file_or_folder> [--db <db_dir>] [--pattern "*.txt"]

    # 将 db_shards/ 合并至 db/chroma_db/
    python -m LLM_Lib.RAG merge --db <db_dir> [--db-shards <shards_dir>]

    # 交互式 / 命令行语义检索
    python -m LLM_Lib.RAG query --db <db_dir> [-q <query>] [-k 8]

  目录结构约定:
    <base>/
      db/                   ← --db 指向此目录
        chroma_db/           ← 统一向量数据库
        file_list.txt
        indexed_files.txt
      db_shards/             ← 与 db/ 同级，每个文件一个分片子目录
        [标签] 文件名.txt__<hash>/
          chroma.sqlite3
          ...

  混合检索说明:
    BAAI/bge-m3 强制通过 FlagEmbedding 以 BGE-M3 三向量模式运行，同时提取
    dense 向量与 sparse 词权重。Sparse 权重以紧凑 JSON 存储在每个 chunk 的
    metadata（字段 sparse_weights），检索时通过 Dense + Sparse 双路混合打分，
    精确命中短片段也能获得高分。
    若当前环境未安装 FlagEmbedding 或缺少 BGEM3FlagModel，则直接报错并提示安装；
    不再回退至 SentenceTransformer。

  每个 chunk 的 metadata 字段:
    source_file    文件名（含扩展名）
    source_path    原始文件绝对路径
    book_title     书名（去掉分类前缀和扩展名）
    category       分类标签（从方括号前缀提取，如 "[画家画作个案]"）
    chunk_index    chunk 在文件内的序号（从 0 开始）
    total_chunks   该文件的 chunk 总数
    sparse_weights BGE-M3 稀疏词权重 JSON（仅 FlagEmbedding 后端）
    （其他字段可通过 document_metadata / chunk_metadata_factory 自定义，
     如 page_number、section、scan_folder 等溯源信息）

  Python API 用法:
    from LLM_Lib.RAG import (
        get_device, load_embedding_model, load_chromadb_collection,
        chunk_text, search_collection, format_search_results,
        index_single_file, index_folder, merge_shards,
        search_from_indexed_db, download_model,
    )
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import sys
import time
import traceback
import warnings
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable

# FlagEmbedding (BGEM3FlagModel) 内部用 tokenizer.encode()+pad() 而非
# tokenizer.__call__()，触发 transformers 的冗余建议性警告，在此统一屏蔽。
warnings.filterwarnings(
    "ignore",
    message=r".*using the `__call__` method is faster.*",
    category=UserWarning,
)
# jieba 源码中存在无效转义序列（\. \s），Python 3.12+ 对此发出 SyntaxWarning。
# 这是 jieba 本身的问题，在此屏蔽以避免干扰输出。
warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    module=r"jieba",
)

# Chromadb 通过 opentelemetry-proto 间接依赖 protobuf。
# 旧版 proto 描述符在 protobuf 4+ 运行时可能报 "Descriptors cannot be created directly"。
# 强制使用纯 Python 实现以规避此兼容性问题（对整体性能影响极小）。
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


INDEXING_QUALITY_INSTALL_COMMAND = (
    'uv pip install torch chromadb jieba rank-bm25 sentence-transformers '
    '"transformers==4.46.3" "pyarrow<=20.0.0" '
    'datasets==3.2.0 fsspec==2024.9.0 peft FlagEmbedding==1.3.5'
)

_INDEXING_QUALITY_DEPENDENCIES: tuple[tuple[str, str], ...] = (
    ("torch", "torch"),
    ("chromadb", "chromadb"),
    ("datasets==3.2.0", "datasets"),
    ("jieba", "jieba"),
    ("rank-bm25", "rank_bm25"),
    ("transformers==4.46.3", "transformers"),
    ("sentence-transformers", "sentence_transformers"),
    ("FlagEmbedding==1.3.5", "FlagEmbedding"),
    ("peft", "peft"),
)

_DEPENDENCY_CHECK_CACHE: bool = False


def _ensure_indexing_quality_dependencies(
    model_name: str = "BAAI/bge-m3",
) -> None:
    """
    质量优先模式下，集中检查索引与检索所需依赖。

    约束：
      - 所有关键依赖必须一次性可用；
      - 不因缺少依赖而降级到备选实现；
      - 对已知版本约束（pyarrow / fsspec）执行显式校验。
    """
    global _DEPENDENCY_CHECK_CACHE
    if _DEPENDENCY_CHECK_CACHE:
        return

    missing: list[str] = []
    broken: list[str] = []

    for package_name, module_name in _INDEXING_QUALITY_DEPENDENCIES:
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(package_name)
        except Exception as exc:
            broken.append(f"{package_name}: {exc}")

    if "bge-m3" in model_name.lower():
        try:
            from FlagEmbedding import BGEM3FlagModel  # type: ignore[import]
            if BGEM3FlagModel is None:
                broken.append("FlagEmbedding: BGEM3FlagModel 不可用")
        except Exception as exc:
            broken.append(f"FlagEmbedding.BGEM3FlagModel: {exc}")

    try:
        datasets_version = importlib_metadata.version("datasets")
        if datasets_version != "3.2.0":
            broken.append(
                f"datasets=={datasets_version}；当前要求 datasets==3.2.0"
            )
    except importlib_metadata.PackageNotFoundError:
        missing.append("datasets==3.2.0")

    try:
        transformers_version = importlib_metadata.version("transformers")
        if not transformers_version:
            broken.append(
                "transformers=="
                f"{transformers_version}；当前要求 transformers==4.46.3"
            )
    except importlib_metadata.PackageNotFoundError:
        missing.append("transformers==4.46.3")

    try:
        pyarrow_version = importlib_metadata.version("pyarrow")
        pyarrow_parts = tuple(
            int(part) for part in re.findall(r"\d+", pyarrow_version)[:3]
        )
        if pyarrow_parts and pyarrow_parts > (20, 0, 0):
            broken.append(
                f"pyarrow=={pyarrow_version}；当前要求 pyarrow<=20.0.0"
            )
    except importlib_metadata.PackageNotFoundError:
        missing.append("pyarrow<=20.0.0")

    try:
        fsspec_version = importlib_metadata.version("fsspec")
        if fsspec_version != "2024.9.0":
            broken.append(
                f"fsspec=={fsspec_version}；当前要求 fsspec==2024.9.0"
            )
    except importlib_metadata.PackageNotFoundError:
        missing.append("fsspec==2024.9.0")

    if missing or broken:
        message_lines = [
            "当前实现已启用质量优先模式；索引与检索依赖必须一次性完整可用。",
        ]
        if missing:
            message_lines.append(
                f"缺少依赖: {', '.join(sorted(set(missing)))}"
            )
        if broken:
            message_lines.append("依赖异常:")
            message_lines.extend(f"- {item}" for item in broken)
        message_lines.extend([
            "",
            "建议一次性安装命令:",
            f"  {INDEXING_QUALITY_INSTALL_COMMAND}",
        ])
        raise RuntimeError("\n".join(message_lines))

    _DEPENDENCY_CHECK_CACHE = True

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  设备检测                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def get_device() -> str:
    """
    检测可用计算设备，返回 'cuda' / 'mps' / 'cpu'。

    优先级: CUDA GPU > Apple MPS > CPU
    """

    import torch

    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except AttributeError:
        pass
    return "cpu"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  嵌入模型加载                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _import_sentence_transformer() -> Any:
    """
    导入 SentenceTransformer，并在 Linux/HPC 环境下提供更可诊断的错误信息。

    常见失败原因不是模型本身，而是底层二进制依赖链冲突，例如：
      sentence_transformers -> sklearn -> pyarrow -> libstdc++
    当系统上的 libstdc++.so.6 版本过旧时，会报错缺少 CXXABI / GLIBCXX 符号。
    """
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except Exception as exc:
        error_text = "".join(
            traceback.format_exception_only(type(exc), exc)
        ).strip()
        lower_text = error_text.lower()

        if any(
            token in lower_text
            for token in (
                "libstdc++",
                "cxxabi_",
                "glibcxx_",
                "pyarrow",
            )
        ):
            raise RuntimeError(
                "无法导入 sentence-transformers；当前更像是 Linux/HPC 的二进制库冲突，"
                "而不是模型 BAAI/bge-m3 本身有问题。\n\n"
                "典型链路：sentence_transformers -> sklearn -> pyarrow -> libstdc++\n"
                "当前错误通常表示系统实际加载到的 libstdc++.so.6 版本过旧，"
                "缺少 pyarrow / libarrow 需要的 CXXABI / GLIBCXX 符号。\n\n"
                "建议优先尝试：\n"
                "1. 在当前环境中重装兼容版本：sentence-transformers、scikit-learn、pyarrow。\n"
                "2. 若在 HPC 上混用了系统 GCC runtime 与 Conda runtime，确保优先加载 Conda 自带的 libstdc++.so.6。\n"
                "3. 若不需要 pyarrow，可尝试卸载 pyarrow 后重新导入 sklearn / sentence-transformers。\n\n"
                f"原始导入错误: {error_text}"
            ) from exc

        raise RuntimeError(
            "无法导入 sentence-transformers。请确认当前环境已安装兼容版本的 "
            "sentence-transformers、transformers、torch、scikit-learn。\n\n"
            f"原始导入错误: {error_text}"
        ) from exc


class _BGEM3Wrapper:
    """
    封装 ``FlagEmbedding.BGEM3FlagModel``，提供：

    * ``encode()``        — 仅 dense 向量，API 与 SentenceTransformer.encode() 兼容。
    * ``encode_hybrid()`` — dense + sparse 词权重（可选 ColBERT 多向量）。
    * ``is_bgem3 = True`` — 供调用方检测当前后端类型。

    质量优先模式要求 ``FlagEmbedding`` 与其相关依赖在当前环境中一次性完整安装。
    """

    def __init__(self, flag_model: Any) -> None:
        self._model = flag_model
        self.is_bgem3: bool = True

    def encode(
        self,
        sentences: list[str],
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        batch_size: int = 12,
        **_kwargs: Any,
    ) -> Any:
        """
        Dense-only 编码 — API 与 SentenceTransformer.encode() 兼容。

        返回 np.ndarray (N, D)；normalize_embeddings 参数保留但 BGE-M3
        输出已归一化，无需额外处理。
        """
        import numpy as np

        out = self._model.encode(
            sentences,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
            batch_size=batch_size,
        )
        vecs = out["dense_vecs"]
        if convert_to_numpy:
            return np.array(vecs, dtype="float32")
        return vecs

    def encode_hybrid(
        self,
        sentences: list[str],
        return_colbert: bool = False,
        batch_size: int = 12,
    ) -> dict[str, Any]:
        """
        全混合编码：dense + sparse 词权重（可选 ColBERT token 向量）。

        Returns:
            dict，包含以下键：
              ``dense_vecs``      np.ndarray (N, D)
              ``lexical_weights`` list[dict]  — token_id(str/int) → weight(float)
              ``colbert_vecs``    list[np.ndarray] | None
        """
        return self._model.encode(
            sentences,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=return_colbert,
            batch_size=batch_size,
        )


def _try_load_bgem3_flag(
    model_name: str,
    device: str,
    local_files_only: bool,  # noqa: ARG001  (FlagEmbedding 自行管理缓存)
) -> "_BGEM3Wrapper":
    """
    尝试通过 FlagEmbedding 加载 BGEM3FlagModel。

    Returns:
        ``_BGEM3Wrapper`` 实例。

    Raises:
        RuntimeError: 当前环境缺少 FlagEmbedding / BGEM3FlagModel，或模型加载失败。
    """
    _ensure_indexing_quality_dependencies(model_name)

    try:
        from FlagEmbedding import BGEM3FlagModel  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "当前模型要求使用 FlagEmbedding.BGEM3FlagModel，但当前环境未提供该实现。\n\n"
            f"建议一次性安装命令:\n  {INDEXING_QUALITY_INSTALL_COMMAND}\n\n"
            f"原始导入错误: {exc}"
        ) from exc

    try:
        flag_model = BGEM3FlagModel(
            model_name,
            use_fp16=(device != "cpu"),
        )
        return _BGEM3Wrapper(flag_model)
    except Exception as exc:
        raise RuntimeError(
            "FlagEmbedding.BGEM3FlagModel 初始化失败，无法继续加载 BGE-M3。\n\n"
            "请确认：\n"
            "1. 已按质量优先模式一次性安装全部依赖；\n"
            "2. 当前环境中的 transformers / torch 与 FlagEmbedding 版本兼容；\n"
            "3. 当前环境满足 datasets==3.2.0、pyarrow<=20.0.0 且 fsspec==2024.9.0；\n"
            "4. 模型名称可用，且本地缓存未损坏。\n\n"
            f"原始加载错误: {exc}"
        ) from exc


def load_embedding_model(
    model_name: str,
    device: str = "",
    *,
    local_files_only: bool = True,
    verbose: bool = True,
) -> Any:
    """
    加载嵌入模型。

    对于含 ``bge-m3`` 的模型名，强制使用 FlagEmbedding 后端
    （需安装 ``FlagEmbedding``）以获得 dense + sparse 混合检索能力；
    若任何质量相关依赖缺失或版本不满足要求，则直接报错。

    Args:
        model_name:       模型名称（如 'BAAI/bge-m3'）。
        device:           计算设备（留空则自动检测）。
        local_files_only: 是否仅从本地缓存加载（默认 True）。
        verbose:          是否打印加载成功信息（默认 True）。

    Returns:
        ``_BGEM3Wrapper``（FlagEmbedding 后端）或 SentenceTransformer 实例。
        两者均实现 ``encode(sentences, ...)`` 接口。
    """
    _ensure_indexing_quality_dependencies(model_name)
    device = device or get_device()

    # ── BGE-M3 强制使用 FlagEmbedding 后端（支持 dense + sparse）──
    if "bge-m3" in model_name.lower():
        wrapper = _try_load_bgem3_flag(model_name, device, local_files_only)
        if verbose:
            print(f"  [BGE-M3] FlagEmbedding 后端加载成功 (device={device})")
        return wrapper

    # ── SentenceTransformer 回退 ──
    SentenceTransformer = _import_sentence_transformer()

    if local_files_only:
        try:
            return SentenceTransformer(
                model_name, device=device, local_files_only=True
            )
        except Exception:
            print(
                f"模型 {model_name} 本地未找到，正在从 HuggingFace 下载……"
            )

    return SentenceTransformer(
        model_name, device=device, local_files_only=False
    )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ChromaDB 操作                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def load_chromadb_collection(
    db_dir: Path | str,
    collection_name: str,
    *,
    create_if_missing: bool = False,
    space: str = "cosine",
) -> tuple[Any, Any]:
    """
    加载或创建 ChromaDB 集合。

    Args:
        db_dir:             ChromaDB 持久化目录。
        collection_name:    集合名称。
        create_if_missing:  若集合不存在是否自动创建。
        space:              距离度量（默认 cosine）。

    Returns:
        (client, collection) 元组。
    """
    import chromadb

    db_dir = Path(db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(db_dir))

    if create_if_missing:
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": space},
        )
    else:
        collection = client.get_collection(collection_name)

    return client, collection


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  文本分块（语义感知 + 多语言）                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# 多语言句子边界正则
# 支持：中文、日文（CJK 标点）；英语、德语（西方标点）
#   · CJK 句末：。！？…  以及较弱的 ；
#   · 西方句末：. ! ? 后必须跟空白且下一个字符非数字，
#     避免切断 "3.14"、序号 "1. "、缩写 "Dr."、德语 "bzw."
#   · 换行符视为段落内分句边界
_SENT_BOUNDARY_RE = re.compile(
    r"(?<=[。！？…])\s*"
    r"|(?<=；)\s*"
    r"|(?<=[.!?])\s+(?=[^\s\d])"
    r"|(?<=\n)\s*(?=\S)",
)


def _split_sentences_multilang(text: str) -> list[str]:
    """
    将段落文本切割为句子列表，支持中 / 英 / 德 / 日等多语言。

    切割规则（详见 ``_SENT_BOUNDARY_RE``）：
      - CJK 句末标点（。！？…；）后切割。
      - 西方句末标点（. ! ?）后跟空白 + 非数字字符时切割，
        以避免截断小数、序号和常见缩写。
      - 换行符也视为分句边界。
    """
    parts = _SENT_BOUNDARY_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _content_len(text: str) -> int:
    """Character count excluding heading lines (``#`` prefix) and blank lines.

    Used by ``chunk_text()`` so that injected heading decorations do not
    consume the chunk-size budget.  For plain text without ``#`` lines,
    this returns the sum of all non-blank line lengths.
    """
    return sum(
        len(line.strip())
        for line in text.split('\n')
        if not line.strip().startswith('#')
    )


def _dedup_heading_blocks_in_chunk(text: str) -> str:
    """Remove duplicate heading lines within a single chunk.

    When multiple paragraphs carrying the same ``#Heading`` prefix are
    merged into one chunk, the heading appears once per original paragraph.
    This keeps only the first occurrence of each unique heading line
    (compared after strip), and collapses excessive blank lines left behind.
    """
    lines = text.split('\n')
    seen: set[str] = set()
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#'):
            if stripped in seen:
                continue
            seen.add(stripped)
        result.append(line)
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(result)).strip()


def chunk_text(
    paragraphs: str | list[str],
    chunk_size: int = 800,
    chunk_overlap: int = 200,
    min_chunk_len: int = 50,
    max_chunk_len: int = 1200,
) -> list[str]:
    """
    将文本切割为语义感知的重叠分块，支持中 / 英 / 德 / 日多语言。

    与旧版纯字符窗口切割不同，本实现：
      1. 优先在段落（``\\n\\n``）边界处切分，保留段落完整性。
      2. 超过 ``max_chunk_len`` 的长段落按句子边界二次切割。
      3. 过短的段落/句子向后合并，直到接近 ``chunk_size``。
      4. 相邻分块通过"保留前一分块末尾若干单元"实现重叠，
         重叠量以字符数（``chunk_overlap``）为目标，但对齐到
         段落/句子边界（实际重叠可能略大于目标值）。

    Args:
        paragraphs:    待分块文本（str）或已切好的段落列表（list[str]）。
                       传入 str 时按 ``\\n\\n`` 自动切段落。
        chunk_size:    目标分块字符数（默认 600）。
        chunk_overlap: 相邻分块重叠目标字符数（默认 200），
                       对齐到段落/句子边界，实际重叠量可能略大。
        min_chunk_len: 丢弃短于此长度的分块（默认 50）。
        max_chunk_len: 超过此长度的段落按句子边界切割（默认 1000）。

    Returns:
        分块字符串列表。

    Backward compatibility:
        第一个参数由 ``text: str`` 改为 ``paragraphs: str | list[str]``，
        接受 str 时行为等价（自动按 ``\\n\\n`` 切段落），
        旧代码以位置参数传入 str 无需修改。
    """
    # ── Step 1：规范化为段落列表 ──
    if isinstance(paragraphs, str):
        raw_paras: list[str] = re.split(r"\n{2,}", paragraphs)
    else:
        raw_paras = list(paragraphs)

    # ── Step 2：超长段落按句子边界二次切割 ──
    units: list[str] = []
    for para in raw_paras:
        para = para.strip()
        if not para:
            continue
        if _content_len(para) <= max_chunk_len:
            units.append(para)
        else:
            sents = _split_sentences_multilang(para)
            units.extend(sents if sents else [para])

    if not units:
        return []

    # ── Step 3：贪心装箱 + 对齐到 unit 边界的重叠 ──
    chunks: list[str] = []
    i = 0

    while i < len(units):
        # 从 i 开始向后贪心收集，直到超出 chunk_size
        current: list[str] = []
        current_len = 0
        j = i

        while j < len(units):
            unit_clen = _content_len(units[j])
            sep_len = 1 if (current and unit_clen) else 0
            if current_len + sep_len + unit_clen <= chunk_size:
                current.append(units[j])
                current_len += sep_len + unit_clen
                j += 1
            else:
                break

        if not current:
            # 单个 unit 超过 chunk_size：强制纳入，避免无限循环
            current = [units[i]]
            j = i + 1

        chunk_str = _dedup_heading_blocks_in_chunk("\n\n".join(current))
        if _content_len(chunk_str) >= min_chunk_len:
            chunks.append(chunk_str)

        if j >= len(units):
            break

        # ── 计算下一分块起始位置（overlap 对齐到 unit 边界） ──
        if chunk_overlap > 0:
            overlap_chars = 0
            k = j                        # 从 j 向左回溯
            while k > i + 1:
                k -= 1
                overlap_chars += _content_len(units[k]) + 1
                if overlap_chars >= chunk_overlap:
                    break
            i = k                        # 下一分块从 k 开始（k ≥ i+1，保证进度）
        else:
            i = j

    return chunks


# ── DeepSeekOCR 标题块正则 ──────────────────────────────────────────────────────
# DeepSeekOCR 索引阶段将标题以 "#Title\n#Title" 格式注入 chunk（连续重复两行），
# 前后有多个换行符。此正则用于匹配并剥离整个标题块。
_HEADING_BLOCK_RE = re.compile(r'\n*(#[^\n]+)\n\1\n*')


def _dedup_consecutive_heading_lines(text: str) -> str:
    """去除文本中连续重复的标题行（以 # 开头的行）。

    DeepSeekOCR 索引阶段为增强嵌入效果而将标题写入两次，
    输出时只显示一次。
    """
    lines = text.split('\n')
    result: list[str] = []
    for line in lines:
        if (result
                and line == result[-1]
                and line.strip().startswith('#')):
            continue
        result.append(line)
    return '\n'.join(result)


def _merge_adjacent_chunks_dedup(ordered_docs: list[str]) -> str:
    """
    合并相邻 chunk 文本并去除重叠部分。

    DeepSeekOCR 结果优化：chunk 中可能包含由索引阶段注入的标题块
    （格式为 ``\\n\\n\\n#Title\\n#Title\\n\\n\\n``）。这些标题块会干扰
    公共子串检测（导致错误的短匹配）。处理流程：

      1. 剥离所有标题块，记录每个唯一标题及其后紧跟的文本片段；
      2. 在纯文本上执行重叠检测与自然拼接（不插入 "..."）；
      3. 按文本关联位置将唯一标题插回合并结果中最早出现的位置；
      4. 去除连续重复的标题行（输出时每个标题只显示一次）。
    """
    if not ordered_docs:
        return ""
    if len(ordered_docs) == 1:
        return _dedup_consecutive_heading_lines(ordered_docs[0])

    # ── Step 1: 剥离标题，记录 (heading, 随后的文本片段) 用于重新定位 ──
    stripped_docs: list[str] = []
    # heading_contexts: 按首次出现顺序记录的 (标题文本, 标题后的文本片段)
    heading_contexts: list[tuple[str, str]] = []
    seen_headings: set[str] = set()

    for doc in ordered_docs:
        for m in _HEADING_BLOCK_RE.finditer(doc):
            h = m.group(1)
            if h not in seen_headings:
                seen_headings.add(h)
                # 取标题后最多 200 字符，剥离其中可能嵌套的标题块，
                # 保留前 80 字符作为定位片段
                after_raw = doc[m.end():m.end() + 200].strip()
                after_clean = _HEADING_BLOCK_RE.sub('', after_raw).strip()
                snippet = after_clean[:80] if after_clean else ''
                heading_contexts.append((h, snippet))
        cleaned = _HEADING_BLOCK_RE.sub('\n', doc).strip()
        stripped_docs.append(cleaned)

    # ── Step 2: 在无标题文本上执行重叠检测，自然拼接（不加 "..."） ──
    merged = stripped_docs[0]
    for i in range(1, len(stripped_docs)):
        curr = stripped_docs[i]
        overlap_len = _find_overlap(merged, curr)
        if overlap_len > 0:
            merged = merged + curr[overlap_len:]
        else:
            merged = merged + "\n" + curr

    # ── Step 3: 将唯一标题插回到合并文本的对应位置 ──
    # 用标题后文本片段在合并结果中定位，从后向前插入以保持偏移量正确
    insertions: list[tuple[int, int, str]] = []  # (position, order, block)
    for order, (heading, snippet) in enumerate(heading_contexts):
        heading_block = f"\n\n\n{heading}\n\n\n"
        if not snippet:
            insertions.append((0, order, heading_block))
            continue
        pos = merged.find(snippet)
        if pos < 0 and len(snippet) > 30:
            pos = merged.find(snippet[:30])
        if pos >= 0:
            insertions.append((pos, order, heading_block))

    # 按位置降序、同位置按原始出现顺序升序，从后往前插入
    insertions.sort(key=lambda x: (-x[0], x[1]))
    for pos, _, block in insertions:
        merged = merged[:pos] + block + merged[pos:]

    # ── Step 4: 去除连续重复的标题行 ──
    merged = _dedup_consecutive_heading_lines(merged)

    return merged.strip()


def _find_overlap(prev: str, curr: str) -> int:
    """
    检测 prev 末尾与 curr 开头的最长重叠长度。

    从 curr 开头取逐渐增长的前缀，检查是否出现在 prev 的末尾。
    为避免 O(n^2)，仅检查到两个文本较短者长度的一半。
    """
    max_check = min(len(prev), len(curr)) // 2
    if max_check < 20:
        return 0

    best = 0
    # 在 prev 末尾搜索 curr 开头的文本片段
    # 使用逐步增长的窗口匹配
    for length in range(20, max_check + 1, 10):
        snippet = curr[:length]
        pos = prev.rfind(snippet)
        if pos >= 0:
            # 从 pos 开始向后扩展匹配
            remaining_prev = prev[pos:]
            remaining_curr = curr[:len(remaining_prev)]
            if remaining_prev == remaining_curr:
                best = len(remaining_prev)
            elif best == 0:
                best = length
        elif best > 0:
            break
    return best


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  多编码文本读取                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# 默认编码优先级列表，覆盖：
# UTF-8、中日韩、西欧、BOM
DEFAULT_ENCODINGS: tuple[str, ...] = (
    "utf-8-sig",
    "utf-8",
    "utf-16",
    "gb18030",
    "gbk",
    "big5hkscs",
    "big5",
    "cp932",
    "shift_jis",
    "euc_jp",
    "cp1252",
    "latin-1",
)


def read_text_file(
    filepath: Path | str,
    encodings: tuple[str, ...] = DEFAULT_ENCODINGS,
) -> str:
    """
    尝试以多种编码读取文本文件。

    按 encodings 顺序尝试，首个成功的编码将被使用。

    Args:
        filepath:   文件路径。
        encodings:  尝试的编码列表（按优先级排列）。

    Returns:
        文件内容字符串。读取失败时返回空字符串。
    """
    filepath = Path(filepath)
    for enc in encodings:
        try:
            return filepath.read_text(encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    print(f"  [WARN] 无法解码文件 {filepath.name}，已跳过。")
    return ""


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  分类标签提取                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def extract_category(filename: str) -> str:
    """
    从文件名的方括号前缀中提取分类标签。

    示例:
        '[画家画作个案] 陈洪绶.txt'  →  '画家画作个案'
        '无标签文件.txt'             →  '未分类'
    """
    match = re.match(r"^\[([^\]]+)\]", filename)
    return match.group(1) if match else "未分类"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  目录名安全化                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def sanitize_dirname(name: str) -> str:
    """
    将文件名转换为安全的目录名，添加短哈希后缀防止截断冲突。

    示例:
        '[画家画作个案] 陈洪绶.txt'  →  '[画家画作个案] 陈洪绶.txt__a1b2c3d4'

    Args:
        name: 原始文件名。

    Returns:
        安全的目录名字符串。
    """
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    safe = safe[:120]  # 截断以兼容 Windows 路径长度限制
    short_hash = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
    return f"{safe}__{short_hash}"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Metadata 工具                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

MetadataPrimitive = str | int | float | bool
ChunkMetadataFactory = Callable[[int, str, Path], dict[str, MetadataPrimitive] | None]
FolderMetadataFactory = Callable[[Path], dict[str, MetadataPrimitive] | None]
FolderChunkMetadataFactory = Callable[[Path, int, str], dict[str, MetadataPrimitive] | None]

# 系统保留字段：不允许用户通过自定义 metadata 覆盖
SYSTEM_METADATA_KEYS: frozenset[str] = frozenset({
    "source_file",
    "source_path",
    "book_title",
    "category",
    "chunk_index",
    "total_chunks",
    "sparse_weights",
})


def _normalize_user_metadata(
    metadata: dict[str, Any] | None,
) -> dict[str, MetadataPrimitive]:
    """
    规范化用户自定义 metadata。

    约束：
      - key 必须是非空字符串；
      - 不允许覆盖系统保留字段；
      - value 仅允许 str / int / float / bool；
      - value=None 会被忽略。
    """
    if not metadata:
        return {}

    normalized: dict[str, MetadataPrimitive] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("metadata 的 key 必须是非空字符串")
        if key in SYSTEM_METADATA_KEYS:
            raise ValueError(
                f"metadata 字段 {key!r} 为系统保留字段，不能被用户覆盖"
            )
        if value is None:
            continue
        if isinstance(value, bool):
            normalized[key] = value
        elif isinstance(value, (str, int, float)):
            normalized[key] = value
        else:
            raise TypeError(
                f"metadata 字段 {key!r} 的值类型 {type(value).__name__} 不受支持；"
                "仅支持 str / int / float / bool / None"
            )

    return normalized


def _extract_book_title(filename: str) -> str:
    """
    从文件名中提取书名（仅去掉扩展名，保留方括号分类前缀）。

    示例:
        '[画家画作个案] 陈洪绶.txt'  →  '[画家画作个案] 陈洪绶'
        '达芬奇传记.txt'             →  '达芬奇传记'
        'Vasari_Lives.txt'           →  'Vasari_Lives'
    """
    return re.sub(r"\.[^.]+$", "", filename).strip()


def _build_chunk_metadata(
    *,
    filename: str,
    source_path: str,
    book_title: str,
    category: str,
    chunk_index: int,
    total_chunks: int,
    document_metadata: dict[str, Any] | None = None,
    chunk_metadata: dict[str, Any] | None = None,
) -> dict[str, MetadataPrimitive]:
    """
    构造单个分块的 metadata。

    系统字段（source_file / source_path / book_title / category /
    chunk_index / total_chunks）始终写入。用户字段附加其后。
    sparse_weights 由索引逻辑在编码完成后单独写入。

    用户可通过 document_metadata / chunk_metadata_factory 附加自定义
    溯源字段，如 page_number（原始 PDF 页码）、scan_folder（扫描图像
    目录）、section（章节标题）等。
    """
    metadata: dict[str, MetadataPrimitive] = {
        "source_file": filename,
        "source_path": source_path,
        "book_title": book_title,
        "category": category,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
    }
    metadata.update(_normalize_user_metadata(document_metadata))
    metadata.update(_normalize_user_metadata(chunk_metadata))
    return metadata


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Sparse 权重工具（BGE-M3）                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _serialize_sparse_weights(
    weights: dict,
    top_k: int = 0,
    chunk_text_len: int = 0,
) -> str:
    """
    将 BGE-M3 的稀疏词权重序列化为紧凑 JSON 字符串。

    根据 chunk 文本长度动态决定保留的 token 数量：
      - chunk_text_len <= 300:  top_k = 128
      - chunk_text_len <= 800:  top_k = 192
      - chunk_text_len >  800:  top_k = 256
    若调用方显式传入 top_k > 0，则直接使用该值。

    Args:
        weights:        FlagEmbedding 返回的 lexical_weights dict
                        （token_id str/int → weight float）。
        top_k:          保留的最大词数；0 表示根据 chunk_text_len 自动决定。
        chunk_text_len: 当前 chunk 的字符数，用于动态计算 top_k。

    Returns:
        紧凑 JSON 字符串；输入为空时返回空字符串。
    """
    if not weights:
        return ""
    if top_k <= 0:
        if chunk_text_len <= 300:
            top_k = 128
        elif chunk_text_len <= 800:
            top_k = 192
        else:
            top_k = 256
    top = sorted(weights.items(), key=lambda kv: -abs(float(kv[1])))[:top_k]
    compact = {str(k): round(float(v), 4) for k, v in top}
    return json.dumps(compact, separators=(",", ":"))


def _sparse_dot(
    q_weights: dict[str, float],
    doc_weights_json: str,
) -> float:
    """
    计算查询稀疏权重与文档稀疏权重的点积（BGE-M3 Sparse Similarity）。

    Args:
        q_weights:        查询稀疏权重 dict，key 为字符串 token_id。
        doc_weights_json: 文档 metadata 中的 sparse_weights JSON 字符串。

    Returns:
        稀疏点积分数；doc_weights_json 为空或解析失败时返回 0.0。
    """
    if not doc_weights_json:
        return 0.0
    try:
        d_weights: dict[str, float] = json.loads(doc_weights_json)
    except Exception:
        return 0.0
    score = 0.0
    for token, q_w in q_weights.items():
        d_w = d_weights.get(token)
        if d_w is not None:
            score += float(q_w) * float(d_w)
    return score


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  内容去重（文本哈希注册表）                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def compute_content_hash(text: str) -> str:
    """
    计算文本内容的规范哈希值，用于跨文件去重。

    规范化处理：将所有连续空白（换行、空格、制表符）压缩为单个空格，
    忽略首尾空白。使内容相同但排版（换行数量、空格多少）不同的文本
    产生相同哈希，与文件名、路径、页码等元数据无关。

    Args:
        text: 待哈希的**完整文档**文本（非单个 chunk）。

    Returns:
        64 字符 SHA-256 十六进制字符串。
    """
    normalized = re.sub(r"\s+", " ", text.strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _shard_hash_file(shard_dir: Path) -> Path:
    """分片目录中存储内容哈希的伴随文件路径。"""
    return shard_dir / "content_hash.txt"


def _registry_file(db_dir: Path) -> Path:
    """统一数据库目录中内容哈希注册表的路径。"""
    return db_dir / "content_registry.json"


def load_content_registry(db_dir: Path | str) -> dict[str, dict]:
    """
    加载统一数据库的内容哈希注册表。

    注册表文件：``<db_dir>/content_registry.json``
    格式：``{content_hash: {"source_file": "...", "book_title": "...", "indexed_at": "..."}}``

    Returns:
        注册表 dict；文件不存在或解析失败时返回空 dict。
    """
    path = _registry_file(Path(db_dir))
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_content_registry(registry: dict, db_dir: Path | str) -> None:
    """
    保存内容哈希注册表到 ``<db_dir>/content_registry.json``。
    """
    path = _registry_file(Path(db_dir))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def check_content_duplicate(
    text: str,
    *,
    db_dir: Path | str | None = None,
    shards_dir: Path | str | None = None,
) -> tuple[bool, str | None, str]:
    """
    快速检测文本内容是否已被索引，防止同一文本以不同路径/文件名重复入库。

    仅比较规范化后的文本内容（忽略空白差异），不考虑文件名、路径、
    页码等元数据。

    检测顺序（从快到慢）：
      1. 若提供 ``db_dir``：查询统一数据库的 content_registry.json（O(1)）。
      2. 若提供 ``shards_dir``：遍历各分片目录下的 content_hash.txt 文件。

    Args:
        text:       待检查的完整文档文本。
        db_dir:     统一数据库目录（含 content_registry.json 的目录）。
        shards_dir: 分片父目录（各分片子目录下含 content_hash.txt）。

    Returns:
        ``(is_duplicate, original_source, content_hash)``

        * ``is_duplicate``     True = 内容已存在
        * ``original_source``  首次索引该内容的文件名或分片目录名；
                               未知时为 None
        * ``content_hash``     本次计算的哈希值（无论是否重复均返回）
    """
    content_hash = compute_content_hash(text)

    # ── 检查统一数据库注册表 ──
    if db_dir is not None:
        registry = load_content_registry(Path(db_dir))
        if content_hash in registry:
            original = registry[content_hash].get("source_file")
            return True, original, content_hash

    # ── 遍历分片的 content_hash.txt ──
    if shards_dir is not None:
        shards_path = Path(shards_dir)
        if shards_path.exists():
            for shard_subdir in shards_path.iterdir():
                if not shard_subdir.is_dir():
                    continue
                hash_file = _shard_hash_file(shard_subdir)
                if not hash_file.exists():
                    continue
                try:
                    stored = hash_file.read_text(encoding="utf-8").strip()
                    if stored == content_hash:
                        original = shard_subdir.name.rsplit("__", 1)[0]
                        return True, original, content_hash
                except Exception:
                    continue

    return False, None, content_hash


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  BM25 工具（rank_bm25 回退，非 BGE-M3 环境使用）                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _tokenize(text: str) -> list[str]:
    """
    分词辅助函数。

    质量优先模式下强制使用 jieba，不再回退到简化分词。
    """
    import jieba

    return list(jieba.cut_for_search(text))


def _perform_bm25_search(
    query: str,
    documents: list[str],
    doc_ids: list[str],
    top_k: int,
) -> list[tuple[str, float]]:
    """
    执行 BM25 检索。

    注意：此函数每次调用都会对所有文档重建 BM25 索引，
    适合数据量较小（< 5 万 chunks）的场景。
    安装 FlagEmbedding 后，BGE-M3 Sparse 替代此方案，不再调用本函数。
    """
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [_tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    results = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)
    return results[:top_k]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Cross-Encoder 重排序                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

_cross_encoder_cache: dict[str, Any] = {}


def _load_cross_encoder(model_name: str = "BAAI/bge-reranker-v2-m3", verbose: bool = True) -> Any:
    """
    加载 Cross-Encoder 重排序模型。

    优先使用 FlagEmbedding 的 FlagReranker；若不可用则回退到
    sentence-transformers 的 CrossEncoder。

    返回一个具有 compute_score(pairs) 方法的对象。
    """
    if model_name in _cross_encoder_cache:
        return _cross_encoder_cache[model_name]

    try:
        from FlagEmbedding import FlagReranker  # type: ignore[import]
        reranker = FlagReranker(model_name, use_fp16=False)
        _cross_encoder_cache[model_name] = reranker
        if verbose:
            print(f"  [Reranker] FlagReranker 加载成功: {model_name}")
        return reranker
    except ImportError:
        pass

    try:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(model_name)
        _cross_encoder_cache[model_name] = reranker
        if verbose:
            print(f"  [Reranker] CrossEncoder 加载成功: {model_name}")
        return reranker
    except ImportError:
        pass

    raise RuntimeError(
        f"无法加载 Cross-Encoder 模型 {model_name}。\n"
        "请安装 FlagEmbedding 或 sentence-transformers。"
    )


def _rerank_with_cross_encoder(
    query: str,
    hits: list[dict],
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3",
    progress_callback: Callable[[str], None] | None = None,
) -> list[dict]:
    """
    使用 Cross-Encoder 对检索候选进行重排序（分批处理，支持进度回调）。

    Args:
        query:              原始查询文本。
        hits:               bi-encoder 检索返回的候选列表。
        reranker_model_name: Cross-Encoder 模型名称。
        progress_callback:  进度回调，接收进度消息字符串。

    Returns:
        按 Cross-Encoder 分数降序排列的候选列表。
    """
    import math

    if not hits:
        return hits

    reranker = _load_cross_encoder(reranker_model_name)
    pairs = [(query, h["document"]) for h in hits]

    batch_size = 4
    scores: list[float] = []
    total = len(pairs)
    for i in range(0, total, batch_size):
        batch = pairs[i : i + batch_size]
        batch_scores = reranker.compute_score(batch)
        if isinstance(batch_scores, (int, float)):
            batch_scores = [batch_scores]
        scores.extend(float(s) for s in batch_scores)
        done = min(i + batch_size, total)
        if progress_callback:
            # \r 前缀表示覆盖上一行（前端可据此实现"替换最后一行"显示）
            prefix = "\r" if i > 0 else ""
            progress_callback(f"{prefix}  [Rerank] {done}/{total} 已重排序…")

    for hit, score in zip(hits, scores):
        hit["reranker_score"] = score
        norm_score = 1.0 / (1.0 + math.exp(-score))
        hit["distance"] = max(0.0, 1.0 - norm_score)

    return sorted(hits, key=lambda h: h["reranker_score"], reverse=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  LLM 查询扩展                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _expand_query_with_llm(query: str, *, confirm: bool = False) -> list[str]:
    """
    使用 Gemini Flash 模型对查询进行扩展，生成同义词和相关术语。

    返回扩展词列表（不含原始查询），用于多查询检索。
    调用方应将原始查询与扩展词分别编码检索，再合并候选集。

    Args:
        query:   原始查询文本。
        confirm: 若为 True，LLM 生成后暂停，允许用户确认/编辑扩展结果
                 或重新定义 LLM 输入后再次运行。

    Returns:
        扩展词列表（每个元素为一个短语/术语）。若 LLM 调用失败则返回空列表。
    """
    try:
        from LLM_Lib.LLM import call_gemini, GEMINI_FLASH
    except ImportError:
        return []

    prompt = f"""You are a search query expansion assistant for a semantic vector search system. Given the user's search query, generate a SHORT list of the most semantically relevant alternative phrasings and key terms.

Rules:
- Output ONLY the expanded terms, one per line
- Each term should be a COMPLETE, self-contained phrase that could independently retrieve relevant documents (not single isolated words)
- Prioritize: direct synonyms > closely related technical terms > translations
- The core concept should cover multiple languages, including Chinese, English and German
  - For example, if the query is in English, include 1-2 Chinese translations and 1-2 German translations of the core concept
- Convert between Latin scientific names and common names when applicable (e.g., leopard gecko ↔ Eublepharis macularius)
- Do NOT include vague or overly broad terms that would match unrelated documents
- Do NOT repeat the original query
- Output exactly 5-8 terms/phrases — quality over quantity

Query: {query}

Expanded terms:"""

    def _parse_terms(text: str) -> list[str]:
        """将 LLM 输出解析为扩展词列表，过滤空行。"""
        return [line.strip().lstrip("-•*0123456789.) ") for line in text.strip().splitlines() if line.strip()]

    try:
        response = call_gemini(
            prompt,
            model=GEMINI_FLASH,
            confirm=False,
            stream=True,
            verbose=True,
            show_cost=True,
            reasoning=False,
        )
        if response and response.strip():
            terms = _parse_terms(response)
            print("  [LLM Expansion] 扩展要素:\n", "\n".join(terms))
            print("\n\n\n")

            if confirm:
                while True:
                    all_queries = [query] + terms
                    print(f"  [LLM Expansion] 完整查询列表（原始 + 扩展）:\n")
                    for i, q in enumerate(all_queries):
                        print(f"    [{i}] {q}")
                    print()
                    print("请选择操作:")
                    print("  [Enter]  直接运行（使用当前扩展查询）")
                    print("  [e]      直接修改整个扩展查询（以 end 单独一行结束）")
                    print("  [llm]    重新定义 LLM 输入（以 end 单独一行结束）再次运行")
                    choice = input("> ").strip().lower()

                    if choice == "":
                        break

                    elif choice == "e":
                        print("请输入新的扩展词列表（每行一个，以 end 单独一行结束）:")
                        lines: list[str] = []
                        while True:
                            line = input()
                            if line.strip() == "end":
                                break
                            if line.strip():
                                lines.append(line.strip())
                        terms = lines
                        break

                    elif choice == "llm":
                        print("请输入新的 LLM 提示内容（以 end 单独一行结束）:")
                        lines = []
                        while True:
                            line = input()
                            if line.strip() == "end":
                                break
                            lines.append(line)
                        new_prompt = "\n".join(lines)
                        try:
                            response = call_gemini(
                                new_prompt,
                                model=GEMINI_FLASH,
                                confirm=False,
                                stream=True,
                                verbose=True,
                                show_cost=True,
                                reasoning=False,
                            )
                            if response and response.strip():
                                terms = _parse_terms(response)
                                print("  [LLM Expansion] 扩展要素:\n", "\n".join(terms))
                                print("\n\n\n")
                            else:
                                print("  [WARN] LLM 返回为空，保留原扩展结果")
                        except Exception as exc:
                            print(f"  [WARN] LLM 调用失败: {exc}，保留原扩展结果")
                        # 继续循环，展示最新结果后再次确认
                        continue

                    else:
                        print(f"  [WARN] 未知选项: {choice!r}，请重新输入")
                        continue

            return terms
    except Exception:
        pass

    return []


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  向量检索（混合 Dense + Sparse / BM25）                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def _enrich_with_chunk_sources(
    hits: list[dict],
    sources_db_path: Path | str,
) -> None:
    """从 chunk_sources.db 查询多来源 metadata，就地附加到每个结果的 all_sources 字段。

    同一个 content_hash 可能在 chunk_sources 表中有多行（同一篇论文归入多个
    分类目录），此函数将所有来源行聚合为 ``hit["all_sources"]: list[dict]``。
    """
    import sqlite3

    sources_db_path = Path(sources_db_path)
    if not sources_db_path.exists():
        return

    conn = sqlite3.connect(str(sources_db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        for hit in hits:
            content_hash = hit["id"]
            cursor.execute(
                "SELECT source_file, source_path, json_file, json_path,"
                "       book_title, book_source_path, category,"
                "       page_number, chunk_index, total_chunks"
                " FROM chunk_sources WHERE content_hash = ?",
                (content_hash,),
            )
            rows = cursor.fetchall()
            if rows:
                hit["all_sources"] = [dict(row) for row in rows]
    finally:
        conn.close()


def search_collection(
    query: str,
    model: Any,
    collection: Any,
    top_k: int = 8,
    *,
    category_filter: str | None = None,
    where: dict | None = None,
    adjacent_chunks: int = 0,
    enable_ensemble: bool = True,
    enable_reranker: bool = True,
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3",
    enable_llm_expansion: bool = True,
    confirm_llm_expansion: bool = False,
    sources_db_path: Path | str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> list[dict]:
    """
    对 ChromaDB 集合进行混合检索。

    检索策略（优先级由高到低）：

    1. **Dense + BGE-M3 Sparse 重打分**（推荐）
       当模型为 ``_BGEM3Wrapper``（FlagEmbedding 后端）且 metadata 中存在
       ``sparse_weights`` 时启用。
       · Dense 向量检索获取 fetch_k 个候选；
       · 对每个候选计算 BGE-M3 Sparse 点积分数；
       · 归一化后融合：``score = 0.6 × dense_sim + 0.4 × sparse_sim_norm``。

    2. **Dense + BM25 RRF 融合**（回退）
       非 BGE-M3 模型 + ``enable_ensemble=True`` 时启用。

    3. **Pure Dense**
       以上均不可用时，仅使用 Dense 向量相似度。

    4. **Cross-Encoder 重排序**（可选，默认启用）
       使用 Cross-Encoder（如 BAAI/bge-reranker-v2-m3）对候选进行精确重排序。
       初始检索获取 top_k * 10 个候选，经 Cross-Encoder 重排后返回 top_k 个。

    5. **LLM 查询扩展**（可选，默认启用）
       使用 Gemini Flash 模型生成查询的同义词和相关术语，
       将扩展后的查询用于 embedding 编码，提高召回率。

    Args:
        query:               检索查询文本（支持中 / 英 / 德 / 日）。
        model:               嵌入模型（_BGEM3Wrapper 或 SentenceTransformer）。
        collection:          ChromaDB 集合。
        top_k:               返回的最终结果数。
        category_filter:     按分类过滤（可选，匹配 metadata.category）。
        where:               额外的 ChromaDB where 子句（可选），与 category_filter
                             合并使用。例如 ``{"source_file": {"$in": [...]}}``。
        adjacent_chunks:     合并前后相邻的分块数（0 = 不合并）。
        enable_ensemble:     是否启用混合检索（BGE-M3 Sparse 或 BM25；默认 True）。
        enable_reranker:        是否启用 Cross-Encoder 重排序（默认 True）。
        reranker_model_name:   Cross-Encoder 模型名称。
        enable_llm_expansion:  是否启用 LLM 查询扩展（默认 True）。
        confirm_llm_expansion: 是否在 LLM 扩展后暂停，允许用户确认/编辑结果
                               或重新定义 LLM 输入后再次运行（默认 False）。
        sources_db_path:       chunk_sources.db 路径（可选）。若提供，返回结果中
                               每个 hit 会附加 ``all_sources: list[dict]``，
                               包含该 chunk 在所有来源中的 metadata（多分类、多路径）。

    Returns:
        结果列表，每个元素为 dict，包含：
          id, document, source_file, source_path, book_title, category,
          chunk_index, metadata, distance。
          启用 adjacent_chunks 时还包含 matched_document（原命中 chunk 文本）。
          提供 sources_db_path 时还包含 all_sources（该 chunk 的所有来源记录）。
    """
    def _report(msg: str) -> None:
        print(msg)
        if progress_callback:
            progress_callback(msg)

    # 构建 where 子句：合并 category_filter 和额外 where 条件
    _cat_clause = {"category": category_filter} if category_filter else None
    if _cat_clause and where:
        where_clause = {"$and": [_cat_clause, where]}
    else:
        where_clause = _cat_clause or where
    # 启用 reranker 时初始获取更多候选以供重排序
    if enable_reranker:
        fetch_k = max(top_k * 10, 50)
    else:
        fetch_k = top_k * (20 if adjacent_chunks > 0 else 4)

    is_bgem3 = getattr(model, "is_bgem3", False)

    # ══ 0. LLM 查询扩展 ══
    expansion_terms: list[str] = []
    if enable_llm_expansion:
        try:
            expansion_terms = _expand_query_with_llm(query, confirm=confirm_llm_expansion)
        except Exception:
            expansion_terms = []

    # 构建多查询列表：原始查询 + 扩展词（每个扩展词独立检索）
    all_queries = [query] + expansion_terms
    if expansion_terms:
        _report(f"  [Multi-Query] 共 {len(all_queries)} 条查询（1 原始 + {len(expansion_terms)} 扩展）")

    # ══ 1 & 2. 多查询编码 + Dense 检索 + RRF 合并 ══
    import numpy as np
    vec_hits_map: dict[str, dict] = {}
    # 每条子查询产生一个排序列表，用于 RRF 合并
    per_query_rank_lists: list[list[str]] = []
    # 原始查询的 sparse weights（用于后续 BGE-M3 sparse 重打分）
    q_sparse_weights: dict[str, float] = {}

    per_query_fetch_k = max(fetch_k // max(len(all_queries), 1), top_k * 2)

    for qi, sub_query in enumerate(all_queries):
        label = "原始查询" if qi == 0 else f"扩展词 {qi}/{len(expansion_terms)}"
        _report(f"  [Encode+Retrieve] {label}: {sub_query[:80]}{'…' if len(sub_query) > 80 else ''}")
        if is_bgem3 and enable_ensemble:
            hybrid_out = model.encode_hybrid([sub_query])
            q_dense = np.array(hybrid_out["dense_vecs"], dtype="float32")
            if qi == 0:
                raw_lw = hybrid_out.get("lexical_weights") or [{}]
                q_sparse_weights = {
                    str(k): float(v) for k, v in (raw_lw[0] if raw_lw else {}).items()
                }
        else:
            q_dense = model.encode(
                [sub_query], normalize_embeddings=True, convert_to_numpy=True
            )

        vec_results = collection.query(
            query_embeddings=q_dense.tolist(),
            n_results=per_query_fetch_k,
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        sub_rank_list: list[str] = []
        if vec_results["ids"]:
            for doc, meta, dist, rid in zip(
                vec_results["documents"][0],
                vec_results["metadatas"][0],
                vec_results["distances"][0],
                vec_results["ids"][0],
            ):
                meta = meta or {}
                sub_rank_list.append(rid)
                # 保留最佳 distance（最小值）
                if rid not in vec_hits_map or dist < vec_hits_map[rid]["distance"]:
                    vec_hits_map[rid] = {
                        "id": rid,
                        "document": doc,
                        "source_file": meta.get("source_file", ""),
                        "source_path": meta.get("source_path", ""),
                        "book_title": meta.get("book_title", ""),
                        "category": meta.get("category", ""),
                        "chunk_index": meta.get("chunk_index", 0),
                        "metadata": dict(meta),
                        "distance": dist,
                    }
        per_query_rank_lists.append(sub_rank_list)

    _report(f"  [Retrieve] Dense 检索完成，候选池共 {len(vec_hits_map)} 个唯一 chunk")

    # 多查询 RRF 合并：将各子查询的排序列表融合为统一分数
    if len(per_query_rank_lists) > 1:
        _report(f"  [RRF] 正在合并 {len(per_query_rank_lists)} 条子查询的排序列表…")
        k_rrf_mq = 60
        mq_scores: dict[str, float] = {}
        # 原始查询权重更高（占总权重的一半）
        expansion_weight = 1.0 / max(len(per_query_rank_lists) - 1, 1)
        for qi, rank_list in enumerate(per_query_rank_lists):
            w = 1.0 if qi == 0 else expansion_weight
            for rank, rid in enumerate(rank_list):
                mq_scores[rid] = mq_scores.get(rid, 0.0) + w / (k_rrf_mq + rank + 1)

        # 将 RRF 分数映射回 distance
        max_mq = max(mq_scores.values()) if mq_scores else 1.0
        for rid, score in mq_scores.items():
            if rid in vec_hits_map:
                vec_hits_map[rid]["distance"] = max(0.0, 1.0 - score / (max_mq * 1.01))

    # 构建 vec_rank_list（按 distance 排序，供后续 sparse/BM25 使用）
    vec_rank_list = sorted(vec_hits_map.keys(), key=lambda r: vec_hits_map[r]["distance"])

    # ══ 3a. BGE-M3 Sparse 重打分（优先） ══
    if is_bgem3 and enable_ensemble and q_sparse_weights and vec_hits_map:
        _report(f"  [Sparse] BGE-M3 Sparse 重打分（{len(vec_hits_map)} 个候选）…")
        candidates = list(vec_hits_map.values())

        # 计算所有候选的 sparse 点积分数
        sparse_scores = [
            _sparse_dot(q_sparse_weights, h["metadata"].get("sparse_weights", ""))
            for h in candidates
        ]

        # 归一化 sparse 分数到 [0, 1]（相对于本批候选的最高分）
        max_sparse = max(sparse_scores) if sparse_scores else 0.0
        if max_sparse < 1e-9:
            max_sparse = 1.0

        for hit, sp_score in zip(candidates, sparse_scores):
            dense_sim = 1.0 - hit["distance"]          # cosine dist → sim
            sp_norm = sp_score / max_sparse
            hit["hybrid_score"] = 0.6 * dense_sim + 0.4 * sp_norm

        raw_hits = sorted(candidates, key=lambda h: -h.get("hybrid_score", 0.0))
        # 将 hybrid_score 映射回 distance（越小越好）
        for hit in raw_hits:
            hit["distance"] = max(0.0, 1.0 - hit.get("hybrid_score", 0.0))

    # ══ 3b. BM25 + RRF 回退（仅非 BGE-M3 + enable_ensemble） ══
    elif enable_ensemble and not is_bgem3:
        _report("  [BM25] BM25 + RRF 融合回退…")
        bm25_rank_list: list[str] = []
        try:
            all_docs_data = collection.get(
                where=where_clause,
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            raise RuntimeError(
                "检索质量模式要求 BM25 融合候选读取成功；"
                f"当前 collection.get() 失败: {exc}"
            ) from exc

        if all_docs_data.get("ids"):
            bm25_hits = _perform_bm25_search(
                query,
                all_docs_data["documents"],
                all_docs_data["ids"],
                top_k=fetch_k,
            )
            id_to_meta = {
                rid: (doc, meta)
                for rid, doc, meta in zip(
                    all_docs_data["ids"],
                    all_docs_data["documents"],
                    all_docs_data["metadatas"],
                )
            }
            for rid, _score in bm25_hits:
                bm25_rank_list.append(rid)
                if rid not in vec_hits_map:
                    doc, meta = id_to_meta[rid]
                    meta = meta or {}
                    vec_hits_map[rid] = {
                        "id": rid,
                        "document": doc,
                        "source_file": meta.get("source_file", ""),
                        "source_path": meta.get("source_path", ""),
                        "book_title": meta.get("book_title", ""),
                        "category": meta.get("category", ""),
                        "chunk_index": meta.get("chunk_index", 0),
                        "metadata": dict(meta),
                        "distance": 1.0,
                    }

        # RRF 融合
        k_rrf = 60
        final_scores: dict[str, float] = {}

        def _apply_rrf(rank_list: list[str]) -> None:
            for rank, rid in enumerate(rank_list):
                final_scores[rid] = (
                    final_scores.get(rid, 0.0) + 1.0 / (k_rrf + rank + 1)
                )

        _apply_rrf(vec_rank_list)
        if bm25_rank_list:
            _apply_rrf(bm25_rank_list)

        sorted_rids = sorted(
            final_scores.keys(),
            key=lambda r: final_scores[r],
            reverse=True,
        )[:fetch_k]

        max_score = max(final_scores.values()) if final_scores else 1.0
        raw_hits = []
        for rid in sorted_rids:
            hit = vec_hits_map[rid]
            hit["distance"] = max(0.0, 1.0 - final_scores[rid] / (max_score * 1.01))
            raw_hits.append(hit)

    # ══ Pure Dense ══
    else:
        raw_hits = sorted(vec_hits_map.values(), key=lambda h: h["distance"])

    # ══ 4. 内容去重 ══
    _report(f"  [Dedup] 内容去重（{len(raw_hits)} 个候选）…")
    _doc_to_best: dict[str, dict] = {}
    for hit in raw_hits:
        doc = hit["document"]
        if doc not in _doc_to_best or len(hit["source_file"]) > len(
            _doc_to_best[doc]["source_file"]
        ):
            _doc_to_best[doc] = hit
    raw_hits = sorted(_doc_to_best.values(), key=lambda h: h["distance"])

    # ══ 4b. Cross-Encoder 重排序 ══
    if enable_reranker and len(raw_hits) > 1:
        _report(f"  [Rerank] Cross-Encoder 重排序（{len(raw_hits)} 个候选）…")
        try:
            raw_hits = _rerank_with_cross_encoder(
                query, raw_hits, reranker_model_name,
                progress_callback=_report,
            )
        except Exception as exc:
            _report(f"  [WARN] Cross-Encoder 重排序失败，回退到原始排序: {exc}")

    if adjacent_chunks <= 0:
        final_hits = raw_hits[:top_k]
        if sources_db_path:
            _enrich_with_chunk_sources(final_hits, sources_db_path)
        return final_hits

    # ══ 5. 合并相邻分块（批量查询） ══
    merged_hits: list[dict] = []
    processed_chunks: set[tuple[str, int]] = set()

    for hit in raw_hits:
        if len(merged_hits) >= top_k:
            break

        source_file = hit["source_file"]
        chunk_index = hit["chunk_index"]

        if (source_file, chunk_index) in processed_chunks:
            continue

        start_idx = max(0, chunk_index - adjacent_chunks)
        end_idx = chunk_index + adjacent_chunks

        surrounding = collection.get(
            where={
                "$and": [
                    {"source_file": source_file},
                    {"chunk_index": {"$gte": start_idx}},
                    {"chunk_index": {"$lte": end_idx}},
                ]
            },
            include=["documents", "metadatas"],
        )

        chunks_data = [
            (meta["chunk_index"], doc)
            for doc, meta in zip(
                surrounding["documents"], surrounding["metadatas"]
            )
        ]
        chunks_data.sort(key=lambda x: x[0])
        merged_doc = _merge_adjacent_chunks_dedup([doc for _, doc in chunks_data])

        for idx, _ in chunks_data:
            processed_chunks.add((source_file, idx))

        merged_hits.append({
            "id": hit["id"],
            "document": merged_doc,
            "matched_document": hit["document"],
            "source_file": source_file,
            "source_path": hit.get("source_path", ""),
            "book_title": hit.get("book_title", ""),
            "category": hit["category"],
            "chunk_index": (
                f"{chunks_data[0][0]}-{chunks_data[-1][0]}"
                if len(chunks_data) > 1
                else str(chunk_index)
            ),
            "metadata": dict(hit.get("metadata", {})),
            "distance": hit["distance"],
        })

    if sources_db_path:
        _enrich_with_chunk_sources(merged_hits, sources_db_path)
    return merged_hits


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  检索结果格式化                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def format_search_results(hits: list[dict]) -> str:
    """
    将检索结果格式化为人类可读的 Markdown 文本。

    Args:
        hits: search_collection() 返回的结果列表。

    Returns:
        格式化的文本字符串。
    """
    lines = []
    for i, h in enumerate(hits, 1):
        similarity = 1.0 - h["distance"]

        all_sources = h.get("all_sources")
        if all_sources and len(all_sources) > 1:
            # 多来源：聚合去重后的 category 和 source_path
            categories = list(dict.fromkeys(
                s["category"] for s in all_sources if s.get("category")
            ))
            cat_str = " | ".join(categories) if categories else h.get("category", "")

            book_titles = list(dict.fromkeys(
                s["book_title"] for s in all_sources if s.get("book_title")
            ))
            book = " / ".join(book_titles) if book_titles else (
                h.get("book_title") or h.get("source_file", "")
            )

            src_paths = list(dict.fromkeys(
                s.get("source_path") or s.get("book_source_path", "")
                for s in all_sources
            ))
            path_note = "".join(f"  \n  `{p}`" for p in src_paths if p)
        else:
            cat_str = h.get("category", "")
            book = h.get("book_title") or h.get("source_file", "")
            src_path = h.get("source_path", "")
            path_note = f"  \n  `{src_path}`" if src_path else ""

        # DeepSeekOCR: 输出时去除连续重复的标题行（索引阶段标题写入两次以增强嵌入，
        # 显示时只需要一次）
        doc_text = _dedup_consecutive_heading_lines(h['document'])
        lines.append(
            f"========================================\n\n"
            f"#### {i}. [{cat_str}] {book}  "
            f"(chunk {h['chunk_index']})  "
            f"相似度: {similarity*10:.2f}{path_note}\n\n"
            f"{doc_text}\n\n"
        )
    return "\n".join(lines)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  索引构建                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def index_chunks(
    chunks: list[str],
    metadata_list: list[dict],
    ids: list[str],
    model: Any,
    collection: Any,
    batch_size: int = 16,
) -> int:
    """
    将文本分块嵌入并写入 ChromaDB 集合。

    Args:
        chunks:        文本分块列表。
        metadata_list: 每个分块对应的元数据 dict 列表。
        ids:           每个分块的唯一 ID 列表。
        model:         嵌入模型（_BGEM3Wrapper 或 SentenceTransformer）。
        collection:    ChromaDB 集合。
        batch_size:    每批嵌入的分块数。

    Returns:
        成功写入的分块数。

    Note:
        此函数仅写入 dense 向量；如需同时写入 sparse_weights，
        请使用 index_single_file() 或自行调用 model.encode_hybrid()。
    """
    total = len(chunks)
    added = 0

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_chunks = chunks[start:end]
        batch_meta = metadata_list[start:end]
        batch_ids = ids[start:end]

        embeddings = model.encode(
            batch_chunks,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        collection.add(
            ids=batch_ids,
            embeddings=embeddings.tolist(),
            documents=batch_chunks,
            metadatas=batch_meta,
        )
        added += len(batch_chunks)

    return added


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  高层工作流：索引单个文件                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def index_single_file(
    filepath: Path | str,
    *,
    db_dir: Path | str | None = None,
    model_name: str = "BAAI/bge-m3",
    collection_name: str = "thesis_sources",
    chunk_size: int = 800,
    chunk_overlap: int = 200,
    max_chunk_len: int = 1200,
    batch_size: int = 64,
    offline: bool = True,
    allow_private: bool = False,
    document_metadata: dict[str, MetadataPrimitive] | None = None,
    chunk_metadata_factory: ChunkMetadataFactory | None = None,
) -> Path:
    """
    索引单个 txt 文件，生成独立的 ChromaDB 分片。

    流程：读取文件 → 段落感知分块 → 加载嵌入模型 → 创建分片数据库
          → 编码（BGE-M3 同时提取 sparse 权重）→ 写入。

    每个 chunk 的 metadata 自动写入：
      source_file / source_path / book_title / category /
      chunk_index / total_chunks / sparse_weights（BGE-M3 时）

    可通过 chunk_metadata_factory 附加溯源字段，例如：
      (chunk_index, chunk_text, filepath) → {"page_number": 5, "section": "第一章"}

    Args:
        filepath:        待索引的 txt 文件路径。
        db_dir:          db 目录路径（包含 chroma_db/ 的目录）。
                         分片将存储在其同级的 db_shards/ 目录下。
                         若为 None，则在 txt 文件所在目录创建 db_shards/。
        model_name:      嵌入模型名称。
        collection_name: ChromaDB 集合名称。
        chunk_size:      分块目标字符数（默认 600）。
        chunk_overlap:   相邻分块重叠目标字符数（默认 200，对齐到段落/句子边界）。
        max_chunk_len:   超过此长度的段落按句子边界切割（默认 1000）。
        batch_size:      每批嵌入的分块数。
        offline:         是否强制离线模式加载模型（批量索引时避免 API 限速）。
        allow_private:   是否允许索引路径中包含 private 或 secret 的文件。
        document_metadata:
                 应用于该文件全部分块的附加 metadata（用户自定义溯源字段）。
        chunk_metadata_factory:
                 为每个分块动态生成附加 metadata 的回调。
                 签名：(chunk_index, chunk_text, filepath) -> dict | None。

    Returns:
        分片数据库目录的路径。

    Raises:
        FileNotFoundError: 文件不存在时。
        SystemExit:        文件为空或模型加载失败时。
    """
    filepath = Path(filepath).resolve()
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")

    if not allow_private:
        lower_path = str(filepath).lower()
        if "private" in lower_path or "secret" in lower_path:
            print(f"[SKIP] 包含 private 或 secret 的文件被跳过: {filepath}", file=sys.stderr)
            sys.exit(0)

    filename = filepath.name
    category = extract_category(filename)
    book_title = _extract_book_title(filename)
    source_path_str = str(filepath)

    # ── 确定分片目录 ──
    if db_dir is not None:
        base = Path(db_dir).resolve().parent  # db/ 的上级
    else:
        base = filepath.parent
    shard_dir = base / "db_shards" / sanitize_dirname(filename)

    print(f"File       : {filename}")
    print(f"Book title : {book_title}")
    print(f"Category   : {category}")
    print(f"Shard dir  : {shard_dir}")

    t0 = time.time()

    # ── 读取文件 ──
    text = read_text_file(filepath)
    if not text:
        print("[SKIP] 文件为空或无法解码。", file=sys.stderr)
        sys.exit(0)

    # ── 内容去重检查（跨文件名/路径，仅比较文本内容）──
    shards_parent = base / "db_shards"
    is_dup, dup_source, content_hash = check_content_duplicate(
        text,
        db_dir=Path(db_dir).resolve() if db_dir is not None else None,
        shards_dir=shards_parent if shards_parent.exists() else None,
    )
    if is_dup:
        print(
            f"[SKIP] 文本内容已在数据库中（首次索引来源：{dup_source}），"
            "跳过重复索引。",
            file=sys.stderr,
        )
        sys.exit(0)

    # ── 分块 ──
    chunks = chunk_text(
        text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_chunk_len=max_chunk_len,
    )
    print(f"Chunks     : {len(chunks)}")

    if not chunks:
        print("[SKIP] 无有效分块。", file=sys.stderr)
        sys.exit(0)

    # ── 离线模式设置 ──
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    # ── 加载模型 ──
    device = get_device()
    print(f"Device     : {device.upper()}")
    print(f"Model      : {model_name}")

    try:
        model = load_embedding_model(model_name, device, local_files_only=True)
    except Exception:
        if offline:
            for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
                os.environ.pop(key, None)
        try:
            model = load_embedding_model(model_name, device, local_files_only=False)
        except (OSError, Exception) as e:
            print(
                f"\n[ERROR] 嵌入模型加载失败: {e}\n"
                "\n通常原因：模型尚未下载或缓存损坏。\n"
                "请先运行一次模型下载脚本。\n",
                file=sys.stderr,
            )
            sys.exit(1)

    is_bgem3 = getattr(model, "is_bgem3", False)
    backend_name = "BGE-M3/FlagEmbedding (dense + sparse)" if is_bgem3 else "SentenceTransformer (dense only)"
    print(f"Backend    : {backend_name}")

    # ── 创建分片数据库 ──
    _client, collection = load_chromadb_collection(
        shard_dir, collection_name, create_if_missing=True
    )

    # ── 编码并写入 ──
    for b_start in range(0, len(chunks), batch_size):
        batch = chunks[b_start : b_start + batch_size]

        if is_bgem3:
            # BGE-M3：同时获取 dense 向量 + sparse 词权重
            hybrid_out = model.encode_hybrid(
                batch, batch_size=min(batch_size, 12)
            )
            import numpy as np
            embeddings = np.array(hybrid_out["dense_vecs"], dtype="float32")
            raw_lw_list = hybrid_out.get("lexical_weights") or [{}] * len(batch)
            batch_sparse: list[str] = [
                _serialize_sparse_weights(lw, chunk_text_len=len(batch[i]))
                for i, lw in enumerate(raw_lw_list)
            ]
        else:
            embeddings = model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            batch_sparse = [""] * len(batch)

        ids = [f"{filename}::chunk::{b_start + j}" for j in range(len(batch))]
        metadatas = []
        for j, chunk in enumerate(batch):
            chunk_index = b_start + j
            extra_chunk_metadata = None
            if chunk_metadata_factory is not None:
                extra_chunk_metadata = chunk_metadata_factory(
                    chunk_index, chunk, filepath
                )
            meta = _build_chunk_metadata(
                filename=filename,
                source_path=source_path_str,
                book_title=book_title,
                category=category,
                chunk_index=chunk_index,
                total_chunks=len(chunks),
                document_metadata=document_metadata,
                chunk_metadata=extra_chunk_metadata,
            )
            # BGE-M3 sparse 权重写入 metadata
            if batch_sparse[j]:
                meta["sparse_weights"] = batch_sparse[j]
            metadatas.append(meta)

        collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=batch,
            metadatas=metadatas,
        )
        pct = min(100, int((b_start + len(batch)) / len(chunks) * 100))
        print(f"  embedded {b_start + len(batch)}/{len(chunks)}  ({pct}%)")

    # ── 写入内容哈希文件（供后续 check_content_duplicate / merge_shards 去重使用）──
    _shard_hash_file(shard_dir).write_text(content_hash, encoding="utf-8")

    elapsed = time.time() - t0
    print(f"\nDone.  {collection.count()} chunks  |  {elapsed:.1f}s  →  {shard_dir}")
    return shard_dir


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  高层工作流：索引文件夹                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def index_folder(
    folder: Path | str,
    *,
    db_dir: Path | str | None = None,
    file_pattern: str = "*.txt",
    model_name: str = "BAAI/bge-m3",
    collection_name: str = "thesis_sources",
    chunk_size: int = 600,
    chunk_overlap: int = 200,
    max_chunk_len: int = 1000,
    batch_size: int = 64,
    offline: bool = True,
    allow_private: bool = False,
    document_metadata_factory: FolderMetadataFactory | None = None,
    chunk_metadata_factory: FolderChunkMetadataFactory | None = None,
) -> list[Path]:
    """
    索引文件夹中的所有匹配文件，每个文件生成独立的 ChromaDB 分片。

    Args:
        folder:          待索引的文件夹路径。
        db_dir:          db 目录路径（分片存储在其同级 db_shards/）。
                         若为 None，则在文件夹所在目录创建 db_shards/。
        file_pattern:    文件匹配模式（默认 '*.txt'）。
        model_name:      嵌入模型名称。
        collection_name: ChromaDB 集合名称。
        chunk_size:      分块目标字符数（默认 600）。
        chunk_overlap:   相邻分块重叠目标字符数（默认 200）。
        max_chunk_len:   超过此长度的段落按句子边界切割（默认 1000）。
        batch_size:      每批嵌入的分块数。
        offline:         是否强制离线模式加载模型。
        allow_private:   是否允许索引路径中包含 private 或 secret 的文件。
        document_metadata_factory:
                 为每个文件生成通用 metadata 的回调。
                 签名：(filepath) -> dict | None。
        chunk_metadata_factory:
                 为每个文件的每个分块生成 metadata 的回调。
                 签名：(filepath, chunk_index, chunk_text) -> dict | None。

    Returns:
        成功生成的分片目录路径列表。
    """
    folder = Path(folder).resolve()
    if not folder.is_dir():
        print(f"[ERROR] 路径不是文件夹: {folder}")
        return []

    files = sorted(folder.glob(file_pattern))
    if not files:
        print(f"[WARN] 未找到匹配 {file_pattern} 的文件: {folder}")
        return []

    print(f"Found {len(files)} file(s) matching '{file_pattern}' in {folder}\n")

    shard_dirs: list[Path] = []
    for i, filepath in enumerate(files, 1):
        print(f"\n{'═' * 60}")
        print(f"[{i}/{len(files)}] {filepath.name}")
        print(f"{'═' * 60}")
        try:
            per_file_document_metadata = (
                document_metadata_factory(filepath)
                if document_metadata_factory is not None
                else None
            )

            per_file_chunk_metadata_factory: ChunkMetadataFactory | None = None
            if chunk_metadata_factory is not None:
                def per_file_chunk_metadata_factory(
                    chunk_index: int,
                    chunk_text_str: str,
                    resolved_filepath: Path,
                ) -> dict[str, MetadataPrimitive] | None:
                    return chunk_metadata_factory(
                        resolved_filepath, chunk_index, chunk_text_str
                    )

            shard = index_single_file(
                filepath,
                db_dir=db_dir,
                model_name=model_name,
                collection_name=collection_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                max_chunk_len=max_chunk_len,
                batch_size=batch_size,
                offline=offline,
                allow_private=allow_private,
                document_metadata=per_file_document_metadata,
                chunk_metadata_factory=per_file_chunk_metadata_factory,
            )
            shard_dirs.append(shard)
        except SystemExit:
            print(f"  [SKIP] {filepath.name}")
        except Exception as e:
            print(f"  [ERROR] {filepath.name}: {e}")

    print(f"\n{'═' * 60}")
    print(f"Folder indexing complete: {len(shard_dirs)}/{len(files)} files indexed.")
    return shard_dirs


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  模型下载                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def download_model(model_name: str = "BAAI/bge-m3") -> None:
    """
    下载嵌入模型到本地 HuggingFace 缓存。

    首次运行时需要网络连接，后续可离线使用。
    建议在并行索引前运行一次。

    Args:
        model_name: 要下载的模型名称。
    """
    # 清除离线模式环境变量以允许下载
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    os.environ.pop("HF_DATASETS_OFFLINE", None)

    device = get_device()
    print(f"Device : {device.upper()}")
    print(f"Model  : {model_name}")

    try:
        print("Checking if model is already downloaded...")
        model = load_embedding_model(model_name, device, local_files_only=True)
        print("Model already exists locally.")
    except Exception:
        print("Downloading model (this may take a few minutes on first run) …")
        model = load_embedding_model(model_name, device, local_files_only=False)

    # 测试嵌入
    test = model.encode(["Test: embedding model verification"], normalize_embeddings=True)
    print(f"\nModel loaded and tested OK.  Embedding dim = {test.shape[-1]}")
    print("\nYou can now run indexing jobs safely.")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  高层工作流：合并分片                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _try_open_shard(
    shard_path: Path, collection_name: str
) -> tuple[Any, int, str | None]:
    """
    尝试打开分片数据库。

    Returns:
        (collection, count, error_string)。打开失败时 collection=None。
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(shard_path))
        col = client.get_collection(collection_name)
        count = col.count()
        return col, count, None
    except Exception as e:
        return None, 0, str(e)


def merge_shards(
    db_dir: Path | str,
    *,
    shards_dir: Path | str | None = None,
    collection_name: str = "thesis_sources",
    source_dir: Path | str | None = None,
    page_size: int = 5000,
) -> dict[str, int | float]:
    """
    将 db_shards/ 中的所有分片合并至统一的 ChromaDB 数据库。

    增量合并：已存在的 chunk ID 会被跳过，可安全重复运行。

    Args:
        db_dir:          目标数据库目录（包含 chroma_db/ 的目录）。
        shards_dir:      分片目录。默认为 db_dir 同级的 db_shards/。
        collection_name: ChromaDB 集合名称。
        source_dir:      源文件目录（可选），用于检查哪些文件尚未索引。
        page_size:       每次从分片中读取的记录数。

    Returns:
        统计信息字典：
          before, added, duplicates, after, shards_total,
          shards_ok, shards_bad, elapsed
    """
    import chromadb

    db_dir = Path(db_dir).resolve()
    chroma_dir = db_dir / "chroma_db"

    if shards_dir is None:
        shards_dir = db_dir.parent / "db_shards"
    else:
        shards_dir = Path(shards_dir).resolve()

    if not shards_dir.exists():
        print(f"[ERROR] 分片目录不存在: {shards_dir}")
        sys.exit(1)

    # ── 查找所有分片 ──
    shard_dirs = sorted(
        d for d in shards_dir.iterdir()
        if d.is_dir() and (d / "chroma.sqlite3").exists()
    )

    if not shard_dirs:
        print(f"[ERROR] 未找到有效的 ChromaDB 分片: {shards_dir}")
        sys.exit(1)

    # ── 预览：展示分片列表与目标数据库 ──
    print(f"Found {len(shard_dirs)} shards:")
    for d in shard_dirs:
        display = d.name.rsplit("__", 1)[0]
        print(f"  · {display}")

    print()
    print(f"Target DB : {chroma_dir}")
    print()
    try:
        confirm = input("Proceed with merge? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        sys.exit(0)
    if confirm not in ("y", "yes"):
        print("Aborted.")
        sys.exit(0)
    print()

    # ── Phase 0：检查缺失的源文件 ──
    if source_dir is not None:
        source_dir = Path(source_dir)
        if source_dir.exists():
            source_files = sorted(source_dir.glob("*.txt"))
            expected = {sanitize_dirname(f.name): f.name for f in source_files}
            present = {d.name for d in shard_dirs}
            missing = [
                orig for shard_name, orig in expected.items()
                if shard_name not in present
            ]
            if missing:
                print(f"{'─' * 60}")
                print(f"NOT YET INDEXED — {len(missing)} source file(s) have no shard:")
                for fname in missing:
                    print(f"  {fname}")
                print(f"{'─' * 60}\n")
            else:
                print(f"All {len(source_files)} source files have a corresponding shard. ✓\n")

    # ── Phase 1：预扫描损坏分片 ──
    print("Pre-scanning shards for corruption …")
    good_shards: list[Path] = []
    bad_shards: list[tuple[str, str]] = []

    for shard_path in shard_dirs:
        _, _, err = _try_open_shard(shard_path, collection_name)
        if err:
            bad_shards.append((shard_path.name, err))
        else:
            good_shards.append(shard_path)

    print(f"\n  OK        : {len(good_shards)}")
    print(f"  Corrupted : {len(bad_shards)}")

    if bad_shards:
        print(f"\n{'─' * 60}")
        print("CORRUPTED SHARDS — 需要重新索引的文件:")
        for name, reason in bad_shards:
            original = name.rsplit("__", 1)[0]
            print(f"  File   : {original}")
            print(f"  Reason : {reason[:160]}")
            print()
        print(f"{'─' * 60}\n")

    if not good_shards:
        print("[ERROR] 没有可合并的分片。")
        sys.exit(1)

    # ── Phase 2：打开 / 创建目标数据库 ──
    chroma_dir.mkdir(parents=True, exist_ok=True)
    target_client = chromadb.PersistentClient(path=str(chroma_dir))
    target = target_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    before = target.count()
    print(f"Target DB currently has {before} chunks.")
    print(
        f"Merging {len(good_shards)} readable shards "
        f"(duplicates will be skipped) …\n"
    )

    # ── 加载内容哈希注册表（用于文档级去重） ──
    content_registry = load_content_registry(db_dir)
    skipped_content_dup: list[str] = []

    t0 = time.time()
    added = 0
    skipped_dup = 0
    read_errors: list[tuple[str, str]] = []

    for shard_path in good_shards:
        display_name = shard_path.name.rsplit("__", 1)[0]

        # ── 文档级内容去重：检查 content_hash.txt ──
        hash_file = _shard_hash_file(shard_path)
        if hash_file.exists():
            try:
                shard_hash = hash_file.read_text(encoding="utf-8").strip()
                if shard_hash in content_registry:
                    original = content_registry[shard_hash].get("source_file", "?")
                    print(
                        f"  [CONTENT-DUP] {display_name}  "
                        f"— 文本内容与已索引文件相同（{original}），跳过"
                    )
                    skipped_content_dup.append(display_name)
                    continue
            except Exception:
                pass  # hash 文件读取失败时忽略，继续正常流程

        col, count, err = _try_open_shard(shard_path, collection_name)
        if err:
            print(f"  [WARN] 分片打开失败: {shard_path.name}")
            read_errors.append((shard_path.name, err))
            continue

        if count == 0:
            continue

        offset = 0
        shard_added = 0
        shard_duped = 0
        while offset < count:
            try:
                batch = col.get(
                    limit=page_size,
                    offset=offset,
                    include=["documents", "metadatas", "embeddings"],
                )
            except Exception as e:
                print(
                    f"  [WARN] 读取错误 {shard_path.name} offset={offset}: {e}"
                )
                read_errors.append((shard_path.name, str(e)))
                break

            if not batch["ids"]:
                break

            # ── 去重：仅插入目标中尚不存在的 ID ──
            existing = target.get(ids=batch["ids"], include=[])
            existing_ids = set(existing["ids"])

            new_indices = [
                i for i, id_ in enumerate(batch["ids"])
                if id_ not in existing_ids
            ]
            batch_duped = len(batch["ids"]) - len(new_indices)
            shard_duped += batch_duped
            skipped_dup += batch_duped

            if new_indices:
                target.add(
                    ids=[batch["ids"][i] for i in new_indices],
                    embeddings=[batch["embeddings"][i] for i in new_indices],
                    documents=[batch["documents"][i] for i in new_indices],
                    metadatas=[batch["metadatas"][i] for i in new_indices],
                )
                shard_added += len(new_indices)
                added += len(new_indices)

            offset += page_size

        # ── 逐分片实时报告 ──
        if shard_duped > 0 and shard_added == 0:
            print(
                f"  [SKIP]    {display_name}  "
                f"— all {shard_duped} chunks already in DB"
            )
        elif shard_duped > 0:
            print(
                f"  [PARTIAL] {display_name}  "
                f"— added {shard_added}, skipped {shard_duped} duplicates"
            )
        else:
            print(f"  [OK]      {display_name}  — added {shard_added} chunks")

        # ── 更新内容哈希注册表（仅当该分片有新 chunk 被添加时） ──
        if shard_added > 0 and hash_file.exists():
            try:
                shard_hash = hash_file.read_text(encoding="utf-8").strip()
                if shard_hash and shard_hash not in content_registry:
                    content_registry[shard_hash] = {
                        "source_file": display_name,
                        "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
            except Exception:
                pass

    # ── 保存更新后的内容哈希注册表 ──
    save_content_registry(content_registry, db_dir)

    elapsed = time.time() - t0
    after = target.count()

    print(f"\n{'─' * 60}")
    print("Merge complete.")
    print(f"  Chunks before       : {before}")
    print(f"  Chunks added        : {added}")
    print(f"  ID duplicates       : {skipped_dup} (skipped)")
    print(f"  Content duplicates  : {len(skipped_content_dup)} shard(s) skipped")
    print(f"  Chunks after        : {after}")
    print(
        f"  Shards merged       : "
        f"{len(good_shards) - len(read_errors) - len(skipped_content_dup)}"
        f" / {len(good_shards)}"
    )
    print(f"  Time                : {elapsed:.1f} s")
    print(f"  Target DB           : {chroma_dir}")

    all_bad = bad_shards + read_errors
    if all_bad:
        print(f"\n  {len(all_bad)} shard(s) had errors — see details above.")
    if skipped_content_dup:
        print(f"  {len(skipped_content_dup)} shard(s) skipped (content already indexed).")

    return {
        "before": before,
        "added": added,
        "duplicates": skipped_dup,
        "content_duplicates": len(skipped_content_dup),
        "after": after,
        "shards_total": len(shard_dirs),
        "shards_ok": len(good_shards) - len(read_errors) - len(skipped_content_dup),
        "shards_bad": len(bad_shards) + len(read_errors),
        "elapsed": round(elapsed, 1),
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  高层工作流：一站式检索                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def search_from_indexed_db(
    query: str,
    db_dir: Path | str,
    *,
    model_name: str = "BAAI/bge-m3",
    collection_name: str = "thesis_sources",
    top_k: int = 8,
    category_filter: str | None = None,
    adjacent_chunks: int = 1,
    enable_ensemble: bool = True,
    enable_reranker: bool = True,
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3",
    enable_llm_expansion: bool = True,
    progress_callback: Callable[[str], None] | None = None,
) -> list[dict]:
    """
    一站式加载向量数据库并执行语义检索。

    自动处理模型加载、数据库连接、检索和可选的相邻分块合并。
    适用于只需一次检索的脚本或交互式调用场景。
    若需多次检索（如循环），建议自行加载模型和集合后直接调用
    search_collection() 以避免重复加载开销。

    Args:
        query:            检索查询文本（支持中 / 英 / 德 / 日）。
        db_dir:           数据库目录（包含 chroma_db/ 的目录）。
        model_name:       嵌入模型名称。
        collection_name:  ChromaDB 集合名称。
        top_k:            返回结果数。
        category_filter:  按分类过滤。
        adjacent_chunks:  合并前后相邻的分块数（0 = 不合并）。
        enable_ensemble:  是否启用混合检索（BGE-M3 Sparse 或 BM25 回退）。
        enable_reranker:  是否启用 Cross-Encoder 重排序。
        reranker_model_name: Cross-Encoder 模型名称。
        enable_llm_expansion: 是否启用 LLM 查询扩展。

    Returns:
        结果列表，每个元素为 dict，包含：
          id, document, source_file, source_path, book_title,
          category, chunk_index, distance。
          若存在 chunk_sources.db 还包含 all_sources（所有来源记录）。
    """
    db_dir = Path(db_dir).resolve()
    chroma_dir = db_dir / "chroma_db"

    # 自动探测 chunk_sources.db（merge 阶段生成的多来源 metadata 数据库）
    sources_db = db_dir / "chunk_sources.db"
    if not sources_db.exists():
        sources_db = chroma_dir / "chunk_sources.db"
    sources_db_path = sources_db if sources_db.exists() else None

    if progress_callback:
        progress_callback(f"[Load] 加载嵌入模型 {model_name}…")
    device = get_device()
    model = load_embedding_model(model_name, device)

    if progress_callback:
        progress_callback(f"[Load] 连接集合 {collection_name}…")
    _, collection = load_chromadb_collection(chroma_dir, collection_name)

    return search_collection(
        query,
        model,
        collection,
        top_k,
        category_filter=category_filter,
        adjacent_chunks=adjacent_chunks,
        enable_ensemble=enable_ensemble,
        enable_reranker=enable_reranker,
        reranker_model_name=reranker_model_name,
        enable_llm_expansion=enable_llm_expansion,
        sources_db_path=sources_db_path,
        progress_callback=progress_callback,
    )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  命令行界面                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _build_cli():
    """构建 argparse CLI 解析器。"""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m LLM_Lib.RAG",
        description=(
            "向量数据库工具：索引文件、合并分片、语义检索。\n"
            "不传参数直接运行可进入交互模式。\n\n"
            "目录结构约定：\n"
            "  <base>/db/           ← --db 指向此目录\n"
            "  <base>/db/chroma_db/ ← 统一向量数据库\n"
            "  <base>/db_shards/    ← 与 db/ 同级的分片目录"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", help="子命令")

    # ── index ──
    p_idx = sub.add_parser(
        "index",
        help="索引文件或文件夹，生成 ChromaDB 分片",
        description=(
            "读取文件或文件夹中的所有匹配文件 → 段落感知分块 → 嵌入 → 写入 db_shards/ 分片。\n"
            "传入文件则索引单个文件；传入文件夹则索引其中所有匹配的文件。\n"
            "可在多台机器 / 多进程下并行运行，之后用 merge 合并。\n"
            "当使用 BAAI/bge-m3 时，必须安装 FlagEmbedding 以启用 BGE-M3 dense + sparse 双路索引。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_idx.add_argument("path", type=str, help="待索引的文件或文件夹路径")
    p_idx.add_argument(
        "--db", type=str, default=None,
        help="db 目录路径（分片存储在其同级 db_shards/）；"
             "不指定则在文件所在目录创建 db_shards/",
    )
    p_idx.add_argument(
        "--pattern", type=str, default="*.txt",
        help="文件夹模式下的文件匹配模式 (default: *.txt)",
    )
    p_idx.add_argument(
        "--model", type=str, default="BAAI/bge-m3",
        help="嵌入模型名称 (default: BAAI/bge-m3)",
    )
    p_idx.add_argument(
        "--collection", type=str, default="thesis_sources",
        help="ChromaDB 集合名称 (default: thesis_sources)",
    )
    p_idx.add_argument(
        "--chunk-size", type=int, default=600,
        help="分块目标字符数 (default: 600)",
    )
    p_idx.add_argument(
        "--chunk-overlap", type=int, default=200,
        help="相邻分块重叠目标字符数，对齐到段落/句子边界 (default: 200)",
    )
    p_idx.add_argument(
        "--max-chunk-len", type=int, default=1000,
        help="超过此长度的段落按句子边界切割 (default: 1000)",
    )
    p_idx.add_argument(
        "--batch-size", type=int, default=64,
        help="每批嵌入的分块数 (default: 64)",
    )
    p_idx.add_argument(
        "--no-offline", action="store_true",
        help="不设置 HuggingFace 离线模式环境变量",
    )
    p_idx.add_argument(
        "--allow-private", action="store_true",
        help="允许索引路径中包含 private 或 secret 的文件",
    )

    # ── download ──
    p_dl = sub.add_parser(
        "download",
        help="下载嵌入模型到本地缓存",
        description=(
            "预下载嵌入模型到本地 HuggingFace 缓存。\n"
            "后续索引时可设置离线模式，避免 API 限速。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_dl.add_argument(
        "--model", type=str, default="BAAI/bge-m3",
        help="嵌入模型名称 (default: BAAI/bge-m3)",
    )

    # ── merge ──
    p_merge = sub.add_parser(
        "merge",
        help="将 db_shards/ 合并至统一 ChromaDB",
        description=(
            "遍历 db_shards/ 中的所有分片，增量合并到 db/chroma_db/。\n"
            "已存在的 chunk 会被跳过，可安全重复运行。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_merge.add_argument(
        "--db", type=str, required=True,
        help="目标数据库目录（chroma_db/ 所在目录）",
    )
    p_merge.add_argument(
        "--db-shards", type=str, default=None,
        help="分片目录（默认：与 --db 同级的 db_shards/）",
    )
    p_merge.add_argument(
        "--collection", type=str, default="thesis_sources",
        help="ChromaDB 集合名称 (default: thesis_sources)",
    )
    p_merge.add_argument(
        "--source-dir", type=str, default=None,
        help="源文件目录（可选），用于检查缺失的索引",
    )
    p_merge.add_argument(
        "--page-size", type=int, default=5000,
        help="每次从分片读取的记录数 (default: 5000)",
    )

    # ── query ──
    p_query = sub.add_parser(
        "query",
        help="对已索引数据库进行语义检索",
        description=(
            "从 db/chroma_db/ 加载向量数据库，执行混合检索。\n"
            "BGE-M3 + FlagEmbedding：dense + sparse 双路打分。\n"
            "若不提供 -q 参数则进入交互式检索模式。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_query.add_argument(
        "--db", type=str, required=True,
        help="数据库目录（chroma_db/ 所在目录）",
    )
    p_query.add_argument(
        "-q", "--query", type=str, default=None,
        help="检索查询文本（不提供则进入交互模式）",
    )
    p_query.add_argument(
        "-k", "--top-k", type=int, default=8,
        help="返回结果数 (default: 8)",
    )
    p_query.add_argument(
        "-c", "--category", type=str, default=None,
        help="按分类过滤 (e.g. '画家画作个案')",
    )
    p_query.add_argument(
        "--adjacent-chunks", type=int, default=1,
        help="合并前后相邻分块数 (default: 1, 0=不合并)",
    )
    p_query.add_argument(
        "--model", type=str, default="BAAI/bge-m3",
        help="嵌入模型名称 (default: BAAI/bge-m3)",
    )
    p_query.add_argument(
        "--collection", type=str, default="thesis_sources",
        help="ChromaDB 集合名称 (default: thesis_sources)",
    )
    p_query.add_argument(
        "--no-ensemble", action="store_true",
        help="禁用混合检索（BGE-M3 Sparse / BM25），仅使用 dense 向量",
    )
    p_query.add_argument(
        "--no-reranker", action="store_true",
        help="禁用 Cross-Encoder 重排序",
    )
    p_query.add_argument(
        "--no-llm-expansion", action="store_true",
        help="禁用 LLM 查询扩展",
    )

    return parser


# ── 交互式入口 ─────────────────────────────────────────────────────────────────


def _prompt(label: str, default: str = "", required: bool = False) -> str:
    """交互式提示输入，支持默认值和必填校验。返回空串表示用户跳过。"""
    if default:
        suffix = f" [{default}]"
    elif required:
        suffix = " (必填)"
    else:
        suffix = " (回车跳过)"
    while True:
        raw = input(f"  {label}{suffix}: ").strip()
        if raw:
            return raw
        if default:
            return default
        if not required:
            return ""
        print("    ⚠ 此项为必填，请输入。")


def _prompt_int(label: str, default: int) -> int:
    """交互式提示输入整数。"""
    while True:
        raw = _prompt(label, default=str(default))
        try:
            return int(raw)
        except (ValueError, TypeError):
            print("    ⚠ 请输入有效整数。")


def _prompt_bool(label: str, default: bool = False) -> bool:
    """交互式提示 y/n。"""
    hint = "Y/n" if default else "y/N"
    raw = input(f"  {label} [{hint}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes", "是")


def _interactive_index() -> None:
    """交互式索引流程。"""
    print("\n── 索引文件或文件夹 ──\n")
    path_str = _prompt("待索引路径", required=True)
    path = Path(path_str).resolve()
    if not path.exists():
        print(f"  ⚠ 路径不存在: {path}")
        return

    db_dir = _prompt("数据库目录 (db/)") or None
    pattern = _prompt("文件匹配模式", default="*.txt") if path.is_dir() else "*.txt"
    model = _prompt("嵌入模型", default="BAAI/bge-m3")

    # 高级参数
    advanced = _prompt_bool("自定义高级参数?")
    if advanced:
        collection = _prompt("集合名称", default="thesis_sources")
        chunk_size = _prompt_int("分块字符数", default=600)
        chunk_overlap = _prompt_int("分块重叠字符数", default=200)
        max_chunk_len = _prompt_int("最大段落长度", default=1000)
        batch_size = _prompt_int("嵌入批大小", default=64)
        offline = _prompt_bool("启用离线模式?", default=True)
        allow_private = _prompt_bool("允许 private/secret 路径?")
    else:
        collection = "thesis_sources"
        chunk_size, chunk_overlap, max_chunk_len = 600, 200, 1000
        batch_size = 64
        offline, allow_private = True, False

    print()
    if path.is_dir():
        index_folder(
            path,
            db_dir=db_dir,
            file_pattern=pattern,
            model_name=model,
            collection_name=collection,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_chunk_len=max_chunk_len,
            batch_size=batch_size,
            offline=offline,
            allow_private=allow_private,
        )
    else:
        index_single_file(
            path,
            db_dir=db_dir,
            model_name=model,
            collection_name=collection,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_chunk_len=max_chunk_len,
            batch_size=batch_size,
            offline=offline,
            allow_private=allow_private,
        )


def _interactive_download() -> None:
    """交互式下载模型。"""
    print("\n── 下载嵌入模型 ──\n")
    model = _prompt("嵌入模型", default="BAAI/bge-m3")
    print()
    download_model(model)


def _interactive_merge() -> None:
    """交互式合并分片。"""
    print("\n── 合并分片到统一数据库 ──\n")
    db_dir = _prompt("数据库目录 (db/)", required=True)
    shards_dir = _prompt("分片目录 (默认: db/ 同级 db_shards/)") or None
    collection = _prompt("集合名称", default="thesis_sources")
    source_dir = _prompt("源文件目录 (用于检查缺失索引)") or None
    page_size = _prompt_int("每批读取记录数", default=5000)
    print()
    merge_shards(
        db_dir,
        shards_dir=shards_dir,
        collection_name=collection,
        source_dir=source_dir,
        page_size=page_size,
    )


def _load_with_spinner(message: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
    """
    在后台线程中执行 fn(*args, **kwargs)，同时在主线程显示旋转进度指示器。

    设计原则：fn 执行期间不应有任何 print 输出（建议传入 verbose=False），
    以确保 \\r 覆写动画不被打断。完成后由调用方自行打印成功信息。
    """
    import threading

    result_holder: list[Any] = [None]
    exc_holder: list[BaseException | None] = [None]
    done = threading.Event()

    def _worker() -> None:
        try:
            result_holder[0] = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001
            exc_holder[0] = exc
        finally:
            done.set()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    i = 0
    while not done.wait(0.12):
        sys.stdout.write(f"\r  {message} {spinner_chars[i % len(spinner_chars)]}")
        sys.stdout.flush()
        i += 1

    # 清除进度行
    clear_len = len(message) + 6
    sys.stdout.write(f"\r{' ' * clear_len}\r")
    sys.stdout.flush()

    t.join()
    if exc_holder[0] is not None:
        raise exc_holder[0]
    return result_holder[0]


def _pick_from_list(label: str, options: list[str], allow_skip: bool = True) -> str:
    """展示编号列表让用户选择，返回选中项或空串（跳过）。"""
    if not options:
        return ""
    print(f"\n  可用{label}:")
    for i, opt in enumerate(options, 1):
        print(f"    {i}. {opt}")
    hint = "输入编号或名称, 回车跳过" if allow_skip else "输入编号或名称"
    raw = input(f"  {label} ({hint}): ").strip()
    if not raw:
        return "" if allow_skip else options[0]
    # 尝试按编号选择
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(options):
            return options[idx]
    except ValueError:
        pass
    # 原样返回用户输入
    return raw


def _interactive_query() -> None:
    """交互式语义检索。"""
    import chromadb

    print("\n── 语义检索 ──\n")
    db_file_str = _prompt("数据库文件路径 (chroma.sqlite3)", required=True)
    db_file = Path(db_file_str.strip('"')).resolve()

    if not db_file.exists():
        print(f"  ⚠ 文件不存在: {db_file}")
        return

    chroma_dir = db_file.parent

    # 读取可用集合列表
    client = chromadb.PersistentClient(path=str(chroma_dir))
    all_collections = [c.name for c in client.list_collections()]
    if not all_collections:
        print("  ⚠ 数据库中没有集合。")
        return

    if len(all_collections) == 1:
        collection_name = all_collections[0]
        print(f"  集合: {collection_name}")
    else:
        collection_name = _pick_from_list("集合", all_collections, allow_skip=False)
    coll = client.get_collection(collection_name)
    chunk_count = coll.count()
    print(f"  数据库: {chunk_count} chunks 已索引")

    # 读取可用分类
    categories: list[str] = []
    if chunk_count > 0:
        sample = coll.get(limit=chunk_count, include=["metadatas"])
        if sample and sample["metadatas"]:
            cats = {
                m.get("category", "")
                for m in sample["metadatas"]
                if isinstance(m, dict)
            }
            categories = sorted(c for c in cats if c)

    category: str | None = None
    if categories:
        picked = _pick_from_list("分类过滤", categories, allow_skip=True)
        category = picked or None

    model = _prompt("嵌入模型", default="BAAI/bge-m3")
    top_k = _prompt_int("返回结果数", default=8)
    adjacent = _prompt_int("合并前后相邻分块数", default=1)
    enable_ensemble = _prompt_bool("启用混合检索?", default=True)
    enable_reranker = _prompt_bool("启用 Cross-Encoder 重排序?", default=True)
    enable_llm_expansion = _prompt_bool("启用 LLM 查询扩展?", default=True)

    # ── 先收集第一个 query，再加载模型（避免用户在漫长加载后才能输入）──
    print()
    while True:
        try:
            q = input("Query (输入 exit 退出): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[退出检索]")
            return
        if q.lower() in ("exit", "quit", "q"):
            return
        if q:
            break

    # ── 加载嵌入模型（带进度指示器）──
    embed_model = _load_with_spinner(
        f"正在加载嵌入模型 {model} …",
        load_embedding_model, model, "cpu", verbose=False,
    )
    is_bgem3 = getattr(embed_model, "is_bgem3", False)
    device = getattr(getattr(embed_model, "_model", None), "device", "cpu") or "cpu"
    backend = "BGE-M3/FlagEmbedding" if is_bgem3 else "SentenceTransformer"
    if is_bgem3:
        print(f"  [BGE-M3] FlagEmbedding 后端加载成功 (device={device})")
    else:
        print(f"  [SentenceTransformer] 后端加载成功")
    print(f"后端: {backend}")

    # ── 预加载重排序模型（带进度指示器，仅首次查询前执行）──
    _reranker_model_name = "BAAI/bge-reranker-v2-m3"
    if enable_reranker:
        _load_with_spinner(
            f"正在加载重排序模型 {_reranker_model_name} …",
            _load_cross_encoder, _reranker_model_name, False,
        )
        # 判断实际使用的后端
        _rer = _cross_encoder_cache.get(_reranker_model_name)
        _rer_cls = type(_rer).__name__ if _rer is not None else "Unknown"
        if "FlagReranker" in _rer_cls:
            print(f"  [Reranker] FlagReranker 加载成功: {_reranker_model_name}")
        else:
            print(f"  [Reranker] CrossEncoder 加载成功: {_reranker_model_name}")
    print()

    # ── 执行第一次查询 ──
    hits = search_collection(
        q, embed_model, coll, top_k,
        category_filter=category,
        adjacent_chunks=adjacent,
        enable_ensemble=enable_ensemble,
        enable_reranker=enable_reranker,
        enable_llm_expansion=enable_llm_expansion,
    )
    print(format_search_results(hits))

    # ── 后续查询循环 ──
    while True:
        try:
            q = input("Query (输入 exit 退出): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[退出检索]")
            break
        if q.lower() in ("exit", "quit", "q"):
            break
        if not q:
            continue
        hits = search_collection(
            q, embed_model, coll, top_k,
            category_filter=category,
            adjacent_chunks=adjacent,
            enable_ensemble=enable_ensemble,
            enable_reranker=enable_reranker,
            enable_llm_expansion=enable_llm_expansion,
        )
        print(format_search_results(hits))


def _interactive_main() -> None:
    """交互式入口：通过引导式提示收集参数并执行操作。"""
    _COMMANDS = {
        "1": "index", "2": "download", "3": "merge", "4": "query",
        "index": "index", "download": "download",
        "merge": "merge", "query": "query",
    }

    while True:
        print()
        print("=" * 60)
        print("  向量数据库工具 — 交互模式")
        print("=" * 60)
        print()
        print("  1. index    — 索引文件或文件夹")
        print("  2. download — 下载嵌入模型")
        print("  3. merge    — 合并分片到统一数据库")
        print("  4. query    — 语义检索")
        print("  0. exit     — 退出")
        print()

        try:
            choice = input("请选择操作 [1/2/3/4/0]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n[退出]")
            break

        if choice in ("0", "exit", "quit", "q", ""):
            break

        cmd = _COMMANDS.get(choice)
        if not cmd:
            print("  ⚠ 无效选择，请重新输入。")
            continue

        try:
            if cmd == "index":
                _interactive_index()
            elif cmd == "download":
                _interactive_download()
            elif cmd == "merge":
                _interactive_merge()
            elif cmd == "query":
                _interactive_query()
        except (EOFError, KeyboardInterrupt):
            print("\n  [操作已取消]")


def main() -> None:
    """
    CLI 入口：根据子命令执行索引 / 合并 / 检索。
    """
    parser = _build_cli()
    args = parser.parse_args()

    if args.command is None:
        _interactive_main()
        return

    if args.command == "index":
        path = Path(args.path).resolve()
        if path.is_dir():
            index_folder(
                path,
                db_dir=args.db,
                file_pattern=args.pattern,
                model_name=args.model,
                collection_name=args.collection,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                max_chunk_len=args.max_chunk_len,
                batch_size=args.batch_size,
                offline=not args.no_offline,
                allow_private=args.allow_private,
            )
        else:
            index_single_file(
                path,
                db_dir=args.db,
                model_name=args.model,
                collection_name=args.collection,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                max_chunk_len=args.max_chunk_len,
                batch_size=args.batch_size,
                offline=not args.no_offline,
                allow_private=args.allow_private,
            )

    elif args.command == "download":
        download_model(args.model)

    elif args.command == "merge":
        merge_shards(
            args.db,
            shards_dir=args.db_shards,
            collection_name=args.collection,
            source_dir=args.source_dir,
            page_size=args.page_size,
        )

    elif args.command == "query":
        db_dir = Path(args.db).resolve()
        chroma_dir = db_dir / "chroma_db"

        enable_ensemble = not args.no_ensemble
        enable_reranker = not args.no_reranker
        enable_llm_expansion = not args.no_llm_expansion
        print(f"Loading model and database (ensemble={enable_ensemble}) …")
        device = get_device()
        model = load_embedding_model(args.model, device)
        _, collection = load_chromadb_collection(chroma_dir, args.collection)
        is_bgem3 = getattr(model, "is_bgem3", False)
        backend = "BGE-M3/FlagEmbedding" if is_bgem3 else "SentenceTransformer"
        print(f"Database: {collection.count()} chunks indexed.  Backend: {backend}\n")

        query_text = args.query
        if not query_text:
            # 交互式模式
            while True:
                try:
                    q = input("\nQuery (or 'exit'): ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n[退出]")
                    break

                if q.lower() in ("exit", "quit", "q"):
                    break
                if not q:
                    continue

                hits = search_collection(
                    q,
                    model,
                    collection,
                    args.top_k,
                    category_filter=args.category,
                    adjacent_chunks=args.adjacent_chunks,
                    enable_ensemble=enable_ensemble,
                    enable_reranker=enable_reranker,
                    enable_llm_expansion=enable_llm_expansion,
                )
                print(format_search_results(hits))
        else:
            hits = search_collection(
                query_text,
                model,
                collection,
                args.top_k,
                category_filter=args.category,
                adjacent_chunks=args.adjacent_chunks,
                enable_ensemble=enable_ensemble,
                enable_reranker=enable_reranker,
                enable_llm_expansion=enable_llm_expansion,
            )
            print(format_search_results(hits))


if __name__ == "__main__":
    main()
