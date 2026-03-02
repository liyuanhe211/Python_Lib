"""
Indexing.py — 向量数据库构建与检索工具库

══════════════════════════════════════════════════════════════════════════════
  提供跨项目可复用的向量数据库操作功能：

  基础工具函数:
    - get_device():              检测 CUDA / MPS / CPU
    - load_embedding_model():    加载 SentenceTransformer 嵌入模型
    - load_chromadb_collection(): 加载或创建 ChromaDB 集合
    - chunk_text():              文本分块（滑动窗口 + 重叠）
    - read_text_file():          多编码文本文件读取
    - extract_category():        从文件名方括号中提取分类标签
    - sanitize_dirname():        将文件名转为安全的目录名
    - search_collection():       向量语义检索（支持相邻分块合并）
    - format_search_results():   格式化检索结果为可读文本
    - index_chunks():            将文本分块嵌入并写入 ChromaDB

  高层工作流函数:
    - index_single_file():       索引单个 txt 文件 → db_shards/ 分片
    - index_folder():            索引文件夹中所有匹配文件 → db_shards/ 分片
    - merge_shards():            将 db_shards/ 合并至统一 ChromaDB
    - search_from_indexed_db():  一站式加载数据库并执行语义检索
    - download_model():          预下载嵌入模型到本地缓存

  命令行用法 (python -m LLM_Lib.Indexing):
    # 索引单个文件或整个文件夹 → 生成分片到 db_shards/
    python -m LLM_Lib.Indexing index <file_or_folder> [--db <db_dir>] [--pattern "*.txt"]

    # 预下载嵌入模型（并行索引前运行一次）
    python -m LLM_Lib.Indexing download [--model BAAI/bge-m3]

    # 将 db_shards/ 合并至 db/chroma_db/
    python -m LLM_Lib.Indexing merge --db <db_dir> [--db-shards <shards_dir>]

    # 交互式 / 命令行语义检索
    python -m LLM_Lib.Indexing query --db <db_dir> [-q <query>] [-k 8]

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

Python API 用法:
    from LLM_Lib.Indexing import (
        get_device, load_embedding_model, load_chromadb_collection,
        chunk_text, search_collection, format_search_results,
        index_single_file, index_folder, merge_shards,
        search_from_indexed_db, download_model,
    )
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  设备检测                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def get_device() -> str:
    """
    检测可用计算设备，返回 'cuda' / 'mps' / 'cpu'。

    优先级: CUDA GPU > Apple MPS > CPU
    """
    try:
        import torch
    except ImportError:
        return "cpu"

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

def load_embedding_model(
    model_name: str,
    device: str = "",
    *,
    local_files_only: bool = True,
) -> Any:
    """
    加载 SentenceTransformer 嵌入模型。

    优先从本地缓存加载；若 local_files_only=True 且本地不存在则尝试下载。

    Args:
        model_name:       模型名称（如 'BAAI/bge-m3'）。
        device:           计算设备（留空则自动检测）。
        local_files_only: 是否仅从本地加载（默认 True）。

    Returns:
        SentenceTransformer 模型实例。
    """
    from sentence_transformers import SentenceTransformer

    device = device or get_device()

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
# ║  文本分块                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    min_chunk_len: int = 50,
) -> list[str]:
    """
    将文本分割为重叠的分块。

    Args:
        text:         待分块文本。
        chunk_size:   每个分块的字符数。
        chunk_overlap: 相邻分块的重叠字符数。
        min_chunk_len: 丢弃短于此长度的分块。

    Returns:
        分块列表。
    """
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if len(chunk) >= min_chunk_len:
            chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks


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
# ║  向量检索 (Ensemble Retriever)                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _tokenize(text: str) -> list[str]:
    """
    分词辅助函数。优先使用 jieba，否则使用简单的字符/正则分词。
    """
    try:
        import jieba
        return list(jieba.cut_for_search(text))
    except ImportError:
        # 回退策略：中文字符拆开，英文保留单词
        # 这是一个简单的近似，好过没有分词
        tokens = []
        for word in re.split(r'([a-zA-Z0-9]+|\s+)', text):
            word = word.strip()
            if not word:
                continue
            if re.match(r'^[a-zA-Z0-9]+$', word):
                tokens.append(word.lower())
            else:
                tokens.extend(list(word))
        return tokens

def _perform_bm25_search(
    query: str,
    documents: list[str],
    doc_ids: list[str],
    top_k: int
) -> list[tuple[str, float]]:
    """
    执行 BM25 检索。如果未安装 rank_bm25，返回空列表。
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("[WARN] 未安装 'rank_bm25' 库，跳过关键词检索。建议 pip install rank_bm25 jieba")
        return []

    tokenized_corpus = [_tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    
    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    
    # 组合 (doc_id, score) 并排序
    results = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)
    return results[:top_k]

def search_collection(
    query: str,
    model: Any,
    collection: Any,
    top_k: int = 8,
    *,
    category_filter: str | None = None,
    adjacent_chunks: int = 0,
    enable_ensemble: bool = True,
) -> list[dict]:
    """
    对 ChromaDB 集合进行混合语义检索 (Ensemble Retriever)。
    结合了 Vector Search (语义) 和 BM25 (关键词)。

    Args:
        query:            检索查询文本。
        model:            SentenceTransformer 嵌入模型。
        collection:       ChromaDB 集合。
        top_k:            返回的最终结果数。
        category_filter:  按分类过滤（可选）。
        adjacent_chunks:  合并前后相邻的分块数（0 = 不合并）。
        enable_ensemble:  是否启用 BM25 混合检索（默认 True）。

    Returns:
        结果列表，每个元素为 dict。
    """
    # ── 1. 向量检索 (Vector Search) ──
    query_embedding = model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    )
    where_clause = {"category": category_filter} if category_filter else None

    # 为合并预留更多结果
    # 增加候选集大小，以便 RRF 和后续去重/合并
    fetch_k = top_k * (20 if adjacent_chunks > 0 else 2)
    
    vec_results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=fetch_k,
        where=where_clause,
        include=["documents", "metadatas", "distances"],
    )

    # 整理向量检索结果 -> {id: score} (score 用 1-distance 近似，或者直接用 rank)
    # 我们主要用 Rank，所以具体 score 只要单调即可
    vec_hits_map: dict[str, dict] = {}
    vec_rank_list: list[str] = []
    
    if vec_results["ids"]:
        for doc, meta, dist, rid in zip(
            vec_results["documents"][0],
            vec_results["metadatas"][0],
            vec_results["distances"][0],
            vec_results["ids"][0],
        ):
            vec_rank_list.append(rid)
            vec_hits_map[rid] = {
                "id": rid,
                "document": doc,
                "source_file": meta.get("source_file", ""),
                "category": meta.get("category", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "distance": dist, # 原始向量距离
            }

    # ── 2. 关键词检索 (BM25) ──
    bm25_rank_list: list[str] = []
    # 只有当启用了 ensemble 且存在 rank_bm25 时才执行
    if enable_ensemble:
        try:
            # 获取所有符合条件的文档用于 BM25
            all_docs_data = collection.get(
                where=where_clause,
                include=["documents", "metadatas"]
            )
        except Exception:
            all_docs_data = {"ids": []}

        if all_docs_data["ids"]:
            bm25_hits = _perform_bm25_search(
                query, 
                all_docs_data["documents"], 
                all_docs_data["ids"], 
                top_k=fetch_k
            )
            
            # 建立 id -> metadata 映射供后续填充
            id_to_meta = {
                rid: (doc, meta) 
                for rid, doc, meta in zip(all_docs_data["ids"], all_docs_data["documents"], all_docs_data["metadatas"])
            }

            for rid, score in bm25_hits:
                bm25_rank_list.append(rid)
                if rid not in vec_hits_map:
                    doc, meta = id_to_meta[rid]
                    vec_hits_map[rid] = {
                        "id": rid,
                        "document": doc,
                        "source_file": meta.get("source_file", ""),
                        "category": meta.get("category", ""),
                        "chunk_index": meta.get("chunk_index", 0),
                        "distance": 1.0, # 向量距离未知，设为 1.0 (不相关)
                    }

    # ── 3. RRF 融合 (Reciprocal Rank Fusion) ──
    # RRF Score = 1 / (k + rank_i)
    k_rrf = 60
    final_scores: dict[str, float] = {}

    def apply_rrf(rank_list):
        for rank, rid in enumerate(rank_list):
            final_scores[rid] = final_scores.get(rid, 0.0) + 1.0 / (k_rrf + rank + 1)

    apply_rrf(vec_rank_list)
    if bm25_rank_list:
        apply_rrf(bm25_rank_list)

    # 排序并取出足够的候选以供去重和合并
    sorted_rids = sorted(final_scores.keys(), key=lambda r: final_scores[r], reverse=True)[:fetch_k]
    
    # 构建 raw_hits。更新 distance 以反映 RRF 排序 (距离越小越好)
    raw_hits = []
    max_score = max(final_scores.values()) if final_scores else 1.0

    for rid in sorted_rids:
        hit = vec_hits_map[rid]
        # 反转 RRF score: score 越高 -> distance 越小
        hit["distance"] = 1.0 - (final_scores[rid] / (max_score * 1.01))
        raw_hits.append(hit)

    # ── 按文档内容去重：内容完全相同时保留文件名最长的碎片 ──
    _doc_to_best: dict[str, dict] = {}
    for hit in raw_hits:
        doc = hit["document"]
        if doc not in _doc_to_best or len(hit["source_file"]) > len(_doc_to_best[doc]["source_file"]):
            _doc_to_best[doc] = hit
    raw_hits = sorted(_doc_to_best.values(), key=lambda h: h["distance"])

    if adjacent_chunks <= 0:
        return raw_hits[:top_k]

    # ── 合并相邻分块 ──
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

        chunks_data = []
        for doc, meta in zip(
            surrounding["documents"], surrounding["metadatas"]
        ):
            chunks_data.append((meta["chunk_index"], doc))

        chunks_data.sort(key=lambda x: x[0])
        merged_doc = "\n...\n".join(doc for _, doc in chunks_data)

        for idx, _ in chunks_data:
            processed_chunks.add((source_file, idx))

        merged_hit = {
            "id": hit["id"],
            "document": merged_doc,
            "matched_document": hit["document"],
            "source_file": source_file,
            "category": hit["category"],
            "chunk_index": (
                f"{chunks_data[0][0]}-{chunks_data[-1][0]}"
                if len(chunks_data) > 1
                else str(chunk_index)
            ),
            "distance": hit["distance"],
        }
        merged_hits.append(merged_hit)

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
        similarity = 1 - h["distance"]
        lines.append(
            f"#### [{h['category']}] {h['source_file']}  "
            f"(chunk {h['chunk_index']})  "
            f"相似度: {similarity:.3f}\n\n"
            f"{h['document']}\n"
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
        model:         SentenceTransformer 嵌入模型。
        collection:    ChromaDB 集合。
        batch_size:    每批嵌入的分块数。

    Returns:
        成功写入的分块数。
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
    chunk_overlap: int = 100,
    batch_size: int = 64,
    offline: bool = True,
    allow_private: bool = False,
) -> Path:
    """
    索引单个 txt 文件，生成独立的 ChromaDB 分片。

    流程：读取文件 → 分块 → 加载嵌入模型 → 创建分片数据库 → 嵌入写入。

    Args:
        filepath:        待索引的 txt 文件路径。
        db_dir:          db 目录路径（包含 chroma_db/ 的目录）。
                         分片将存储在其同级的 db_shards/ 目录下。
                         若为 None，则在 txt 文件所在目录创建 db_shards/。
        model_name:      嵌入模型名称。
        collection_name: ChromaDB 集合名称。
        chunk_size:      分块字符数。
        chunk_overlap:   相邻分块重叠字符数。
        batch_size:      每批嵌入的分块数。
        offline:         是否强制离线模式加载模型（批量索引时避免 API 限速）。
        allow_private:   是否允许索引路径中包含 private 或 secret 的文件。

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

    # ── 确定分片目录 ──
    if db_dir is not None:
        base = Path(db_dir).resolve().parent  # db/ 的上级
    else:
        base = filepath.parent
    shard_dir = base / "db_shards" / sanitize_dirname(filename)

    print(f"File      : {filename}")
    print(f"Category  : {category}")
    print(f"Shard dir : {shard_dir}")

    t0 = time.time()

    # ── 读取与分块 ──
    text = read_text_file(filepath)
    if not text:
        print("[SKIP] 文件为空或无法解码。", file=sys.stderr)
        sys.exit(0)

    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"Chunks    : {len(chunks)}")

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
    print(f"Device    : {device.upper()}")
    print(f"Model     : {model_name}")

    try:
        model = load_embedding_model(model_name, device, local_files_only=True)
    except Exception:
        if offline:
            # 离线模式下允许临时下载
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

    # ── 创建分片数据库 ──
    client, collection = load_chromadb_collection(
        shard_dir, collection_name, create_if_missing=True
    )

    # ── 嵌入并写入 ──
    from sentence_transformers import SentenceTransformer  # for type hint only

    for b_start in range(0, len(chunks), batch_size):
        batch = chunks[b_start : b_start + batch_size]
        embeddings = model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        ids = [f"{filename}::chunk::{b_start + j}" for j in range(len(batch))]
        metadatas = [
            {
                "source_file": filename,
                "category": category,
                "chunk_index": b_start + j,
                "total_chunks": len(chunks),
            }
            for j in range(len(batch))
        ]

        collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=batch,
            metadatas=metadatas,
        )
        pct = min(100, int((b_start + len(batch)) / len(chunks) * 100))
        print(f"  embedded {b_start + len(batch)}/{len(chunks)}  ({pct}%)")

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
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    batch_size: int = 64,
    offline: bool = True,
    allow_private: bool = False,
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
        chunk_size:      分块字符数。
        chunk_overlap:   相邻分块重叠字符数。
        batch_size:      每批嵌入的分块数。
        offline:         是否强制离线模式加载模型。
        allow_private:   是否允许索引路径中包含 private 或 secret 的文件。

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
            shard = index_single_file(
                filepath,
                db_dir=db_dir,
                model_name=model_name,
                collection_name=collection_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                batch_size=batch_size,
                offline=offline,
                allow_private=allow_private,
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

    t0 = time.time()
    added = 0
    skipped_dup = 0
    read_errors: list[tuple[str, str]] = []

    for shard_path in good_shards:
        col, count, err = _try_open_shard(shard_path, collection_name)
        if err:
            print(f"  [WARN] 分片打开失败: {shard_path.name}")
            read_errors.append((shard_path.name, err))
            continue

        if count == 0:
            continue

        display_name = shard_path.name.rsplit("__", 1)[0]

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

    elapsed = time.time() - t0
    after = target.count()

    print(f"\n{'─' * 60}")
    print("Merge complete.")
    print(f"  Chunks before : {before}")
    print(f"  Chunks added  : {added}")
    print(f"  Duplicates    : {skipped_dup} (skipped)")
    print(f"  Chunks after  : {after}")
    print(
        f"  Shards merged : "
        f"{len(good_shards) - len(read_errors)} / {len(good_shards)}"
    )
    print(f"  Time          : {elapsed:.1f} s")
    print(f"  Target DB     : {chroma_dir}")

    all_bad = bad_shards + read_errors
    if all_bad:
        print(f"\n  {len(all_bad)} shard(s) were skipped — see details above.")

    return {
        "before": before,
        "added": added,
        "duplicates": skipped_dup,
        "after": after,
        "shards_total": len(shard_dirs),
        "shards_ok": len(good_shards) - len(read_errors),
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
) -> list[dict]:
    """
    一站式加载向量数据库并执行语义检索。

    自动处理模型加载、数据库连接、检索和可选的相邻分块合并。
    适用于只需一次检索的脚本或交互式调用场景。
    若需多次检索（如循环），建议自行加载模型和集合后直接调用
    search_collection() 以避免重复加载开销。

    Args:
        query:            检索查询文本。
        db_dir:           数据库目录（包含 chroma_db/ 的目录）。
        model_name:       嵌入模型名称。
        collection_name:  ChromaDB 集合名称。
        top_k:            返回结果数。
        category_filter:  按分类过滤。
        adjacent_chunks:  合并前后相邻的分块数（0 = 不合并）。
        enable_ensemble:  是否启用 BM25 混合检索。

    Returns:
        结果列表，每个元素为 dict，包含：
          id, document, source_file, category, chunk_index, distance
    """
    db_dir = Path(db_dir).resolve()
    chroma_dir = db_dir / "chroma_db"

    device = get_device()
    model = load_embedding_model(model_name, device)
    _, collection = load_chromadb_collection(chroma_dir, collection_name)

    return search_collection(
        query,
        model,
        collection,
        top_k,
        category_filter=category_filter,
        adjacent_chunks=adjacent_chunks,
        enable_ensemble=enable_ensemble,
    )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  命令行界面                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _build_cli():
    """构建 argparse CLI 解析器。"""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m LLM_Lib.Indexing",
        description=(
            "向量数据库工具：索引文件、合并分片、语义检索。\n\n"
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
            "读取文件或文件夹中的所有匹配文件 → 分块 → 嵌入 → 写入 db_shards/ 分片。\n"
            "传入文件则索引单个文件；传入文件夹则索引其中所有匹配的文件。\n"
            "可在多台机器 / 多进程下并行运行，之后用 merge 合并。"
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
        "--chunk-size", type=int, default=800,
        help="分块字符数 (default: 800)",
    )
    p_idx.add_argument(
        "--chunk-overlap", type=int, default=100,
        help="相邻分块重叠字符数 (default: 100)",
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
            "从 db/chroma_db/ 加载向量数据库，执行语义检索。\n"
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
        help="禁用 BM25 混合检索",
    )

    return parser


def main() -> None:
    """
    CLI 入口：根据子命令执行索引 / 合并 / 检索。
    """
    parser = _build_cli()
    args = parser.parse_args()

    # ... (前面的命令处理逻辑保持不变，但由于这里不是 replace，而是整个 function 重写，我需要把前面的部分也包含进来吗？
    # 不，我可以使用 smaller chunks replace.
    # 让我们只替换 p_query 部分和 search_collection 调用部分。) 


    if args.command is None:
        parser.print_help()
        sys.exit(0)

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
        print(f"Loading model and database (Ensemble: {enable_ensemble}) …")
        device = get_device()
        model = load_embedding_model(args.model, device)
        _, collection = load_chromadb_collection(chroma_dir, args.collection)
        print(f"Database: {collection.count()} chunks indexed.\n")

        query_text = args.query
        if not query_text:
            # 交互式模式
            while True:
                try:
                    q = input("\n🔍 Query (or 'exit'): ").strip()
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
            )
            print(format_search_results(hits))


if __name__ == "__main__":
    main()
