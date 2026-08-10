"""
get_sigma_aldrich_pricing.py — Sigma-Aldrich 搜索页价格提取

══════════════════════════════════════════════════════════════════════════════
  根据化学品名称、别名或 CAS 号检索 Sigma-Aldrich 商品，并优先返回 HPLC 级别
  商品及其价格信息。

  设计原则：
    • 主路径使用标准库 HTTP 抓取 + 确定性 HTML 解析
    • 不依赖 LLM 猜测结构化字段，避免价格抽取不稳定
    • 仅当页面未直接返回价格、且本地已安装 Playwright 时，才尝试浏览器展开
    • 优先选择描述中包含 HPLC 的商品；若无 HPLC，再返回最佳备选
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import html
import importlib
import json
import re
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen


_SEARCH_BASE_URL = "https://www.sigmaaldrich.com"
_SEARCH_URL_TEMPLATE = (
    "https://www.sigmaaldrich.com/US/en/search/{slug}"
    "?focus=products&page=1&perpage=30&sort=relevance&term={term}&type=product"
)
_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/133.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip()


def _build_search_url(chemical_identifier: str) -> str:
    query = _normalize_query(chemical_identifier)
    slug = quote(re.sub(r"\s+", "-", query.lower()))
    term = quote(query)
    return _SEARCH_URL_TEMPLATE.format(slug=slug, term=term)


def _fetch_html(url: str, timeout: float = 20.0) -> str:
    request = Request(url, headers=_REQUEST_HEADERS)
    with urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def _try_expand_with_playwright(url: str, timeout_ms: int = 15000) -> str | None:
    """
    若本地安装了 Playwright，则尝试展开搜索结果中的价格区。

    这是可选兜底逻辑：未安装 Playwright 或浏览器展开失败时返回 None。
    """
    try:
        sync_api = importlib.import_module("playwright.sync_api")
    except ImportError:
        return None

    PlaywrightTimeoutError = getattr(sync_api, "TimeoutError")
    sync_playwright = getattr(sync_api, "sync_playwright")

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            page.wait_for_selector('[data-testid="srp-result-count"]', timeout=timeout_ms)

            buttons = page.locator('button[data-testid^="srp-pricing-"]')
            button_count = min(buttons.count(), 12)
            for index in range(button_count):
                button = buttons.nth(index)
                try:
                    label = button.inner_text(timeout=1500).strip().lower()
                except PlaywrightTimeoutError:
                    continue
                if "expand" not in label:
                    continue
                try:
                    button.click(timeout=2500)
                    page.wait_for_timeout(600)
                except PlaywrightTimeoutError:
                    continue

            content = page.content()
            browser.close()
            return content
    except Exception:
        return None


def _strip_tags(fragment: str) -> str:
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", fragment)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def _extract_first(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return _strip_tags(match.group(1))


def _extract_href(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    href = html.unescape(match.group(1)).strip()
    if href.startswith("http://") or href.startswith("https://"):
        return href
    if href.startswith("/"):
        return f"{_SEARCH_BASE_URL}{href}"
    return f"{_SEARCH_BASE_URL}/{href}"


def _extract_cells(row_html: str) -> list[str]:
    return [
        _strip_tags(cell)
        for cell in re.findall(r"(?is)<td\b[^>]*>(.*?)</td>", row_html)
    ]


def _extract_group_segments(search_html: str) -> list[str]:
    starts = [m.start() for m in re.finditer(r'data-testid="srp-substance-group"', search_html)]
    if not starts:
        return []

    segments: list[str] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(search_html)
        segments.append(search_html[start:end])
    return segments


def _extract_pricing_rows(region_html: str) -> list[dict[str, str]]:
    pricing_rows: list[dict[str, str]] = []
    for match in re.finditer(r'(?is)<tr\b[^>]*data-testid="P(?:&amp;|&)A-row-[^"]+"[^>]*>', region_html):
        row_start = match.start()
        row_end = region_html.find("</tr>", match.end())
        if row_end == -1:
            continue
        row_html = region_html[row_start:row_end + 5]
        cells = _extract_cells(row_html)
        if len(cells) < 4:
            continue
        pricing_rows.append({
            "sku": cells[0],
            "pack_size": cells[1],
            "availability": cells[2],
            "price": cells[3],
        })
    return pricing_rows


def _clean_copied_text_lines(search_text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in search_text.splitlines():
        line = html.unescape(raw_line)
        line = line.replace("\xa0", " ").replace("\u200b", " ").strip()
        line = re.sub(r"\s+", " ", line)
        if line:
            lines.append(line)
    return lines


def _looks_like_html(search_text: str) -> bool:
    return bool(re.search(r"<[^>]+>", search_text))


def _looks_like_pack_size(line: str) -> bool:
    return bool(re.match(r"^\d+(?:\.\d+)?(?: x \d+(?:\.\d+)?)?\s*(?:mL|L|g|kg|μg/mL)$", line, flags=re.IGNORECASE))


def _is_probable_product_number(line: str) -> bool:
    if " " in line or line.startswith("$") or len(line) < 3 or len(line) > 24:
        return False
    if not any(ch.isdigit() for ch in line):
        return False
    if line in {"CAS", "SKU", "Pricing", "Quantity"}:
        return False
    return bool(re.match(r"^[A-Z0-9][A-Z0-9.-]*[A-Z0-9]$", line, flags=re.IGNORECASE))


def _is_section_boundary(line: str) -> bool:
    return (
        line.startswith("All Photos")
        or line.startswith("Recommended Products")
        or line.startswith("Page ")
        or line.startswith("Support")
    )


def _parse_pricing_rows_from_text(lines: list[str], start_index: int) -> tuple[list[dict[str, str]], int]:
    pricing_rows: list[dict[str, str]] = []
    index = start_index
    header_lines = {"Pack Size", "Availability", "Price", "Quantity"}

    while index < len(lines):
        if lines[index] in header_lines:
            index += 1
            continue
        if _is_section_boundary(lines[index]) or lines[index].startswith("All Photos"):
            break
        if not _is_probable_product_number(lines[index]):
            break
        if index + 1 >= len(lines) or not _looks_like_pack_size(lines[index + 1]):
            break

        sku = lines[index]
        pack_size = lines[index + 1]
        index += 2

        availability_lines: list[str] = []
        price = ""
        while index < len(lines):
            line = lines[index]
            if line == "Details...":
                index += 1
                continue
            if line.startswith("$"):
                price = line
                index += 1
                break
            if _is_section_boundary(line):
                break
            if _is_probable_product_number(line) and index + 1 < len(lines) and _looks_like_pack_size(lines[index + 1]):
                break
            if line.isdigit():
                index += 1
                continue
            availability_lines.append(line)
            index += 1

        pricing_rows.append({
            "sku": sku,
            "pack_size": pack_size,
            "availability": " ".join(availability_lines).strip(),
            "price": price,
        })

        while index < len(lines) and lines[index].isdigit():
            index += 1

    return pricing_rows, index


def _parse_sigma_search_text(search_text: str) -> list[dict[str, Any]]:
    lines = _clean_copied_text_lines(search_text)
    products: list[dict[str, Any]] = []

    current_substance_name: str | None = None
    current_synonyms: str | None = None
    current_cas_number: str | None = None

    index = 0
    while index < len(lines):
        line = lines[index]

        if index + 1 < len(lines) and lines[index + 1].startswith("Synonym(s):"):
            current_substance_name = line
            current_synonyms = lines[index + 1].split(":", 1)[1].strip() or None
            current_cas_number = None
            index += 2

            while index < len(lines):
                meta_line = lines[index]
                if meta_line == "CAS No.:" and index + 1 < len(lines):
                    current_cas_number = lines[index + 1]
                    index += 2
                    continue
                if meta_line.startswith("Compare"):
                    break
                if _is_section_boundary(meta_line):
                    break
                index += 1
            continue

        if line.startswith("Compare"):
            index += 1
            while index < len(lines):
                if _is_section_boundary(lines[index]):
                    break

                product_number = lines[index]
                if not _is_probable_product_number(product_number):
                    index += 1
                    continue

                index += 1
                description_lines: list[str] = []
                while index < len(lines):
                    current_line = lines[index]
                    if current_line == "greener alternative":
                        index += 1
                        continue
                    if current_line == "SKU":
                        break
                    if _is_section_boundary(current_line):
                        break
                    if _is_probable_product_number(current_line):
                        break
                    description_lines.append(current_line)
                    index += 1

                pricing: list[dict[str, str]] = []
                if index < len(lines) and lines[index] == "SKU":
                    pricing, index = _parse_pricing_rows_from_text(lines, index + 1)

                description = " ".join(description_lines).strip()
                if description:
                    product = {
                        "substance_name": current_substance_name,
                        "synonyms": current_synonyms,
                        "cas_number": current_cas_number,
                        "product_number": product_number,
                        "product_url": None,
                        "brand_code": None,
                        "description": description,
                        "is_hplc": False,
                        "pricing": pricing,
                    }
                    product["is_hplc"] = _is_hplc_product(product)
                    products.append(product)
            continue

        index += 1

    products = _deduplicate_products(products)
    products.sort(key=_product_score, reverse=True)
    return products


def _is_hplc_product(product: dict[str, Any]) -> bool:
    haystack = " ".join([
        product.get("substance_name") or "",
        product.get("description") or "",
    ]).lower()
    return "hplc" in haystack


def _product_score(product: dict[str, Any]) -> tuple[int, int, int]:
    haystack = " ".join([
        product.get("substance_name") or "",
        product.get("description") or "",
    ]).lower()

    grade_score = 0
    if "hplc" in haystack:
        grade_score += 100
    elif "uhplc" in haystack:
        grade_score += 95
    elif "lc/ms" in haystack or "lc-ms" in haystack or "lcms" in haystack:
        grade_score += 60
    elif "gc" in haystack:
        grade_score += 30

    if "suitable for hplc" in haystack:
        grade_score += 20

    pricing_score = len(product.get("pricing") or [])
    availability_score = 1 if any(
        (row.get("availability") or "").strip() for row in product.get("pricing") or []
    ) else 0
    return (grade_score, pricing_score, availability_score)


def _parse_group_products(group_html: str) -> list[dict[str, Any]]:
    products: list[dict[str, Any]] = []
    substance_name = _extract_first(
        r'<h2\b[^>]*id="substance-name"[^>]*>(.*?)</h2>',
        group_html,
    )
    synonyms = _extract_first(
        r'Synonym\(s\):\s*</span>\s*<span[^>]*>(.*?)</span>',
        group_html,
    )
    cas_number = _extract_first(
        r'CAS No\.:\s*</div></dt>\s*<dd>\s*(.*?)\s*</dd>',
        group_html,
    )

    row_starts = list(re.finditer(r'(?is)<tr\b[^>]*data-testid="product-[^"]+"[^>]*>', group_html))
    for index, row_match in enumerate(row_starts):
        row_start = row_match.start()
        row_end = group_html.find("</tr>", row_match.end())
        if row_end == -1:
            continue

        product_row_html = group_html[row_start:row_end + 5]
        next_row_start = row_starts[index + 1].start() if index + 1 < len(row_starts) else len(group_html)
        row_region_html = group_html[row_end + 5:next_row_start]

        product_url = _extract_href(
            r'data-testid="NAME-pdp-link-[^"]+"[^>]*href="([^"]+)"',
            product_row_html,
        )
        product_number = _extract_first(
            r'data-testid="NAME-pdp-link-[^"]+"[^>]*>(.*?)</a>',
            product_row_html,
        )
        description = _extract_first(
            r'class="[^"]*productDescLink[^"]*"[^>]*>(.*?)</span>',
            product_row_html,
        )
        pricing = _extract_pricing_rows(row_region_html)

        if not product_number or not description:
            continue

        brand_code = None
        if product_url:
            brand_match = re.search(r'/product/([^/]+)/', product_url, flags=re.IGNORECASE)
            if brand_match:
                brand_code = brand_match.group(1).upper()

        product = {
            "substance_name": substance_name,
            "synonyms": synonyms,
            "cas_number": cas_number,
            "product_number": product_number,
            "product_url": product_url,
            "brand_code": brand_code,
            "description": description,
            "is_hplc": False,
            "pricing": pricing,
        }
        product["is_hplc"] = _is_hplc_product(product)
        products.append(product)

    return products


def _deduplicate_products(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduplicated: list[dict[str, Any]] = []
    seen_numbers: set[str] = set()
    for product in products:
        product_number = product.get("product_number")
        if not product_number or product_number in seen_numbers:
            continue
        seen_numbers.add(product_number)
        deduplicated.append(product)
    return deduplicated


def _parse_sigma_search_html(search_html: str) -> list[dict[str, Any]]:
    products: list[dict[str, Any]] = []
    for group_html in _extract_group_segments(search_html):
        products.extend(_parse_group_products(group_html))

    products = _deduplicate_products(products)
    products.sort(key=_product_score, reverse=True)
    return products


def _parse_sigma_search_content(search_content: str) -> list[dict[str, Any]]:
    if _looks_like_html(search_content):
        return _parse_sigma_search_html(search_content)
    return _parse_sigma_search_text(search_content)


def _format_product_for_llm(product: dict[str, Any]) -> str:
    lines = [
        f"Product No.: {product.get('product_number') or 'N/A'}",
        f"Description: {product.get('description') or 'N/A'}",
        f"Substance: {product.get('substance_name') or 'N/A'}",
        f"CAS No.: {product.get('cas_number') or 'N/A'}",
    ]

    for row in product.get("pricing") or []:
        lines.append(f"{row.get('sku') or 'N/A'}\t{row.get('pack_size') or 'N/A'}")
        if row.get("availability"):
            lines.append(row["availability"])
        if row.get("price"):
            lines.append(row["price"])

    return "\n".join(lines)


def get_sigma_aldrich_pricing(
    chemical_identifier: str,
    *,
    prefer_hplc: bool = True,
    try_browser_fallback: bool = True,
    html_text: str | None = None,
) -> dict[str, Any] | None:
    """
    检索 Sigma-Aldrich 搜索页，返回优先商品及价格信息。

    Parameters
    ----------
    chemical_identifier:
        化学品名称、别名或 CAS 号。
    prefer_hplc:
        为 True 时优先选择 HPLC 级别商品；若不存在则回退到最佳可用商品。
    try_browser_fallback:
        当静态 HTML 未能解析到任何价格时，若本地已安装 Playwright，尝试浏览器展开。
    html_text:
        仅供测试或离线解析使用；传入时跳过网络请求。
    """
    query = _normalize_query(chemical_identifier)
    if not query:
        raise ValueError("chemical_identifier 不能为空")

    search_url = _build_search_url(query)
    used_browser_fallback = False

    if html_text is None:
        html_text = _fetch_html(search_url)

    products = _parse_sigma_search_content(html_text)

    if try_browser_fallback and not any(product.get("pricing") for product in products):
        browser_html = _try_expand_with_playwright(search_url)
        if browser_html:
            browser_products = _parse_sigma_search_content(browser_html)
            if any(product.get("pricing") for product in browser_products):
                html_text = browser_html
                products = browser_products
                used_browser_fallback = True

    if not products:
        return None

    preferred_products = [product for product in products if product.get("is_hplc")] if prefer_hplc else products
    preferred_product = preferred_products[0] if preferred_products else products[0]

    return {
        "query": query,
        "search_url": search_url,
        "used_browser_fallback": used_browser_fallback,
        "hplc_found": any(product.get("is_hplc") for product in products),
        "preferred_product": preferred_product,
        "preferred_product_text": _format_product_for_llm(preferred_product),
        "products": products,
    }


def _print_product_summary(product: dict[str, Any]) -> None:
    print(f"Product No.: {product.get('product_number') or 'N/A'}")
    print(f"Description: {product.get('description') or 'N/A'}")
    print(f"Substance: {product.get('substance_name') or 'N/A'}")
    print(f"CAS No.: {product.get('cas_number') or 'N/A'}")
    print(f"URL: {product.get('product_url') or 'N/A'}")
    print(f"HPLC match: {'Yes' if product.get('is_hplc') else 'No'}")

    pricing_rows = product.get("pricing") or []
    if not pricing_rows:
        print("Pricing: not found")
        return

    print("Pricing:")
    for row in pricing_rows:
        sku = row.get("sku") or "N/A"
        pack_size = row.get("pack_size") or "N/A"
        availability = row.get("availability") or "N/A"
        price = row.get("price") or "N/A"
        print(f"  - {sku} | {pack_size} | {price} | {availability}")


def _interactive_main() -> None:
    print("Sigma-Aldrich pricing interactive test")
    print("Press Enter on an empty line to exit.")
    print("If you want to test copied page text, paste it after choosing pasted-text mode and finish with a single END line.")


    def _read_pasted_text() -> str:
        print("Paste copied page text below. End with a single END line.")
        lines: list[str] = []
        while True:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        return "\n".join(lines)

    while True:
        query = input("\nChemical identifier: ").strip()
        if not query:
            print("Exit.")
            break

        pasted_mode = input("Use pasted page text? [y/N]: ").strip().lower() in {"y", "yes"}
        search_content = _read_pasted_text() if pasted_mode else None

        try:
            result = get_sigma_aldrich_pricing(
                query,
                html_text=search_content,
                try_browser_fallback=not pasted_mode,
            )
        except Exception as exc:
            print(f"Error: {exc}")
            continue

        if result is None:
            print("No product found.")
            continue

        print(f"Search URL: {result['search_url']}")
        print(f"HPLC found: {'Yes' if result['hplc_found'] else 'No'}")
        print(f"Browser fallback used: {'Yes' if result['used_browser_fallback'] else 'No'}")
        print(f"Total parsed products: {len(result['products'])}")
        print()
        _print_product_summary(result["preferred_product"])

        print("\nPreferred product text for LLM:")
        print(result["preferred_product_text"])

        show_json = input("\nPrint full JSON result? [y/N]: ").strip().lower()
        if show_json in {"y", "yes"}:
            print(json.dumps(result, ensure_ascii=False, indent=2))


__all__ = ["get_sigma_aldrich_pricing"]


if __name__ == "__main__":
    _interactive_main()