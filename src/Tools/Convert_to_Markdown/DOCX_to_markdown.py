import argparse
import os

from markitdown import MarkItDown
from Python_Lib.My_Lib_Stock import get_input_with_while_cycle


def _strip_wrapping_quotes(text):
    return text.strip().strip('"').strip("'")


def convert_docx_to_markdown(docx_path):
    docx_path = os.path.abspath(_strip_wrapping_quotes(docx_path))
    if not os.path.exists(docx_path):
        raise FileNotFoundError(f"Input file does not exist: {docx_path}")
    if not docx_path.lower().endswith(".docx"):
        raise ValueError(f"Input file must be a .docx file: {docx_path}")

    base_name = os.path.splitext(os.path.basename(docx_path))[0]
    output_dir = os.path.dirname(docx_path)
    markdown_output_path = os.path.join(output_dir, f"{base_name}.md")

    print(f"Converting '{docx_path}' to Markdown...")

    converter = MarkItDown()
    result = converter.convert(docx_path)
    markdown_content = result.text_content

    with open(markdown_output_path, "w", encoding="utf-8") as file:
        file.write(markdown_content)

    return {
        "input_path": docx_path,
        "markdown_output_path": markdown_output_path,
    }


def convert_multiple_docx_to_markdown(docx_paths):
    results = []
    for docx_path in docx_paths:
        results.append(convert_docx_to_markdown(docx_path))
    return results


def _build_argument_parser():
    parser = argparse.ArgumentParser(description="Convert DOCX files to same-name Markdown files.")
    parser.add_argument("docx_paths", nargs="*", help="Paths to .docx files")
    return parser


def _interactive_main():
    print("DOCX to Markdown")
    print("Input DOCX paths, one per line. Submit an empty line to start conversion.")
    docx_paths = get_input_with_while_cycle(
        strip_quote=True,
    )
    if not docx_paths:
        print("No DOCX files provided.")
        return []
    return convert_multiple_docx_to_markdown(docx_paths)


def main():
    parser = _build_argument_parser()
    args = parser.parse_args()

    try:
        if args.docx_paths:
            results = convert_multiple_docx_to_markdown(args.docx_paths)
        else:
            results = _interactive_main()
    except Exception as exc:
        print(f"Error during conversion: {exc}")
        raise SystemExit(1) from exc

    for result in results:
        print(f"Markdown output: {result['markdown_output_path']}")


if __name__ == "__main__":
    main()