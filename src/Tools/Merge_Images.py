import os
from PIL import Image
import math

# The image paths to merge
input_files = r'''"E:\My_Program\Voldy_Notes\文献_Private\0 其他两爬物种\蜥蜴亚目\巨蜥科 - 巨蜥属 - 尼罗河巨蜥\Lenz「1995 - Mertensiella」Zur Biologie Und Ökologie Des Nilwarans, Varanus Niloticus (Linnaeus 1766) in Gambia, Westafrika\89.png"
"E:\My_Program\Voldy_Notes\文献_Private\0 其他两爬物种\蜥蜴亚目\巨蜥科 - 巨蜥属 - 尼罗河巨蜥\Lenz「1995 - Mertensiella」Zur Biologie Und Ökologie Des Nilwarans, Varanus Niloticus (Linnaeus 1766) in Gambia, Westafrika\90.png"
"E:\My_Program\Voldy_Notes\文献_Private\0 其他两爬物种\蜥蜴亚目\巨蜥科 - 巨蜥属 - 尼罗河巨蜥\Lenz「1995 - Mertensiella」Zur Biologie Und Ökologie Des Nilwarans, Varanus Niloticus (Linnaeus 1766) in Gambia, Westafrika\91.png"
"E:\My_Program\Voldy_Notes\文献_Private\0 其他两爬物种\蜥蜴亚目\巨蜥科 - 巨蜥属 - 尼罗河巨蜥\Lenz「1995 - Mertensiella」Zur Biologie Und Ökologie Des Nilwarans, Varanus Niloticus (Linnaeus 1766) in Gambia, Westafrika\92.png"
"E:\My_Program\Voldy_Notes\文献_Private\0 其他两爬物种\蜥蜴亚目\巨蜥科 - 巨蜥属 - 尼罗河巨蜥\Lenz「1995 - Mertensiella」Zur Biologie Und Ökologie Des Nilwarans, Varanus Niloticus (Linnaeus 1766) in Gambia, Westafrika\93.png"
"E:\My_Program\Voldy_Notes\文献_Private\0 其他两爬物种\蜥蜴亚目\巨蜥科 - 巨蜥属 - 尼罗河巨蜥\Lenz「1995 - Mertensiella」Zur Biologie Und Ökologie Des Nilwarans, Varanus Niloticus (Linnaeus 1766) in Gambia, Westafrika\94.png"
"E:\My_Program\Voldy_Notes\文献_Private\0 其他两爬物种\蜥蜴亚目\巨蜥科 - 巨蜥属 - 尼罗河巨蜥\Lenz「1995 - Mertensiella」Zur Biologie Und Ökologie Des Nilwarans, Varanus Niloticus (Linnaeus 1766) in Gambia, Westafrika\95.png"
"E:\My_Program\Voldy_Notes\文献_Private\0 其他两爬物种\蜥蜴亚目\巨蜥科 - 巨蜥属 - 尼罗河巨蜥\Lenz「1995 - Mertensiella」Zur Biologie Und Ökologie Des Nilwarans, Varanus Niloticus (Linnaeus 1766) in Gambia, Westafrika\96.png"
"E:\My_Program\Voldy_Notes\文献_Private\0 其他两爬物种\蜥蜴亚目\巨蜥科 - 巨蜥属 - 尼罗河巨蜥\Lenz「1995 - Mertensiella」Zur Biologie Und Ökologie Des Nilwarans, Varanus Niloticus (Linnaeus 1766) in Gambia, Westafrika\98.png"
"E:\My_Program\Voldy_Notes\文献_Private\0 其他两爬物种\蜥蜴亚目\巨蜥科 - 巨蜥属 - 尼罗河巨蜥\Lenz「1995 - Mertensiella」Zur Biologie Und Ökologie Des Nilwarans, Varanus Niloticus (Linnaeus 1766) in Gambia, Westafrika\99.png"
"E:\My_Program\Voldy_Notes\文献_Private\0 其他两爬物种\蜥蜴亚目\巨蜥科 - 巨蜥属 - 尼罗河巨蜥\Lenz「1995 - Mertensiella」Zur Biologie Und Ökologie Des Nilwarans, Varanus Niloticus (Linnaeus 1766) in Gambia, Westafrika\101.png"
"E:\My_Program\Voldy_Notes\文献_Private\0 其他两爬物种\蜥蜴亚目\巨蜥科 - 巨蜥属 - 尼罗河巨蜥\Lenz「1995 - Mertensiella」Zur Biologie Und Ökologie Des Nilwarans, Varanus Niloticus (Linnaeus 1766) in Gambia, Westafrika\102.png"
"E:\My_Program\Voldy_Notes\文献_Private\0 其他两爬物种\蜥蜴亚目\巨蜥科 - 巨蜥属 - 尼罗河巨蜥\Lenz「1995 - Mertensiella」Zur Biologie Und Ökologie Des Nilwarans, Varanus Niloticus (Linnaeus 1766) in Gambia, Westafrika\103.png"
"E:\My_Program\Voldy_Notes\文献_Private\0 其他两爬物种\蜥蜴亚目\巨蜥科 - 巨蜥属 - 尼罗河巨蜥\Lenz「1995 - Mertensiella」Zur Biologie Und Ökologie Des Nilwarans, Varanus Niloticus (Linnaeus 1766) in Gambia, Westafrika\106.png"'''

# Merge mode:
#   "horizontal"        - single horizontal line
#   "vertical"          - single vertical column
#   "horizontal_lines"  - multiple horizontal lines (specify num_columns)
#   "vertical_columns"  - multiple vertical columns (specify num_rows)
merge_mode = "horizontal_lines"

# For "horizontal_lines" mode: how many images per row
# num_columns = 7
# For "vertical_columns" mode: how many images per column
# num_rows = 2

# Background color for padding areas
background_color = (255, 255, 255)


def parse_paths(paths_str):
    paths = [line.strip().strip('"').strip("'") for line in paths_str.split('\n') if line.strip()]
    return [os.path.normpath(p) for p in paths]


def merge_horizontal(images):
    """Merge images in one horizontal line, scaling all to the max height."""
    max_h = max(img.height for img in images)
    resized = []
    for img in images:
        scale = max_h / img.height
        new_w = round(img.width * scale)
        resized.append(img.resize((new_w, max_h), Image.LANCZOS))

    total_w = sum(img.width for img in resized)
    result = Image.new("RGB", (total_w, max_h), background_color)
    x = 0
    for img in resized:
        result.paste(img, (x, 0))
        x += img.width
    return result


def merge_vertical(images):
    """Merge images in one vertical column, scaling all to the max width."""
    max_w = max(img.width for img in images)
    resized = []
    for img in images:
        scale = max_w / img.width
        new_h = round(img.height * scale)
        resized.append(img.resize((max_w, new_h), Image.LANCZOS))

    total_h = sum(img.height for img in resized)
    result = Image.new("RGB", (max_w, total_h), background_color)
    y = 0
    for img in resized:
        result.paste(img, (0, y))
        y += img.height
    return result


def merge_horizontal_lines(images, cols):
    """Merge images into a grid with `cols` images per row.
    Each row is merged horizontally (scaled to that row's max height),
    then all rows are stacked vertically (scaled to the max row width)."""
    rows = []
    for i in range(0, len(images), cols):
        row_images = images[i:i + cols]
        rows.append(merge_horizontal(row_images))
    return merge_vertical(rows)


def merge_vertical_columns(images, rows):
    """Merge images into a grid with `rows` images per column.
    Each column is merged vertically (scaled to that column's max width),
    then all columns are placed horizontally (scaled to the max column height)."""
    cols = math.ceil(len(images) / rows)
    columns = []
    for c in range(cols):
        col_images = [images[c * rows + r] for r in range(rows) if c * rows + r < len(images)]
        columns.append(merge_vertical(col_images))
    return merge_horizontal(columns)


def main():
    paths = parse_paths(input_files)

    for p in paths:
        if not os.path.exists(p):
            print(f"File not found: {p}")
            return

    images = [Image.open(p).convert("RGB") for p in paths]
    print(f"Loaded {len(images)} images")
    for i, (p, img) in enumerate(zip(paths, images)):
        print(f"  [{i}] {img.width}x{img.height}  {os.path.basename(p)}")

    if merge_mode == "horizontal":
        result = merge_horizontal(images)
    elif merge_mode == "vertical":
        result = merge_vertical(images)
    elif merge_mode == "horizontal_lines":
        result = merge_horizontal_lines(images, num_columns)
    elif merge_mode == "vertical_columns":
        result = merge_vertical_columns(images, num_rows)
    else:
        print(f"Unknown merge_mode: {merge_mode}")
        return

    # Save next to the first image
    first_dir = os.path.dirname(paths[0])
    first_name = os.path.splitext(os.path.basename(paths[0]))[0]
    output_path = os.path.join(first_dir, f"{first_name}_merged.png")

    # Avoid overwriting
    counter = 1
    while os.path.exists(output_path):
        output_path = os.path.join(first_dir, f"{first_name}_merged_{counter}.png")
        counter += 1

    result.save(output_path)
    print(f"\nSaved: {output_path}")
    print(f"Size: {result.width}x{result.height}")


if __name__ == "__main__":
    main()
