"""File formats and coordinate validation shared by teacher and student."""
import hashlib
import json
import math
from pathlib import Path

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def read_records(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def write_records(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in records), encoding='utf-8')


def classes_from(path):
    names = read_json(path)
    if not isinstance(names, list) or not names or not all(isinstance(n, str) and n.strip() == n and n for n in names):
        raise ValueError('类别文件必须是非空、无空白名称的 JSON 字符串数组。')
    if len(set(names)) != len(names):
        raise ValueError('类别名称重复；请先统一同义词。')
    return names


def images_in(root):
    root = Path(root).resolve()
    paths = sorted(p for p in root.rglob('*') if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise ValueError(f'没有图片: {root}')
    return paths


def image_path(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'图片不存在或不在图片目录中: {name}')
    return path


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_xyxy(box, width, height):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError(f'非法边界框: {box}')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in box):
        raise ValueError(f'边界框需要有限数字: {box}')
    x1, y1, x2, y2 = map(float, box)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f'边界框越界或面积非正: {box}, image={width}x{height}')
    return [x1, y1, x2, y2]


def parse_qwen_boxes(content, width, height, allowed=None):
    """Qwen3-VL 0..1000 xyxy -> original-image pixel xyxy; never guess units."""
    content = content.strip()
    if content.startswith('```'):
        content = content.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    items = json.loads(content)
    if not isinstance(items, list):
        raise ValueError('教师输出必须是 JSON 数组。')
    boxes = []
    for item in items:
        label = item.get('label')
        if not isinstance(label, str) or not label.strip():
            raise ValueError('缺少 label')
        if allowed is not None and label not in allowed:
            raise ValueError(f'教师输出了类别表外的词: {label}')
        box = validate_xyxy(item.get('bbox_format_1000', item.get('bbox_format', item.get('box', item.get('bbox')))), 1000, 1000)
        box = [box[0] * width / 1000, box[1] * height / 1000,
               box[2] * width / 1000, box[3] * height / 1000]
        # Generative confidence is not a calibrated detector confidence.
        boxes.append({'label': label, 'xyxy': box, 'score': None})
    return boxes
