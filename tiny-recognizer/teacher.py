"""Qwen3-VL OpenAI-compatible offline annotation, resumable and auditable."""
import base64
import io
import json
import os
from pathlib import Path
import urllib.request

from PIL import Image, ImageOps
from common import (classes_from, digest_file, images_in, parse_qwen_boxes,
                    read_json, write_json, write_records)


def ask_qwen(path, names, base_url, model, timeout=180):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
    width, height = image.size
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=95)
    if names:
        instruction = ('Detect ALL instances of these categories, and use exactly these labels: '
                       + json.dumps(names, ensure_ascii=False))
    else:
        instruction = ('Detect all distinct visible physical objects, including uncommon objects. '
                       'Use concise English object names. Do not label the entire scene.')
    prompt = instruction + (
        '\nReturn ONLY a JSON array. Each entry is {"label":"object name",'
        '"bbox_format_1000":[x_min,y_min,x_max,y_max]}. '
        'Coordinates MUST be normalized to integers from 0 to 1000 relative to the supplied image. '
        'Include every instance. Return [] if no requested objects are present. No confidence scores.')
    payload = {
        'model': model, 'temperature': 0, 'max_tokens': 4096,
        'messages': [{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode()}},
            {'type': 'text', 'text': prompt}]}]}
    headers = {'Content-Type': 'application/json'}
    if os.environ.get('QWEN_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['QWEN_API_KEY']
    req = urllib.request.Request(base_url.rstrip('/') + '/chat/completions',
                                 data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        answer = json.load(response)
    choice = answer['choices'][0]
    if choice.get('finish_reason') != 'stop':
        raise ValueError('教师未完整输出（可能达到 token 上限），请拆分类别或图片。')
    content = choice['message']['content']
    return parse_qwen_boxes(content, width, height, names), content, width, height


def label(args):
    root = Path(args.images).resolve()
    names = classes_from(args.classes) if args.classes else None
    chunks = [names[i:i + args.chunk_size] for i in range(0, len(names), args.chunk_size)] if names else [None]
    out = Path(args.out)
    cache = out.parent / (out.stem + '_cache')
    records, errors = [], []
    for path in images_in(root):
        file_hash = digest_file(path)
        request_identity = {'image_sha256': file_hash, 'classes': names, 'model': args.model,
                            'endpoint': args.base_url, 'chunk_size': args.chunk_size, 'schema': 1}
        key = __import__('hashlib').sha256(json.dumps(request_identity, sort_keys=True).encode()).hexdigest()
        saved = cache / (key + '.json')
        try:
            if saved.exists():
                item = read_json(saved)
            else:
                boxes, raw = [], []
                for chunk in chunks:
                    result, text, width, height = ask_qwen(path, chunk, args.base_url, args.model)
                    boxes.extend(result)
                    raw.append(text)
                item = {**request_identity, 'width': width, 'height': height, 'boxes': boxes,
                        'raw_responses': raw, 'provenance': 'qwen3-vl-api'}
                write_json(saved, item)
            records.append({**item, 'file_name': path.relative_to(root).as_posix()})
            print(f'标注完成: {path.name}, {len(item["boxes"])} boxes', flush=True)
        except Exception as exc:
            # Never convert API/parse failures into negative training examples.
            errors.append({'file_name': path.relative_to(root).as_posix(), 'error': str(exc)})
            print(f'标注失败: {path.name}: {exc}', flush=True)
    write_records(out, records)
    write_json(out.with_suffix('.errors.json'), errors)
    discovered = sorted({b['label'] for r in records for b in r['boxes']})
    write_json(out.with_suffix('.discovered_classes.json'), discovered)
    if errors:
        raise RuntimeError(f'{len(errors)} 张图片失败；其余已保存。错误样本未作为背景。')
