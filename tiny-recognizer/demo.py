#!/usr/bin/env python3
"""Open-vocabulary, non-YOLO detection demo. See README.md for the research limits."""
import argparse
import json
from pathlib import Path
import time

from PIL import Image, ImageDraw, ImageOps
import torch

from common import (classes_from, digest_file, image_path, images_in,
                    read_json, write_json, write_records)
from models import (ROOT, OpenVocabularyDETR, RegionBaseline, TinyCLIP,
                    count_parameters, filter_boxes)


def names_from_args(args):
    names = classes_from(args.classes) if args.classes else args.texts
    if not names or len(set(names)) != len(names) or not all(n.strip() for n in names):
        raise ValueError('类别名称必须非空且唯一。')
    return names


def encode(args):
    names = names_from_args(args)
    clip = TinyCLIP(args.clip_weights, args.device).text_only()
    vocab = {'names': names, 'features': clip.texts(names).cpu(),
             'text_sha256': digest_file(args.clip_weights), 'encoder': 'TinyCLIP-ViT-8M-16-Text-3M-YFCC15M',
             'text_parameters_including_token_embeddings': clip.text_parameters}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(vocab, args.out)
    print(f'已生成 {len(names)} 个文本原型: {args.out}')


def detect(args):
    paths = images_in(args.images) if args.images else [Path(args.image).resolve()]
    if args.vocab:
        vocab = torch.load(args.vocab, map_location='cpu', weights_only=True)
        names, features = vocab['names'], vocab['features'].to(args.device)
    else:
        names = names_from_args(args)
        clip = TinyCLIP(args.clip_weights, args.device).text_only()
        features = clip.texts(names)
        vocab = {'names': names, 'text_sha256': digest_file(args.clip_weights),
                 'text_parameters_including_token_embeddings': clip.text_parameters}
        del clip
    if args.backend == 'baseline':
        model = RegionBaseline(args.device, args.size, args.dfine_weights, args.clip_weights)
        if vocab['text_sha256'] != digest_file(args.clip_weights):
            raise ValueError('文本原型和区域编码器的权重不匹配。')
        parameter_report = {'detector': count_parameters(model.detector),
                            'region_image_encoder': model.clip.image_parameters,
                            'text_encoder_including_token_embeddings': model.clip.text_parameters,
                            'total': count_parameters(model.detector) + count_parameters(model.clip.model),
                            'image_path': count_parameters(model.detector) + model.clip.image_parameters}
        predict = lambda image: model.predict(image, names, features, args.proposals, not args.no_dense)
    else:
        if not args.checkpoint:
            raise ValueError('--backend student 需要 --checkpoint。')
        model, metadata = OpenVocabularyDETR.load(args.checkpoint, args.device)
        if metadata['text_sha256'] != vocab['text_sha256']:
            raise ValueError('当前文本编码器与学生训练时使用的编码器不匹配。')
        model.set_vocabulary(names, features)
        parameter_report = {'image_detector': count_parameters(model),
                            'text_encoder_including_token_embeddings': vocab['text_parameters_including_token_embeddings'],
                            'total_with_text_encoder': count_parameters(model) + vocab['text_parameters_including_token_embeddings'],
                            'note': 'cached-vocabulary image inference loads only the detector; new words require text encoding'}
        predict = model.predict
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.images).resolve() if args.images else paths[0].parent
    records, timings = [], []
    for index, path in enumerate(paths):
        with Image.open(path) as original:
            image = ImageOps.exif_transpose(original).convert('RGB')
        if index == 0:
            for _ in range(args.warmup):
                predict(image)
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        start = time.perf_counter()
        boxes = predict(image)
        boxes = filter_boxes(boxes, args.threshold, args.nms_iou, args.max_detections)
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        timings.append(elapsed)
        file_name = path.relative_to(root).as_posix()
        records.append({'file_name': file_name, 'width': image.width, 'height': image.height,
                        'boxes': boxes, 'image_sha256': digest_file(path), 'vocabulary': names,
                        'backend': args.backend, 'inference_ms': elapsed})
        draw = ImageDraw.Draw(image)
        for box in boxes:
            draw.rectangle(box['xyxy'], outline='#00e5b0', width=3)
            draw.text(tuple(box['xyxy'][:2]), f'{box["label"]} {box["score"]:.3f}', fill='red', stroke_width=1)
        image.save(out / f'{index:05d}_{path.stem}.jpg')
        print(f'{file_name}: {len(boxes)} boxes, {elapsed:.1f} ms', flush=True)
    write_records(out / 'predictions.jsonl', records)
    write_json(out / 'report.json', {'backend': args.backend, 'parameters': parameter_report,
               'vocabulary': names, 'device': args.device, 'timings_ms': timings,
               'timing_scope': 'image forward + crop preprocessing + similarity + NMS; excludes loading/text encoding/image disk IO',
               'baseline_limit': 'COCO-biased proposals plus dense windows; open vocabulary does not guarantee novel-object recall',
               'threshold': args.threshold, 'accuracy_benchmark': False})


def fixtures(args):
    """Deterministic synthetic annotations for plumbing tests, never Qwen outputs."""
    import random
    rng = random.Random(42)
    out = Path(args.out)
    names = ['red rectangle', 'blue circle', 'green triangle']
    write_json(out / 'classes.json', names)
    for split, count in [('train', 6), ('val', 3)]:
        records = []
        for i in range(count):
            image = Image.new('RGB', (256, 192), 'white')
            draw = ImageDraw.Draw(image)
            boxes = []
            for j, name in enumerate(names):
                x, y = 8 + j * 80 + rng.randint(0, 8), rng.randint(20, 100)
                xyxy = [x, y, x + 45, y + 50]
                if j == 0:
                    draw.rectangle(xyxy, fill='red')
                elif j == 1:
                    draw.ellipse(xyxy, fill='blue')
                else:
                    draw.polygon([(x, y + 50), (x + 22, y), (x + 45, y + 50)], fill='green')
                boxes.append({'label': name, 'xyxy': xyxy, 'score': None})
            folder = out / split
            folder.mkdir(parents=True, exist_ok=True)
            image.save(folder / f'{i}.png')
            records.append({'file_name': f'{i}.png', 'width': 256, 'height': 192, 'boxes': boxes,
                            'provenance': 'synthetic-fixture-not-qwen', 'image_sha256': digest_file(folder / f'{i}.png')})
        write_records(out / f'{split}.jsonl', records)
    print(f'合成流程测试数据已生成: {out}（不是 Qwen 输出或真实精度数据）')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=2)
    sub = parser.add_subparsers(dest='command', required=True)

    def clip_options(p):
        p.add_argument('--clip-weights', default=str(ROOT / 'assets/tinyclip.pt'))
        p.add_argument('--device', default='cpu')

    def vocabulary_options(p, with_cache=False):
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument('--classes', help='JSON 字符串数组；仅为这次查询词表，不是固定分类头')
        g.add_argument('--texts', nargs='+', help='任意英文目标名称；不需要重新训练分类头')
        if with_cache:
            g.add_argument('--vocab', help='encode 命令生成的文本原型文件')

    p = sub.add_parser('label', help='使用 Qwen3-VL 离线生成框标注')
    p.add_argument('--images', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--classes', help='不提供则要求教师自由发现类别')
    p.add_argument('--chunk-size', type=int, default=16)
    p.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    p.add_argument('--model', default='Qwen/Qwen3-VL-8B-Instruct')
    from teacher import label
    p.set_defaults(func=label)

    p = sub.add_parser('encode', help='为任意新类别计算文本原型')
    vocabulary_options(p)
    clip_options(p)
    p.add_argument('--out', required=True)
    p.set_defaults(func=encode)

    p = sub.add_parser('detect', help='基线或蒸馏后的学生进行开放词汇检测')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--image')
    g.add_argument('--images')
    vocabulary_options(p, with_cache=True)
    clip_options(p)
    p.add_argument('--backend', choices=['baseline', 'student'], default='baseline')
    p.add_argument('--checkpoint')
    p.add_argument('--dfine-weights', default=str(ROOT / 'assets/dfine_n_coco.pth'))
    p.add_argument('--size', type=int, default=640)
    p.add_argument('--proposals', type=int, default=24)
    p.add_argument('--no-dense', action='store_true', help='降低耗时，但会进一步限制未见目标召回')
    p.add_argument('--threshold', type=float, default=0.27)
    p.add_argument('--nms-iou', type=float, default=0.5)
    p.add_argument('--max-detections', type=int, default=30)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--out', required=True)
    p.set_defaults(func=detect)

    p = sub.add_parser('train', help='Qwen 框监督 + 区域语义蒸馏')
    p.add_argument('--images', required=True)
    p.add_argument('--labels', required=True, help='JSONL 框标注')
    p.add_argument('--classes', required=True, help='训练阶段的语义词表；推理时可换成新词表')
    clip_options(p)
    p.add_argument('--init-weights', default=str(ROOT / 'assets/dfine_n_coco.pth'))
    p.add_argument('--variant', choices=['nano', 'micro'], default='nano')
    p.add_argument('--size', type=int, default=640)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--max-steps', type=int, default=0, help='0 表示不限步数；非零用于 smoke test')
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--semantic-weight', type=float, default=2.)
    p.add_argument('--relation-weight', type=float, default=1.)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--freeze-backbone', action='store_true')
    p.add_argument('--include-empty', action='store_true', help='仅在人工确认无目标后启用')
    p.add_argument('--out', required=True)
    from train import train
    p.set_defaults(func=train)

    p = sub.add_parser('evaluate', help='独立人工标注上的 F1 / 类别召回，区分 base/novel')
    p.add_argument('--ground-truth', required=True)
    p.add_argument('--predictions', required=True)
    p.add_argument('--base-classes', help='学生检测训练中出现的名称，用于 novel 分组')
    p.add_argument('--student-checkpoint', help='检查评测图片与该 checkpoint 训练图片的 SHA256 是否重叠')
    p.add_argument('--threshold', type=float, default=0.25)
    p.add_argument('--iou', type=float, default=0.5)
    p.add_argument('--out', required=True)
    from evaluate import evaluate
    p.set_defaults(func=evaluate)

    p = sub.add_parser('fixtures', help='生成合成数据以验证训练流程')
    p.add_argument('--out', default='work/fixtures')
    p.set_defaults(func=fixtures)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads 必须为正数')
    for key in ('epochs', 'batch', 'chunk_size', 'proposals'):
        if hasattr(args, key) and getattr(args, key) <= 0:
            parser.error(f'{key} 必须为正数')
    torch.set_num_threads(args.threads)
    args.func(args)


if __name__ == '__main__':
    main()
