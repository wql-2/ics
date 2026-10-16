"""One-to-one box matching on independent human annotations; novel/base split."""
from collections import Counter
import numpy as np
from scipy.optimize import linear_sum_assignment

from common import read_records, write_json


def iou_matrix(a, b):
    a, b = np.asarray(a, dtype=float).reshape(-1, 4), np.asarray(b, dtype=float).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(rb - lt, 0)
    intersection = wh[..., 0] * wh[..., 1]
    area_a = np.prod(np.maximum(a[:, 2:] - a[:, :2], 0), -1)
    area_b = np.prod(np.maximum(b[:, 2:] - b[:, :2], 0), -1)
    return intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-12)


def evaluate_records(truth, predictions, threshold=0.25, iou_threshold=0.5, base_classes=()):
    def index(records):
        result = {r['file_name']: r for r in records}
        if len(result) != len(records):
            raise ValueError('评测文件有重复图片。')
        return result
    gt, pred = index(truth), index(predictions)
    if set(gt) != set(pred):
        raise ValueError('预测必须包含评测集中每一张图片（包括空结果），且不能包含额外图片。')
    counts = {}
    for name, record in gt.items():
        if (record['width'], record['height']) != (pred[name]['width'], pred[name]['height']):
            raise ValueError(f'预测与真值图片尺寸不同: {name}')
        g = record['boxes']
        p = [b for b in pred[name]['boxes'] if b.get('score') is None or b['score'] >= threshold]
        for label in {b['label'] for b in g + p}:
            gold = [b['xyxy'] for b in g if b['label'] == label]
            guesses = [b['xyxy'] for b in p if b['label'] == label]
            ious = iou_matrix(gold, guesses)
            # Maximize number of valid matches first, total IoU second.
            r, c = linear_sum_assignment(-((ious >= iou_threshold) * 1000 + ious))
            tp = int((ious[r, c] >= iou_threshold).sum())
            count = counts.setdefault(label, Counter())
            count.update(tp=tp, fp=len(guesses) - tp, fn=len(gold) - tp, support=len(gold))
    def summarize(selected):
        tp = sum(v['tp'] for v in selected.values())
        fp = sum(v['fp'] for v in selected.values())
        fn = sum(v['fn'] for v in selected.values())
        recalls = [v['tp'] / v['support'] for v in selected.values() if v['support']]
        return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': tp / max(tp + fp, 1),
                'recall': tp / max(tp + fn, 1), 'f1': 2 * tp / max(2 * tp + fp + fn, 1),
                'macro_class_recall': float(np.mean(recalls)) if recalls else None,
                'supported_classes': sum(v['support'] > 0 for v in selected.values())}
    base = set(base_classes)
    return {'images': len(gt), 'iou': iou_threshold, 'score_threshold': threshold,
            'all': summarize(counts), 'base': summarize({k: v for k, v in counts.items() if k in base}),
            'novel': summarize({k: v for k, v in counts.items() if k not in base}),
            'per_class': {k: dict(v) for k, v in sorted(counts.items())},
            'novel_definition': 'absent from student detection training vocabulary; not necessarily absent from CLIP pretraining',
            'annotation_requirement': 'use independent human ground truth; teacher agreement is not accuracy'}


def evaluate(args):
    from common import classes_from
    base = classes_from(args.base_classes) if args.base_classes else []
    leakage_checked = False
    if args.student_checkpoint:
        import torch
        metadata = torch.load(args.student_checkpoint, map_location='cpu', weights_only=True)
        train_hashes = set(metadata.get('training_image_sha256', []))
        val_hashes = {r['image_sha256'] for r in read_records(args.ground_truth) if r.get('image_sha256')}
        if len(val_hashes) != len(read_records(args.ground_truth)):
            raise ValueError('检查训练/评测泄漏需要真值记录包含 image_sha256。')
        if train_hashes & val_hashes:
            raise ValueError('评测图片与学生训练图片有相同内容，拒绝报告泛化精度。')
        leakage_checked = True
    report = evaluate_records(read_records(args.ground_truth), read_records(args.predictions),
                              args.threshold, args.iou, base)
    report['coco_ap'] = coco_ap(read_records(args.ground_truth), read_records(args.predictions), base)
    if leakage_checked:
        report['train_eval_disjoint_by_sha256'] = True
    write_json(args.out, report)
    print(__import__('json').dumps(report, indent=2, ensure_ascii=False))
    return report


def coco_ap(truth, predictions, base_classes=()):
    """COCO AP50:95, without F1 threshold filtering. Teacher has no ranked scores."""
    from faster_coco_eval import COCO, COCOeval_faster
    if any(b.get('score') is None for r in predictions for b in r['boxes']):
        return {'available': False, 'reason': 'generative teacher has no calibrated/ranked detector scores; compare F1 and recall'}
    names = sorted({b['label'] for r in truth + predictions for b in r['boxes']})
    if not names:
        return {'available': False, 'reason': 'no categories in ground truth or predictions'}
    category = {name: i + 1 for i, name in enumerate(names)}
    image_ids = {r['file_name']: i + 1 for i, r in enumerate(truth)}
    annotations, detections = [], []
    for r in truth:
        for b in r['boxes']:
            x1, y1, x2, y2 = b['xyxy']
            annotations.append({'id': len(annotations) + 1, 'image_id': image_ids[r['file_name']],
                                'category_id': category[b['label']], 'bbox': [x1, y1, x2 - x1, y2 - y1],
                                'area': (x2 - x1) * (y2 - y1), 'iscrowd': 0})
    for r in predictions:
        for b in r['boxes']:
            x1, y1, x2, y2 = b['xyxy']
            detections.append({'image_id': image_ids[r['file_name']], 'category_id': category[b['label']],
                               'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': b['score']})
    if not detections:
        return {'available': True, 'all': {'ap50_95': 0., 'ap50': 0.}, 'base': None, 'novel': None}
    gt = COCO()
    gt.dataset = {'images': [{'id': image_ids[r['file_name']], 'width': r['width'], 'height': r['height']} for r in truth],
                  'categories': [{'id': i, 'name': n} for n, i in category.items()],
                  'annotations': annotations, 'info': {}}
    gt.createIndex()
    dt = gt.loadRes(detections)
    report = {'available': True, 'max_detections': 100,
              'note': 'AP uses saved scores before evaluate --threshold; export low-threshold predictions for meaningful recall'}
    for group in ['all', 'base', 'novel']:
        ids = [category[n] for n in names if group == 'all' or (group == 'base') == (n in set(base_classes))]
        supported = {a['category_id'] for a in annotations}
        ids = [i for i in ids if i in supported]
        if not ids:
            report[group] = None
            continue
        evaluator = COCOeval_faster(gt, dt, 'bbox')
        evaluator.params.catIds = ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        report[group] = {'ap50_95': float(evaluator.stats[0]), 'ap50': float(evaluator.stats[1])}
    return report
