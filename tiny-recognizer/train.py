"""Offline Qwen box supervision + frozen region semantic distillation."""
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from common import (classes_from, digest_file, image_path, read_records,
                    validate_xyxy, write_json)
from models import (OpenVocabularyDETR, TinyCLIP, ROOT, count_parameters)


class RegionDataset(Dataset):
    def __init__(self, images, records, names, clip, size=640, include_empty=False):
        self.root = Path(images).resolve()
        self.records = read_records(records)
        if len({r['file_name'] for r in self.records}) != len(self.records):
            raise ValueError('训练集中图片重复。')
        self.names, self.size = names, size
        self.ids = {name: i for i, name in enumerate(names)}
        self.items = []
        for record in self.records:
            path = image_path(self.root, record['file_name'])
            with Image.open(path) as original:
                image = ImageOps.exif_transpose(original).convert('RGB')
            width, height = image.size
            if (width, height) != (record['width'], record['height']):
                raise ValueError(f'标注图片尺寸不一致: {path}')
            if 'image_sha256' in record and digest_file(path) != record['image_sha256']:
                raise ValueError(f'标注之后图片发生变化: {path}')
            boxes = record['boxes']
            if not boxes and not include_empty:
                continue
            for box in boxes:
                validate_xyxy(box['xyxy'], width, height)
                if box['label'] not in self.ids:
                    raise ValueError(f'训练词表缺少标签: {box["label"]}')
            # Region encoder is training-only; no image crops enter deployment.
            features = clip.images([image.crop(b['xyxy']) for b in boxes]).cpu()
            self.items.append((record, features))
        if not self.items:
            raise ValueError('没有可训练标注；空标注默认排除，避免把漏检当负样本。')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        record, features = self.items[index]
        with Image.open(image_path(self.root, record['file_name'])) as original:
            image = ImageOps.exif_transpose(original).convert('RGB')
        width, height = image.size
        boxes = torch.tensor([b['xyxy'] for b in record['boxes']], dtype=torch.float32).reshape(-1, 4)
        if len(boxes):
            boxes = boxes / boxes.new_tensor([width, height, width, height])
            boxes = torch.cat([(boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2]], -1)
        tensor = TF.to_tensor(TF.resize(image, [self.size, self.size]))
        target = {'boxes': boxes, 'labels': torch.tensor([self.ids[b['label']] for b in record['boxes']], dtype=torch.int64),
                  'region_features': features}
        return tensor, target


def collate(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


def region_distillation_loss(model, outputs, targets, matcher, temperature=0.1):
    indices = matcher({'pred_logits': outputs['pred_logits'], 'pred_boxes': outputs['pred_boxes']}, targets)['indices']
    embeddings = model.semantic_head.last_embedding
    source, reference = [], []
    for batch_id, (prediction_ids, target_ids) in enumerate(indices):
        if len(prediction_ids):
            source.append(embeddings[batch_id, prediction_ids])
            reference.append(targets[batch_id]['region_features'][target_ids])
    if not source:
        return embeddings.sum() * 0, embeddings.sum() * 0
    source, reference = torch.cat(source).float(), torch.cat(reference).float()
    alignment = (1 - F.cosine_similarity(source, reference, dim=-1)).mean()
    prototypes = model.semantic_head.prototypes.float()
    # Preserve relationships to the full training vocabulary, not just one hot IDs.
    soft_targets = F.softmax(reference @ prototypes.T / temperature, -1)
    relation = F.kl_div(F.log_softmax(source @ prototypes.T / temperature, -1),
                        soft_targets, reduction='batchmean') * temperature ** 2
    return alignment, relation


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('当前 PyTorch 无可用 CUDA；请安装 CUDA 版本或显式 --device cpu。')
    names = classes_from(args.classes)
    clip = TinyCLIP(args.clip_weights, args.device)
    prototypes = clip.texts(names).detach()
    dataset = RegionDataset(args.images, args.labels, names, clip, args.size, args.include_empty)
    text_hash = digest_file(args.clip_weights)
    clip_counts = {'auxiliary_visual': clip.image_parameters, 'text_encoder': clip.text_parameters}
    del clip  # Region features and text prototypes are cached in RAM for the demo.
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()
    model = OpenVocabularyDETR(args.variant, args.size, args.init_weights).to(args.device)
    model.set_vocabulary(names, prototypes)
    if args.freeze_backbone:
        model.model.backbone.requires_grad_(False)
    criterion = model.criterion().to(args.device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(dataset, batch_size=min(args.batch, len(dataset)), shuffle=True,
                        collate_fn=collate, num_workers=0)
    amp = args.amp and args.device.startswith('cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    history, steps = [], 0
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(args.epochs):
        model.train()
        if args.freeze_backbone:
            model.model.backbone.eval()
        for images, targets in loader:
            images = images.to(args.device)
            targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=torch.device(args.device).type, dtype=torch.float16, enabled=amp):
                outputs = model(images, targets)
                detection_losses = criterion(outputs, targets)
                box_loss = sum(detection_losses.values())
                alignment, relation = region_distillation_loss(model, outputs, targets, criterion.matcher)
                loss = box_loss + args.semantic_weight * alignment + args.relation_weight * relation
            if not torch.isfinite(loss):
                raise RuntimeError('loss 非有限值；未保存该步，请检查标注或关闭 AMP。')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            steps += 1
            item = {'epoch': epoch, 'step': steps, 'loss': float(loss.detach()),
                    'detection_loss': float(box_loss.detach()), 'semantic_loss': float(alignment.detach()),
                    'relation_loss': float(relation.detach()), 'gradient_norm': float(norm)}
            history.append(item)
            print(json.dumps(item), flush=True)
            if args.max_steps and steps >= args.max_steps:
                break
        model.save(out / 'last.pt', text_hash, {'steps': steps, 'seed': args.seed,
                   'training_image_sha256': [digest_file(image_path(args.images, r['file_name'])) for r, _ in dataset.items]})
        if args.max_steps and steps >= args.max_steps:
            break
    report = {'architecture': f'open-vocabulary-dfine-{args.variant}', 'parameters': count_parameters(model),
              'training_classes': names, 'training_images': len(dataset), 'steps': steps,
              'initialization': model.initialization, 'auxiliary_parameters': clip_counts,
              'history': history, 'accuracy_claim': 'none: training smoke test is not an accuracy benchmark',
              'device': args.device, 'amp': amp, 'seed': args.seed}
    if args.device.startswith('cuda'):
        report['peak_gpu_memory_mb'] = torch.cuda.max_memory_allocated() / 1024 ** 2
    write_json(out / 'train_report.json', report)
    return report
