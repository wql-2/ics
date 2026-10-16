"""Pretrained open-vocabulary baseline and a trainable, vocabulary-dynamic DETR."""
import copy
import math
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF
import open_clip

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor' / 'dfine'))
from src.core import YAMLConfig
from src.data.dataset.coco_dataset import mscoco_category2name

COCO_NAMES = list(mscoco_category2name.values())


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


class TinyCLIP:
    """Frozen auxiliary encoder. Text includes token embeddings in parameter counts."""
    def __init__(self, weights=ROOT / 'assets/tinyclip.pt', device='cpu'):
        open_clip.add_model_config(ROOT / 'assets/tinyclip_config.json')
        self.model, _, self.transform = open_clip.create_model_and_transforms(
            'tinyclip_config')
        state = torch.load(weights, map_location='cpu', weights_only=True)
        state = state.get('state_dict', state)
        # Original TinyCLIP release uses separate training-wrapper prefixes.
        prefixes = ('_image_encoder.module.', '_text_encoder.module.', '_logit_scale.module.')
        clean = {}
        for key, value in state.items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    break
            clean[key] = value
        self.model.load_state_dict(clean, strict=True)
        self.model.eval().requires_grad_(False).to(device)
        self.tokenizer = open_clip.get_tokenizer('tinyclip_config')
        self.device = torch.device(device)
        self.image_parameters = count_parameters(self.model.visual)
        self.text_parameters = count_parameters(self.model) - self.image_parameters - 1
        self.vision_available = True

    def text_only(self):
        """Free the unused image encoder after encoding a new vocabulary."""
        del self.model.visual
        self.vision_available = False
        return self

    @torch.no_grad()
    def texts(self, names):
        # English templates match TinyCLIP's pretraining; Chinese labels can be mapped in the app.
        templates = ('a photo of a {}.', 'a close-up photo of a {}.')
        result = []
        for start in range(0, len(names), 64):
            chunk = names[start:start + 64]
            prompts = [t.format(name) for name in chunk for t in templates]
            z = self.model.encode_text(self.tokenizer(prompts).to(self.device), normalize=True)
            z = z.reshape(len(chunk), len(templates), -1).mean(1)
            result.append(F.normalize(z, dim=-1))
        return torch.cat(result)

    @torch.no_grad()
    def images(self, images, batch_size=16):
        if not self.vision_available:
            raise RuntimeError('text_only encoder cannot encode image regions')
        if not images:
            return torch.empty((0, 512), device=self.device)
        result = []
        for start in range(0, len(images), batch_size):
            batch = torch.stack([self.transform(im) for im in images[start:start + batch_size]]).to(self.device)
            result.append(self.model.encode_image(batch, normalize=True))
        return torch.cat(result)


def make_dfine(num_classes=80, variant='nano', size=640):
    if size < 128 or size % 32:
        raise ValueError('size 必须 >=128 且为 32 的倍数。')
    cfg = YAMLConfig(str(ROOT / 'vendor/dfine/configs/dfine/dfine_hgnetv2_n_coco.yml'))
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False  # Never download implicitly.
    cfg.yaml_cfg['num_classes'] = num_classes
    cfg.yaml_cfg['remap_mscoco_category'] = False
    cfg.yaml_cfg['eval_spatial_size'] = [size, size]
    # At 128px there are fewer than 300 feature locations.
    cfg.yaml_cfg['DFINETransformer']['num_queries'] = min(300, (size // 16) ** 2 + (size // 32) ** 2)
    if variant == 'micro':
        cfg.yaml_cfg['HybridEncoder'].update(hidden_dim=96, dim_feedforward=384)
        cfg.yaml_cfg['DFINETransformer'].update(hidden_dim=96, feat_channels=[96, 96],
                                               dim_feedforward=384, num_layers=2)
    elif variant != 'nano':
        raise ValueError(f'unknown variant: {variant}')
    return cfg.model, cfg


def pretrained_state(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    return checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint.get('model', checkpoint)


def load_matching(model, state):
    own = model.state_dict()
    matching = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    report = model.load_state_dict(matching, strict=False)
    return {'loaded_tensors': len(matching), 'total_tensors': len(own),
            'loaded_elements': sum(v.numel() for v in matching.values()),
            'missing_tensors': len(report.missing_keys)}


def image_tensor(image, size, device):
    return TF.to_tensor(TF.resize(image, [size, size])).unsqueeze(0).to(device)


def to_xyxy(boxes, width, height):
    xyxy = torch.cat([boxes[..., :2] - boxes[..., 2:] / 2,
                      boxes[..., :2] + boxes[..., 2:] / 2], dim=-1)
    return xyxy.clamp(0, 1) * xyxy.new_tensor([width, height, width, height])


class RegionBaseline:
    """D-FINE geometry + dense windows + pretrained TinyCLIP region semantics.

    COCO class names are discarded. Dense windows cover novel objects not
    proposed by the COCO detector, at a cost in latency and localization quality.
    """
    def __init__(self, device='cpu', size=640, weights=ROOT / 'assets/dfine_n_coco.pth',
                 clip_weights=ROOT / 'assets/tinyclip.pt'):
        self.detector, _ = make_dfine(size=size)
        self.detector.load_state_dict(pretrained_state(weights), strict=True)
        self.detector.eval().to(device)
        self.clip = TinyCLIP(clip_weights, device)
        self.device, self.size = device, size

    @torch.no_grad()
    def predict(self, image, names, text_features, proposal_count=32, dense=True):
        w, h = image.size
        outputs = self.detector(image_tensor(image, self.size, self.device))
        raw = outputs['pred_boxes'][0]
        # Only objectness ranking is reused, never the closed-vocabulary labels.
        objectness = outputs['pred_logits'][0].sigmoid().max(-1).values
        order = objectness.argsort(descending=True)[:proposal_count]
        geometry = to_xyxy(raw[order], w, h).tolist()
        candidates = [(b, float(objectness[i])) for b, i in zip(geometry, order) if objectness[i] >= 0.05]
        if dense:
            # A modest prior lets dense windows find novel objects, while strong
            # learned proposals rank above arbitrary object fragments. This is
            # a heuristic score, not a calibrated detection probability.
            candidates += [(b, 0.3) for b in dense_windows(w, h)]
        # Exact duplicates can otherwise create artificial repeated detections.
        unique = {}
        for b, prior in candidates:
            if b[2] - b[0] >= 8 and b[3] - b[1] >= 8:
                key = tuple(round(v, 2) for v in b)
                if key not in unique or unique[key][1] < prior:
                    unique[key] = (b, prior)
        if not unique:
            return []
        boxes, priors = zip(*unique.values())
        features = self.clip.images([image.crop(b) for b in boxes])
        similarity = features @ text_features.T
        values, indices = similarity.max(-1)
        return [{'label': names[int(i)], 'xyxy': b, 'score': float(s) * math.sqrt(prior),
                 'cosine_similarity': float(s), 'proposal_prior': prior,
                 'score_type': 'cosine_times_sqrt_proposal_prior'}
                for b, s, i, prior in zip(boxes, values, indices, priors)]


def dense_windows(width, height):
    boxes = [[0., 0., float(width), float(height)]]
    for scale in (0.35, 0.55, 0.8):
        for aspect in (0.75, 1., 1.33):
            bw, bh = min(width, width * scale * aspect), min(height, height * scale / aspect)
            for iy in range(3):
                for ix in range(3):
                    x, y = ix * (width - bw) / 2, iy * (height - bh) / 2
                    boxes.append([x, y, x + bw, y + bh])
    return boxes


class CosineHead(nn.Module):
    """Number of trainable weights does not depend on vocabulary size."""
    def __init__(self, hidden_dim, embed_dim=512):
        super().__init__()
        self.projection = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.)))
        self.bias = nn.Parameter(torch.tensor(-5.))
        self.register_buffer('prototypes', torch.empty((0, embed_dim)), persistent=False)
        self.last_embedding = None

    def forward(self, tokens):
        if not self.prototypes.numel():
            raise RuntimeError('请先 set_vocabulary()')
        self.last_embedding = F.normalize(self.projection(tokens), dim=-1)
        return self.log_scale.exp().clamp(max=100) * self.last_embedding @ self.prototypes.T + self.bias


class OpenVocabularyDETR(nn.Module):
    """Experimental student, needs broad region/text alignment training.

    Pretrained D-FINE weights initialize geometry, not zero-shot semantics.
    The semantic projector is learned using pseudo boxes plus CLIP crop features.
    """
    def __init__(self, variant='nano', size=640, init_weights=None):
        super().__init__()
        self.variant, self.size = variant, size
        self.model, self.cfg = make_dfine(80, variant, size)
        self.initialization = {}
        if init_weights:
            self.initialization = load_matching(self.model, pretrained_state(init_weights))
        decoder = self.model.decoder
        # Disable learned category-index denoising embeddings: no category-index
        # parameters remain in the student, including the training-only modules.
        decoder.num_denoising = 0
        del decoder.denoising_class_embed
        head = CosineHead(decoder.hidden_dim)
        decoder.enc_score_head = head
        decoder.dec_score_head = nn.ModuleList([head] * decoder.num_layers)
        self.names = []

    @property
    def semantic_head(self):
        return self.model.decoder.enc_score_head

    def set_vocabulary(self, names, prototypes):
        if prototypes.shape != (len(names), 512) or not names:
            raise ValueError('文本原型应为 [类别数,512]')
        self.names = list(names)
        self.model.decoder.num_classes = len(names)
        self.semantic_head.prototypes = F.normalize(prototypes.detach(), dim=-1).to(self.semantic_head.projection.weight)

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def criterion(self):
        self.cfg.yaml_cfg['num_classes'] = len(self.names)
        return self.cfg.criterion

    @torch.no_grad()
    def predict(self, image, names=None, prototypes=None):
        if names is not None:
            self.set_vocabulary(names, prototypes)
        device = self.semantic_head.projection.weight.device
        outputs = self(image_tensor(image, self.size, device))
        scores, labels = outputs['pred_logits'][0].sigmoid().max(-1)
        boxes = to_xyxy(outputs['pred_boxes'][0], *image.size)
        return [{'label': self.names[int(i)], 'xyxy': b.tolist(), 'score': float(s),
                 'score_type': 'student_sigmoid'} for b, s, i in zip(boxes, scores, labels)]

    def save(self, path, text_hash, extra=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({'format': 'ov-detr-demo-v1', 'model': self.state_dict(),
                    'variant': self.variant, 'size': self.size, 'text_sha256': text_hash,
                    'training_vocabulary': self.names, **(extra or {})}, path)

    @classmethod
    def load(cls, path, device='cpu'):
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state.get('format') != 'ov-detr-demo-v1':
            raise ValueError('需要本项目训练的开放词汇 checkpoint；不能直接把 COCO 权重当作开放词汇权重。')
        model = cls(state['variant'], state['size'])
        model.load_state_dict(state['model'], strict=True)
        return model.to(device).eval(), state


def filter_boxes(boxes, threshold=0.25, nms_iou=0.5, max_detections=100):
    from torchvision.ops import batched_nms
    boxes = [b for b in boxes if b['score'] >= threshold and b['xyxy'][2] > b['xyxy'][0]
             and b['xyxy'][3] > b['xyxy'][1]]
    if not boxes:
        return []
    ids = {name: i for i, name in enumerate(sorted({b['label'] for b in boxes}))}
    keep = batched_nms(torch.tensor([b['xyxy'] for b in boxes]),
                       torch.tensor([b['score'] for b in boxes]),
                       torch.tensor([ids[b['label']] for b in boxes]), nms_iou)
    return [boxes[i] for i in keep[:max_detections].tolist()]
