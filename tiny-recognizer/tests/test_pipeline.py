import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from common import parse_qwen_boxes, read_records, write_records
from evaluate import evaluate_records
from models import OpenVocabularyDETR, count_parameters


def test_qwen_coordinate_conversion_and_strict_validation():
    text = '```json\n[{"label":"screwdriver","bbox_format_1000":[100,200,800,900]}]\n```'
    box = parse_qwen_boxes(text, 200, 100, ['screwdriver'])[0]
    assert box['xyxy'] == [20., 20., 160., 90.]
    assert box['score'] is None
    for invalid in ['[800,200,100,900]', '[0,0,1001,900]', '[0,0,NaN,900]', '[0,0,0,900]']:
        with pytest.raises(ValueError):
            parse_qwen_boxes('[{"label":"x","bbox_format_1000":' + invalid + '}]', 200, 100)
    with pytest.raises(ValueError, match='类别表外'):
        parse_qwen_boxes(text, 200, 100, ['wrench'])


def test_vocabulary_changes_without_changing_parameters():
    torch.set_num_threads(2)
    model = OpenVocabularyDETR(size=128).eval()
    before = count_parameters(model)
    model.set_vocabulary(['dog', 'cat'], torch.randn(2, 512))
    with torch.no_grad():
        assert model(torch.rand(1, 3, 128, 128))['pred_logits'].shape[-1] == 2
        model.set_vocabulary([f'novel object {i}' for i in range(137)], torch.randn(137, 512))
        assert model(torch.rand(1, 3, 128, 128))['pred_logits'].shape[-1] == 137
    assert count_parameters(model) == before
    assert not model.semantic_head.prototypes.requires_grad
    assert not any('denoising_class_embed' in name for name, _ in model.named_parameters())


def test_one_to_one_matching_and_novel_split():
    gt = [{'file_name': 'x.png', 'width': 100, 'height': 100, 'boxes': [
        {'label': 'novel', 'xyxy': [0, 0, 20, 20]}]}]
    pred = [{'file_name': 'x.png', 'width': 100, 'height': 100, 'boxes': [
        {'label': 'novel', 'xyxy': [0, 0, 20, 20], 'score': .9},
        {'label': 'novel', 'xyxy': [0, 0, 20, 20], 'score': .8}]}]
    report = evaluate_records(gt, pred, base_classes=['base'])
    assert report['all']['tp'] == 1
    assert report['all']['fp'] == 1
    assert report['novel']['recall'] == 1
    assert report['base']['supported_classes'] == 0
    with pytest.raises(ValueError, match='每一张'):
        evaluate_records(gt, [])


def test_failed_teacher_request_is_not_a_negative_label(tmp_path, monkeypatch):
    import teacher
    image_dir = tmp_path / 'images'
    image_dir.mkdir()
    Image.new('RGB', (20, 10)).save(image_dir / 'x.png')
    def fail(*_):
        raise ValueError('truncated output')
    monkeypatch.setattr(teacher, 'ask_qwen', fail)
    args = SimpleNamespace(images=str(image_dir), classes=None, out=str(tmp_path / 'labels.jsonl'),
                           model='test-only-mock', base_url='mock://no-network', chunk_size=16)
    with pytest.raises(RuntimeError):
        teacher.label(args)
    assert read_records(args.out) == []
    assert len(json.loads((tmp_path / 'labels.errors.json').read_text())) == 1


def test_teacher_cache_invalidated_by_image_content(tmp_path, monkeypatch):
    import teacher
    folder = tmp_path / 'images'
    folder.mkdir()
    Image.new('RGB', (20, 10)).save(folder / 'x.png')
    calls = []
    def answer(*_):
        calls.append(1)
        return [], '[]', 20, 10
    monkeypatch.setattr(teacher, 'ask_qwen', answer)
    args = SimpleNamespace(images=str(folder), classes=None, out=str(tmp_path / 'labels.jsonl'),
                           model='test-only-mock', base_url='mock://no-network', chunk_size=16)
    teacher.label(args)
    teacher.label(args)
    assert len(calls) == 1
    Image.new('RGB', (20, 10), 'red').save(folder / 'x.png')
    teacher.label(args)
    assert len(calls) == 2


def test_coco_ap_perfect_prediction_and_teacher_unranked():
    from evaluate import coco_ap
    truth = [{'file_name': 'a.png', 'width': 100, 'height': 100, 'boxes': [
        {'label': 'novel', 'xyxy': [10, 10, 70, 70], 'score': None}]}]
    prediction = [{'file_name': 'a.png', 'width': 100, 'height': 100, 'boxes': [
        {'label': 'novel', 'xyxy': [10, 10, 70, 70], 'score': .9}]}]
    result = coco_ap(truth, prediction, ['other'])
    assert result['all']['ap50_95'] == pytest.approx(1.)
    assert result['novel']['ap50_95'] == pytest.approx(1.)
    assert result['base'] is None
    assert not coco_ap(truth, truth)['available']


def test_reject_training_eval_overlap(tmp_path):
    from evaluate import evaluate
    gt = [{'file_name': 'a.png', 'width': 100, 'height': 100, 'boxes': [], 'image_sha256': 'identical-image'}]
    write_records(tmp_path / 'gt.jsonl', gt)
    write_records(tmp_path / 'pred.jsonl', gt)
    torch.save({'training_image_sha256': ['identical-image']}, tmp_path / 'model.pt')
    args = SimpleNamespace(ground_truth=tmp_path / 'gt.jsonl', predictions=tmp_path / 'pred.jsonl',
                           student_checkpoint=tmp_path / 'model.pt', threshold=.25, iou=.5,
                           out=tmp_path / 'metrics.json', base_classes=None)
    with pytest.raises(ValueError, match='相同内容'):
        evaluate(args)
    assert not args.out.exists()
