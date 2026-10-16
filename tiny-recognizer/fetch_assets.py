"""Download public official checkpoints. TLS is verified and SHA256 checked."""
import argparse
from pathlib import Path
import urllib.request

from common import digest_file

FILES = {
    'dfine_n_coco.pth': (
        'https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_n_coco.pth',
        '41973938d2784d38a9836990d805b8392855ebf611aba55f0f7add90e110744c'),
    'tinyclip.pt': (
        'https://github.com/wkcn/TinyCLIP-model-zoo/releases/download/checkpoints/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M.pt',
        '3d9f86a556cd13acc1aa6cc18495a79ffd1f58b6f17a552400bdefd244c3a92c'),
    'dog.jpg': (
        'https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg',
        'f3f87bb8ab3c26c7ecfd3ac60421d7f32b0503d1d6c5baf8bac42ed93d86351a'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(Path(__file__).resolve().parent / 'assets'))
    args = parser.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in FILES.items():
        path = root / name
        if path.exists() and digest_file(path) == expected:
            print(f'已存在且校验通过: {name}')
            continue
        tmp = path.with_suffix(path.suffix + '.part')
        print(f'下载 {name}', flush=True)
        with urllib.request.urlopen(url, timeout=120) as source, tmp.open('wb') as target:
            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                target.write(chunk)
        if digest_file(tmp) != expected:
            tmp.unlink()
            raise RuntimeError(f'SHA256 校验失败: {name}')
        tmp.replace(path)


if __name__ == '__main__':
    main()
