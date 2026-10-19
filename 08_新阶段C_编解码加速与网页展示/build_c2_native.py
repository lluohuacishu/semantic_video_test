"""Build the opt-in native adapter into a new local evidence directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from c2_watch import utc, write

HERE = Path(__file__).resolve().parent
UPSTREAM_SHA256 = '66e9521b4485c1b36ee7527f1a6cd7c9fe7f8495f81420602fbdc26b7273bea6'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compiler', type=Path, default=Path('C:/mingw64/bin/gcc.exe'))
    args = parser.parse_args()
    files = [HERE / 'native/rans_byte.h', HERE / 'native/glc_rans.c']
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    if hashes['rans_byte.h'] != UPSTREAM_SHA256:
        raise ValueError('Pinned public-domain header changed')
    args.output.mkdir(parents=True, exist_ok=False)
    for path in files:
        shutil.copy2(path, args.output / path.name)
    target = args.output / 'glc_rans.dll'
    # Use relative ASCII filenames for reproducible tool paths; copy the binary
    # via Python. -B below selects the working bin/as.exe on this installation.
    command = [str(args.compiler), '-B', str(args.compiler.parent) + '/', '-std=c11', '-O3', '-Wall', '-Wextra', '-Werror', '-shared',
               '-static-libgcc', 'glc_rans.c', '-o', 'glc_rans.dll']
    with tempfile.TemporaryDirectory(prefix='glc_c2_build_') as temp:
        for path in files:
            shutil.copy2(path, Path(temp) / path.name)
        result = subprocess.run(command, cwd=temp, capture_output=True, text=True, encoding='utf-8', errors='replace',
                                env=dict(os.environ, PATH=str(args.compiler.parent) + os.pathsep + os.environ.get('PATH', '')))
        if result.returncode == 0:
            shutil.copy2(Path(temp) / 'glc_rans.dll', target)
    value = {'created_utc': utc(), 'argv': command, 'returncode': result.returncode,
             'stdout': result.stdout, 'stderr': result.stderr, 'source_sha256': hashes,
             'upstream_commit': 'c9d162d996fd600315af9ae8eb89d832576cb32d',
             'compiler': subprocess.run([str(args.compiler), '--version'], capture_output=True, text=True).stdout}
    if target.exists():
        value['library_sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
    write(args.output / 'build.json', value)
    if result.returncode:
        raise RuntimeError(result.stderr)
    print(json.dumps(value, ensure_ascii=False))


if __name__ == '__main__':
    main()
