"""CPU regression runner with persisted commands, exit codes and untruncated logs."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys

from c2_watch import utc, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    library = args.native_library.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for name, folder in (('c', here), ('b', here.parent / '07_新阶段B_GLC语义保护与传输')):
        command = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(folder), '-p', 'test_*.py', '-v']
        result = subprocess.run(command, cwd=here.parent, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                env=dict(os.environ, GLC_RANS_DLL=str(library), PYTHONIOENCODING='utf-8'))
        path = args.output / f'{name}_regression.log'
        path.write_bytes(result.stdout)
        results.append({'suite': name, 'argv': command, 'returncode': result.returncode,
                        'log_path': str(path.resolve()), 'log_sha256': hashlib.sha256(result.stdout).hexdigest()})
        print(name + ': exit ' + str(result.returncode) + '\n' + result.stdout.decode('utf-8', errors='replace')[-700:], flush=True)
    write(args.output / 'checks.json', {'created_utc': utc(), 'native_library': str(library),
          'library_sha256': hashlib.sha256(library.read_bytes()).hexdigest(), 'results': results,
          'test_sources': {str(p.relative_to(here.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for folder in (here, here.parent / '07_新阶段B_GLC语义保护与传输') for p in folder.glob('test_*.py')},
          'all_passed': all(row['returncode'] == 0 for row in results)})
    sys.exit(0 if all(row['returncode'] == 0 for row in results) else 1)


if __name__ == '__main__':
    main()
