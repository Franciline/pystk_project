"""Build an isolated PySTK 0.7.4 with camera/rescue state patches (macOS setup)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.request

ROOT=Path(__file__).resolve().parents[2]


def build(root, jobs):
    root.mkdir(parents=True,exist_ok=True)
    source=root/'pysupertuxkart2-0.7.4'
    if not source.exists():
        with urllib.request.urlopen('https://pypi.org/pypi/PySuperTuxKart2/0.7.4/json') as response:
            info=json.load(response)
        release=next(item for item in info['urls'] if item['packagetype']=='sdist')
        archive=root/'source.tar.gz'
        urllib.request.urlretrieve(release['url'],archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=release['digests']['sha256']:
            raise RuntimeError('PySTK source checksum mismatch')
        with tarfile.open(archive) as package:
            package.extractall(root,filter='data')
    state=source/'pystk_cpp/state.cpp'
    if 'static_cast<unsigned int>(player_id) < Camera::getNumCameras()' not in state.read_text():
        subprocess.run(['patch','-p1','-i',str(ROOT/'patches/pystk2-camera-state.patch')],cwd=source,check=True)
    if 'bool rescue_active = false;' not in state.read_text():
        subprocess.run(['patch','-p1','-i',str(ROOT/'patches/pystk2-rescue-state.patch')],cwd=source,check=True)
    if '#include "guiengine/engine.hpp"' not in state.read_text():
        subprocess.run(['patch','-p1','-i',str(ROOT/'patches/pystk2-headless-camera-guard.patch')],cwd=source,check=True)
    tools=root/'tools'
    cmake=tools/'cmake/data/bin/cmake'
    ninja=tools/'bin/ninja'
    if not cmake.exists() or not ninja.exists():
        subprocess.run(['uv','pip','install','--python',sys.executable,'--target',str(tools),'cmake','ninja'],check=True)
    destination=root/'build312'
    command=[str(cmake),'-S',str(source),'-B',str(destination),'-G','Ninja',
             f'-DCMAKE_MAKE_PROGRAM={ninja}',f'-DPython_EXECUTABLE={sys.executable}',
             '-DCMAKE_BUILD_TYPE=Release','-DNO_SHADERC=ON','-DBUILD_RECORDER=OFF',
             '-DCHECK_ASSETS=OFF','-DUSE_WIIUSE=OFF','-DUSE_CRYPTO_OPENSSL=OFF','-DUSE_SQLITE3=OFF']
    if Path('/opt/homebrew').exists():
        command.append('-DCMAKE_PREFIX_PATH=/opt/homebrew')
    subprocess.run(command,check=True)
    subprocess.run([str(cmake),'--build',str(destination),'--target','pystk2','-j',str(jobs)],check=True)
    print(f'Camera/rescue binding built: {destination}')
    print(f'Use PYTHONPATH={destination} with the same Python interpreter to record videos.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-root',type=Path,default=Path('/tmp/pystk-camera-build'))
    parser.add_argument('--jobs',type=int,default=min(6,os.cpu_count() or 1))
    args=parser.parse_args()
    if args.jobs<1:
        parser.error('jobs must be positive')
    if shutil.which('uv') is None or shutil.which('patch') is None:
        parser.error('uv and patch must be installed')
    build(args.build_root.resolve(),args.jobs)
