"""Build a self-contained Apple Silicon developer preview; no models or credentials included."""
import argparse
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import importlib.metadata
import sysconfig

ROOT = Path(__file__).resolve().parents[1]
VERSION = '0.2.0-alpha.2'


def run(*args):
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-engine', action='store_true', help='Reuse a previously built engine')
    options = parser.parse_args()
    if not options.skip_engine:
        run(sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean', 'macos/engine.spec')
    app = ROOT / 'dist/MLX Peer.app'
    if app.exists(): shutil.rmtree(app)
    contents = app / 'Contents'
    (contents / 'MacOS').mkdir(parents=True)
    (contents / 'Resources').mkdir()
    shutil.copytree(ROOT / 'dist/MLX Peer Engine.app', contents / 'Helpers/MLX Peer Engine.app', symlinks=True)
    run('xcrun', 'swiftc', '-parse-as-library', '-O', '-module-cache-path', ROOT / 'build/swift-cache',
        '-target', 'arm64-apple-macos14.0', 'macos/MLXPeerMac.swift', '-o', contents / 'MacOS/MLX Peer',
        '-framework', 'SwiftUI', '-framework', 'AppKit')
    info = {'CFBundleExecutable': 'MLX Peer', 'CFBundleIdentifier': 'dev.mlxpeer.mac',
            'CFBundleName': 'MLX Peer', 'CFBundleDisplayName': 'MLX Peer', 'CFBundlePackageType': 'APPL',
            'CFBundleShortVersionString': '0.2.0', 'CFBundleVersion': '3',
            'LSMinimumSystemVersion': '14.0', 'NSHighResolutionCapable': True,
            'NSHumanReadableCopyright': '© 2026 Samuel Reyes. MLX Peer and third-party licenses apply.'}
    with (contents / 'Info.plist').open('wb') as stream: plistlib.dump(info, stream)
    shutil.copy(ROOT / 'NOTICE', contents / 'Resources/NOTICE.txt')
    shutil.copytree(ROOT / 'licenses', contents / 'Resources/licenses')
    acknowledgements = contents / 'Resources/licenses/Dependencies'
    acknowledgements.mkdir()
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata['Name']
        for file in distribution.files or []:
            if any(part.lower().startswith(('license', 'copying', 'notice')) for part in file.parts) and '.dist-info/' in str(file):
                source = Path(distribution.locate_file(file))
                if source.is_file():
                    destination = acknowledgements / name / Path(*file.parts[1:])
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(source, destination)
    shutil.copy(Path(sysconfig.get_path('stdlib')) / 'LICENSE.txt', acknowledgements / 'Python-LICENSE.txt')
    # Ad-hoc signing enables local Apple Silicon execution. This is NOT Developer ID signing or notarization.
    run('codesign', '--force', '--deep', '--sign', '-', app)
    run('codesign', '--verify', '--deep', '--strict', app)
    archive = ROOT / f'dist/MLX-Peer-{VERSION}-macOS-arm64.zip'
    archive.unlink(missing_ok=True)
    run('ditto', '-c', '-k', '--sequesterRsrc', '--keepParent', app, archive)
    print(f'Built developer preview: {archive}')


if __name__ == '__main__': main()
