"""Sign a copy of the packaged Mac app, including its embedded Python engine.

Signing alone does not notarize the app or make it ready for public distribution.
"""
import argparse
from pathlib import Path
import shutil
import subprocess

MACH_O_MAGIC = {
    b'\xfe\xed\xfa\xce', b'\xce\xfa\xed\xfe',
    b'\xfe\xed\xfa\xcf', b'\xcf\xfa\xed\xfe',
    b'\xca\xfe\xba\xbe', b'\xbe\xba\xfe\xca',
    b'\xca\xfe\xba\xbf', b'\xbf\xba\xfe\xca',
}


def signable_paths(app):
    """Return physical binaries and enclosing code bundles, children first."""
    paths = []
    for path in app.rglob('*'):
        if path.is_symlink():
            target = path.resolve(strict=True)
            if not target.is_relative_to(app):
                raise ValueError(f'Symlink leaves the app: {path}')
            continue
        if path.is_dir() and path.suffix in ('.app', '.framework', '.xpc', '.appex'):
            paths.append(path)
        elif path.is_file():
            with path.open('rb') as source:
                if source.read(4) in MACH_O_MAGIC:
                    paths.append(path)
    return sorted(paths, key=lambda path: (-len(path.parts), str(path))) + [app]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app', type=Path, required=True, help='Existing packaged .app')
    parser.add_argument('--output', type=Path, required=True, help='New .app path; must not exist')
    parser.add_argument('--identity', required=True, help='Exact installed signing certificate name')
    parser.add_argument('--development-check', action='store_true',
                        help='Permit Apple Development signing for local runtime testing only')
    parser.add_argument('--dry-run', action='store_true', help='Inspect without copying or signing')
    options = parser.parse_args()
    app = options.app.resolve(strict=True)
    output = options.output.absolute()
    if app.suffix != '.app' or not (app / 'Contents/Info.plist').is_file():
        parser.error('--app must be a packaged macOS app')
    if output.suffix != '.app' or output.exists() or output.is_symlink():
        parser.error('--output must be a new .app path')
    if output.resolve().is_relative_to(app):
        parser.error('--output must be outside the source app')
    prefix = 'Apple Development: ' if options.development_check else 'Developer ID Application: '
    if not options.identity.startswith(prefix):
        parser.error(f'Expected a {prefix.strip()} signing identity')
    inventory = signable_paths(app)
    print(f'Found {len(inventory)} code objects; source app will be preserved.')
    if options.dry_run:
        for path in inventory:
            print(path.relative_to(app))
        return
    identities = subprocess.check_output(
        ['security', 'find-identity', '-v', '-p', 'codesigning'], text=True)
    if f'"{options.identity}"' not in identities:
        parser.error('The requested signing identity is not installed with a private key')
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(app, output, symlinks=True)
    # Sign leaf Mach-O files before frameworks, helper apps, and the outer app.
    # No library-validation or executable-memory exceptions are added by default.
    for path in signable_paths(output.resolve()):
        subprocess.run(['codesign', '--force', '--options', 'runtime', '--timestamp',
                        '--sign', options.identity, str(path)], check=True)
    subprocess.run(['codesign', '--verify', '--deep', '--strict', '--verbose=2', str(output)], check=True)
    subprocess.run(['codesign', '--display', '--verbose=4', str(output)], check=True)
    if options.development_check:
        print(f'LOCAL DEVELOPMENT CHECK ONLY; not suitable for distribution: {output}')
    else:
        print(f'Signed app (not yet notarized): {output}')
        print('Notarize with Apple, staple the accepted ticket, and assess with spctl before release.')


if __name__ == '__main__':
    main()
