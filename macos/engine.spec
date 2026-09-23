# Build from the repository root with the pinned Python 3.12 environment.
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata

root = Path(SPECPATH).parent
datas = collect_data_files('mlx', includes=['lib/*.metallib'])
binaries = collect_dynamic_libs('mlx')
# Transformers discovers tokenizer modules by inspecting package source files.
datas += collect_data_files('transformers', include_py_files=True)
for distribution in ['mlx', 'mlx-metal', 'mlx-lm', 'transformers', 'tokenizers', 'numpy', 'safetensors', 'huggingface-hub', 'regex', 'tqdm', 'packaging', 'pyyaml', 'jinja2', 'sentencepiece']:
    datas += copy_metadata(distribution)
hiddenimports = collect_submodules('mlx') + collect_submodules('tokenizers')
hiddenimports += ['transformers.models.qwen2.tokenization_qwen2', 'transformers.models.qwen2.configuration_qwen2']
a = Analysis([str(root / 'scripts/desktop_entry.py')], pathex=[str(root / 'src')], binaries=binaries,
             datas=datas, hiddenimports=hiddenimports,
             excludes=['torch', 'tensorflow', 'jax', 'matplotlib', 'IPython', 'pytest'],
             noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='mlx-peer-engine', debug=False,
          bootloader_ignore_signals=False, strip=False, upx=False, console=True, target_arch='arm64')
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='mlx-peer-engine')
app = BUNDLE(coll, name='MLX Peer Engine.app', bundle_identifier='dev.mlxpeer.engine',
             info_plist={'LSUIElement': True, 'CFBundleShortVersionString': '0.2.0', 'CFBundleVersion': '3', 'LSMinimumSystemVersion': '14.0'})
