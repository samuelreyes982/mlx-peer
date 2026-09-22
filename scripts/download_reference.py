"""Download the fixed public small-model reference; never run repository code."""
import hashlib
from pathlib import Path
from huggingface_hub import snapshot_download

REPOSITORY = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
WEIGHT_SHA256 = "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"


def main():
    target = Path(__file__).resolve().parent.parent / "artifacts" / "qwen-0.5b-source"
    snapshot_download(REPOSITORY, revision=REVISION, token=False, local_dir=target,
                      allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                                      "tokenizer_config.json", "LICENSE", "README.md"])
    with (target / "model.safetensors").open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != WEIGHT_SHA256:
        raise RuntimeError("Downloaded reference weight checksum mismatch")
    print(f"Verified {REPOSITORY}@{REVISION}: {target}")


if __name__ == "__main__":
    main()
