# Validation record

## Publication check — September 22, 2026

The source prepared for publication passed **105 Python tests and 25 subtests** on Apple Silicon using the existing pinned experiment environment:

```sh
PYTHONPATH=src python -m pytest -q
```

This exercises Python protocol, manifest, capacity, and MLX mechanics. It does not rerun a physical iPhone test or establish the unresolved 27B prefill parity criterion.

## Recorded device experiments

- [Initial validation summary](VALIDATION.json)
- [Physical iPhone reference comparison](IPHONE_VALIDATION.md)
- [USB execution and transport](USB_PROBE.md)
- [Exploratory 27B generation and limitations](QWEN38_SPLIT.md)
- [Subsequent functional smoke cases](QWEN38_FUNCTIONAL_RETEST.md)

The device reports retain their original dates, measurements, and failed checks. Local workstation paths were normalized for publication; model weights, raw fixtures, and credentials are excluded. The source uses pinned upstream dependencies rather than distributing local dependency checkouts.
