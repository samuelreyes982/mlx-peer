# MLX Peer website

Static product website migrated from the existing Sites publication. The original layout, device illustration, mission, interactive architecture explanation, and experiment links are preserved. No framework build, environment secrets, database, or analytics SDK is required.

Live site: **https://mlx-peer.vercel.app**. Vercel project: `mlx-peer`. Production deploys automatically from this repository's `main` branch. The first production migration was verified on September 23, 2026: the page, styles, script and hero image were publicly available over HTTPS and matched the source files exactly.

## Local preview

From this directory:

```sh
python3 -m http.server 4173 --directory public
```

## Vercel

- Project root: `website` within the MLX Peer repository.
- Framework: Other.
- Build and install commands: none.
- Output directory: `public`.
- Configuration: `vercel.json`.

Commit changes under `website/` and push to `main` to deploy through the connected GitHub integration. Check the Vercel deployment reaches Ready before treating an update as published. Local Vercel account links and environment files are ignored and must not be committed.

Add the confirmed custom domain to the Vercel project, inspect its required DNS values, and apply only those records at the domain's existing DNS provider. Preserve unrelated mail and verification records. Vercel provisions HTTPS after domain verification.

The Sites source was synchronized at commit `10bea7c53a5d3f09118fc22ebafb42cba1cb9989` before this migration. The old site remains available until the new host and custom domain are verified.
