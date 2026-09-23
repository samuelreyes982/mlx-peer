# MLX Peer website

Static product website migrated from the existing Sites publication. The original layout, device illustration, mission, interactive architecture explanation, and experiment links are preserved. No framework build, environment secrets, database, or analytics SDK is required.

Live site: **https://mlx-peer.com** (also available at https://mlx-peer.vercel.app). Vercel project: `mlx-peer`. Production deploys automatically from this repository's `main` branch. The first production migration was verified on September 23, 2026: the page, styles, script and hero image were publicly available over HTTPS and matched the source files exactly.

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

The domain is registered at Spaceship and keeps Spaceship nameservers. Both `mlx-peer.com` and `www.mlx-peer.com` are attached to the Vercel project. The `www` hostname redirects permanently to `https://mlx-peer.com`, preserving the path.

DNS values supplied by Vercel on September 23, 2026:

| Type | Host | Value | TTL |
| --- | --- | --- | --- |
| A | @ | 216.198.79.1 | 30 minutes |
| A | @ | 64.29.17.1 | 30 minutes |
| CNAME | www | 43aa4dbc7aefe229.vercel-dns-017.com | 30 minutes |

Before future DNS changes, recheck Vercel's recommended records and preserve unrelated mail and verification records. Vercel manages the HTTPS certificate.

The Sites source was synchronized at commit `10bea7c53a5d3f09118fc22ebafb42cba1cb9989` before this migration. The old site remains available until the new host and custom domain are verified.
