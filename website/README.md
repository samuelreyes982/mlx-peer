# MLX Peer website

Static product website migrated from the existing Sites publication. The original layout, device illustration, mission, interactive architecture explanation, and experiment links are preserved. No framework build, environment secrets, database, or analytics SDK is required.

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

To deploy from this directory with an authenticated Vercel CLI:

```sh
npx vercel
npx vercel --prod
```

Add the confirmed custom domain to the Vercel project, inspect its required DNS values, and apply only those records at the domain's existing DNS provider. Preserve unrelated mail and verification records. Vercel provisions HTTPS after domain verification.

The Sites source was synchronized at commit `10bea7c53a5d3f09118fc22ebafb42cba1cb9989` before this migration. The old site remains available until the new host and custom domain are verified.
