# Owner actions — llm-eval-pipeline

This file records manual steps that cannot be automated via API.

## 1. GitHub repository social preview (manual upload)

Pages `og:image` (`https://ianlyoo.github.io/llm-eval-pipeline/assets/social-preview.png`) is served automatically after push. GitHub repository social preview (link unfurl card) requires manual upload:

- Path: GitHub repository → Settings → General → Social preview → Edit → Upload an image
- File: `docs/assets/social-preview.png` (1280×640, <1 MiB, background #0f172a)
- API limitation: GitHub has no write API for repository social preview image.

Verification after manual upload:

```bash
gh api graphql -f query='query{repository(owner:"ianlyoo",name:"llm-eval-pipeline"){openGraphImageUrl}}'
# before: https://opengraph.githubassets.com/... (default)
# after:  https://user-images.githubusercontent.com/... or repository-images URL
```

Live Pages preview (automatic):

```bash
curl -fsSL https://ianlyoo.github.io/llm-eval-pipeline/ | grep -o 'og:image[^>]*content="[^"]*"'
curl -fsSL https://ianlyoo.github.io/llm-eval-pipeline/assets/social-preview.png -o /tmp/p.png && ls -lh /tmp/p.png
```

## 2. Optional profile pin

- Path: Profile → Customize your pins → select `ianlyoo/llm-eval-pipeline`
- This is optional and also manual.

## 3. Do not attempt

- Do not attempt to upload repository social preview via API.
- Do not force push.
