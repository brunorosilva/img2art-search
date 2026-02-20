# Deployment Options for img2art-search

This project has two components:
1. **Backend**: Python/Gradio ML service (image search via Pinecone + ViT model)
2. **Frontend**: Next.js static site with gallery UI

## Current Architecture

```
┌─────────────────┐         ┌──────────────────────────┐
│   Frontend      │  API    │   Backend (HF Space)     │
│   (Static)      │ ──────> │   chicelli-img2art-search│
│                 │         │   .hf.space              │
└─────────────────┘         └──────────────────────────┘
```

The frontend is fully client-side and calls your Hugging Face Space API for search functionality.

---

## Option 1: GitHub Pages (Project Site) ✅ Recommended

Deploy the frontend as a static site at `brunorosilva.github.io/img2art-search/`

### Setup Steps:
1. Go to repository **Settings** > **Pages**
2. Under "Build and deployment" > "Source", select **GitHub Actions**
3. Push any change to trigger the workflow (or manually run it from Actions tab)

### Pros:
- Free hosting
- Auto-deploys on push
- Works with existing backend on HF Space

### Cons:
- URL has `/img2art-search/` prefix
- Relies on HF Space being up

---

## Option 2: Vercel ✅ Best Developer Experience

Deploy the Next.js frontend on Vercel with automatic previews.

### Setup Steps:
1. Go to [vercel.com](https://vercel.com)
2. Import `brunorosilva/img2art-search` repository
3. Set root directory to `frontend`
4. Deploy

### Pros:
- Free tier available
- Preview deployments for PRs
- Custom domain support
- No configuration needed (works with Next.js natively)
- Edge network for fast global delivery

### Cons:
- Separate service to manage

---

## Option 3: Netlify

Similar to Vercel, deploy as a static site.

### Setup Steps:
1. Go to [netlify.com](https://netlify.com)
2. Connect GitHub repository
3. Build command: `cd frontend && npm run build`
4. Publish directory: `frontend/out`

### Pros:
- Free tier
- Easy setup
- Form handling, functions available

---

## Option 4: Deploy to Existing GitHub.io Site

Copy the frontend build to your `brunorosilva.github.io` repository.

### Setup Steps:
1. Build locally: `cd frontend && GITHUB_PAGES=true npm run build`
2. Copy `frontend/out/*` to your github.io repo
3. Push changes

### Pros:
- Uses your existing site
- Can be at root URL

### Cons:
- Manual process
- Mixes with personal site content

---

## Option 5: Hugging Face Spaces (Full Stack) ✅ Already Active

Your backend is already deployed here. You could also host the frontend alongside it.

**Current Demo**: https://huggingface.co/spaces/chicelli/img2art-search

### Pros:
- Everything in one place
- Free GPU/CPU tier
- ML-focused community

---

## Quick Comparison

| Option | Cost | Custom Domain | Auto Deploy | Best For |
|--------|------|---------------|-------------|----------|
| GitHub Pages | Free | Yes (with setup) | Yes | Simple hosting |
| Vercel | Free tier | Yes | Yes | Next.js apps |
| Netlify | Free tier | Yes | Yes | Static sites |
| HF Spaces | Free tier | No | Yes | ML demos |

---

## Recommended Path

1. **For showcasing**: Keep using Hugging Face Space (already works!)
2. **For custom frontend**: Deploy to Vercel for best Next.js experience
3. **For GitHub ecosystem**: Enable GitHub Actions in Pages settings

The GitHub Actions workflow in `.github/workflows/deploy.yml` is already configured. Just change your Pages source to "GitHub Actions" to use it.
