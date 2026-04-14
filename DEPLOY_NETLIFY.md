# Netlify Deployment Guide

GitHub Actions has permission issues that cannot be resolved. Use Netlify for automated deployment instead.

## Steps:

1. Create a Netlify account at https://www.netlify.com/
2. Click "Add new site" → "Import an existing project"
3. Connect to GitHub and select the `bettingsportsmodel` repository
4. Configure build settings:
   - Build command: `python generate_daily_card.py --skip-provider-sync`
   - Publish directory: `output`
5. Add environment variable:
   - Key: `THE_ODDS_API_KEY`
   - Value: `69ed5657a5448219c17d11c2066b41a0`
6. Click "Deploy site"

Netlify will build and deploy the site automatically. When you refresh the Netlify URL, you'll see the updated picks.
