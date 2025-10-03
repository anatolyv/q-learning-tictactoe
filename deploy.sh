#!/bin/bash

# 🚀 Q-Learning App Deployment Script
# This script commits changes and triggers deployment

set -e

echo "🎯 Q-Learning Deployment Pipeline Starting..."

# Check if we have changes
if ! git diff --quiet HEAD; then
    echo "📝 Changes detected, committing..."

    # Add all changes
    git add .

    # Get commit message from user or use default
    if [ "$#" -eq 0 ]; then
        COMMIT_MSG="Update Q-Learning app - $(date '+%Y-%m-%d %H:%M')"
    else
        COMMIT_MSG="$*"
    fi

    # Commit changes
    git commit -m "$COMMIT_MSG"
    echo "✅ Changes committed: $COMMIT_MSG"
else
    echo "ℹ️  No changes to commit"
fi

# Push to GitHub (triggers auto-deployment)
echo "🚀 Pushing to GitHub..."
git push origin main

echo "✅ Deployment pipeline triggered!"
echo "🌐 Your app will be live in 2-3 minutes"
echo "📊 Check deployment status at:"
echo "   - GitHub: https://github.com/anatolyv/q-learning-tictactoe/actions"
echo "   - Railway: https://railway.app/dashboard"
echo ""
echo "🎉 Done! Changes are being deployed automatically."