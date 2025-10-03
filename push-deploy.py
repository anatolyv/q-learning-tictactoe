#!/usr/bin/env python3
"""
🚀 Q-Learning App Push-to-Deploy Script
Handles git operations and triggers automatic deployment
"""

import subprocess
import sys
import os
from datetime import datetime

def run_command(cmd, description):
    """Run a shell command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Error: {result.stderr}")
            return False
        if result.stdout.strip():
            print(f"✅ {result.stdout.strip()}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("🎯 Q-Learning App Deployment Pipeline")
    print("=" * 50)

    # Get commit message
    if len(sys.argv) > 1:
        commit_msg = " ".join(sys.argv[1:])
    else:
        commit_msg = f"Update Q-Learning app - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    # Check for changes
    result = subprocess.run("git status --porcelain", shell=True, capture_output=True, text=True)
    if result.stdout.strip():
        print("📝 Changes detected, preparing commit...")

        # Add all changes
        if not run_command("git add .", "Adding changes"):
            sys.exit(1)

        # Commit changes
        commit_cmd = f'git commit -m "{commit_msg}"'
        if not run_command(commit_cmd, "Committing changes"):
            sys.exit(1)
    else:
        print("ℹ️  No changes to commit")

    # Push to GitHub
    print("\n🚀 Pushing to GitHub (triggers auto-deployment)...")
    if not run_command("git push origin main", "Pushing to GitHub"):
        print("\n🔐 Authentication needed. Please:")
        print("1. Get GitHub Personal Access Token: https://github.com/settings/tokens")
        print("2. When prompted for password, use your token (not GitHub password)")
        print("3. Or set up SSH keys for passwordless authentication")
        sys.exit(1)

    print("\n🎉 SUCCESS! Deployment pipeline triggered!")
    print("🌐 Your app will be live in 2-3 minutes")
    print("📊 Monitor deployment:")
    print("   - GitHub Actions: https://github.com/anatolyv/q-learning-tictactoe/actions")
    print("   - Railway Dashboard: https://railway.app/dashboard")

if __name__ == "__main__":
    main()