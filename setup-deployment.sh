#!/bin/bash

# 🔧 One-time Setup for Automated Deployment
echo "🚀 Setting up automated deployment for Q-Learning App"

# Create .gitignore if it doesn't exist
if [ ! -f .gitignore ]; then
    cat > .gitignore << EOF
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt
.tox/
.coverage
.cache
nosetests.xml
coverage.xml
*.log
.DS_Store
*.pkl
demo_training_progress.png
EOF
    echo "✅ Created .gitignore"
fi

# Update requirements with gunicorn for production
if ! grep -q "gunicorn" requirements.txt; then
    echo "gunicorn>=20.1.0" >> requirements.txt
    echo "✅ Added gunicorn to requirements.txt"
fi

echo ""
echo "🎯 Next Steps:"
echo "1. Create GitHub repository at: https://github.com/anatolyv/q-learning-tictactoe"
echo "2. Get GitHub Personal Access Token from: https://github.com/settings/tokens"
echo "3. Run: git push origin main (enter token as password)"
echo "4. Set up Railway.app or Render.com with your GitHub repo"
echo ""
echo "📝 Then you can use: ./deploy.sh 'your commit message'"
echo "   This will automatically commit changes and deploy to production!"
echo ""
echo "🔐 For Railway.app setup:"
echo "   1. Go to railway.app"
echo "   2. Connect GitHub"
echo "   3. Select your repo"
echo "   4. Auto-deploys on every git push!"