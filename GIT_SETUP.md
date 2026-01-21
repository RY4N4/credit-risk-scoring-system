# 🚀 Git Setup Guide - Add Project to GitHub

Complete step-by-step guide to add your Credit Risk Scoring System to Git/GitHub.

---

## 📋 Prerequisites

- Git installed on your Mac: `git --version`
- GitHub account: https://github.com

---

## 🎯 Step-by-Step Instructions

### Step 1: Initialize Git Repository

Open Terminal in your project directory and run:

```bash
cd /Users/cliveleealves/Desktop/CreditRisk

# Initialize git repository
git init

# Verify initialization
git status
```

**Expected output:**
```
Initialized empty Git repository in /Users/cliveleealves/Desktop/CreditRisk/.git/
```

---

### Step 2: Configure Git (First Time Only)

If you haven't configured git before:

```bash
# Set your name
git config --global user.name "Your Name"

# Set your email (use GitHub email)
git config --global user.email "your.email@example.com"

# Verify configuration
git config --global --list
```

---

### Step 3: Review .gitignore (Already Created)

The project already has a `.gitignore` file that excludes:
- `__pycache__/` - Python cache files
- `*.pyc` - Compiled Python files
- `models/` - Large model files (optional: you can include these)
- `data/raw/` - Large datasets
- `.env` - Environment variables
- Virtual environments

**To include trained models in Git (recommended for portfolio):**

```bash
# Edit .gitignore and comment out the models/ line
nano .gitignore
# Find: models/
# Change to: # models/
# Save: Ctrl+X, Y, Enter
```

---

### Step 4: Add All Files to Git

```bash
# Add all files
git add .

# Check what will be committed
git status
```

**Expected output:**
```
On branch main

Changes to be committed:
  new file:   README.md
  new file:   QUICKSTART.md
  new file:   requirements.txt
  new file:   src/data_processing.py
  new file:   src/train.py
  ... (and many more files)
```

---

### Step 5: Create Initial Commit

```bash
# Commit with descriptive message
git commit -m "Initial commit: Credit Risk Scoring System with XGBoost, FastAPI, and Streamlit UI"

# Verify commit
git log --oneline
```

**Expected output:**
```
abc1234 Initial commit: Credit Risk Scoring System with XGBoost, FastAPI, and Streamlit UI
```

---

### Step 6: Create GitHub Repository

**Option A: Via GitHub Website (Recommended)**

1. Go to https://github.com/new
2. Repository name: `credit-risk-scoring-system`
3. Description: `ML system for loan default prediction with XGBoost, FastAPI, and Streamlit UI`
4. Choose **Public** (for portfolio) or **Private**
5. **DON'T** check "Initialize with README" (you already have one)
6. Click **"Create repository"**

**Option B: Via GitHub CLI (if installed)**

```bash
# Install GitHub CLI (if needed)
brew install gh

# Login to GitHub
gh auth login

# Create repository
gh repo create credit-risk-scoring-system --public --source=. --remote=origin

# Push code
git push -u origin main
```

---

### Step 7: Connect Local Repo to GitHub

Copy the commands from GitHub's "Quick setup" page, or use these:

```bash
# Add remote repository (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/credit-risk-scoring-system.git

# Verify remote
git remote -v
```

**Expected output:**
```
origin  https://github.com/YOUR_USERNAME/credit-risk-scoring-system.git (fetch)
origin  https://github.com/YOUR_USERNAME/credit-risk-scoring-system.git (push)
```

---

### Step 8: Push to GitHub

```bash
# Rename branch to main (if needed)
git branch -M main

# Push code to GitHub
git push -u origin main
```

**Expected output:**
```
Enumerating objects: 50, done.
Counting objects: 100% (50/50), done.
...
To https://github.com/YOUR_USERNAME/credit-risk-scoring-system.git
 * [new branch]      main -> main
Branch 'main' set up to track remote branch 'main' from 'origin'.
```

---

### Step 9: Verify on GitHub

1. Go to: `https://github.com/YOUR_USERNAME/credit-risk-scoring-system`
2. You should see:
   - All your files
   - README.md rendered nicely
   - Green "Code" button
   - Commit history

---

## 🎨 Optional: Add GitHub Topics

Make your repo more discoverable:

1. On GitHub repo page, click ⚙️ (Settings icon) next to "About"
2. Add topics:
   - `machine-learning`
   - `credit-risk`
   - `xgboost`
   - `fastapi`
   - `streamlit`
   - `fintech`
   - `loan-prediction`
   - `python`
   - `data-science`
3. Save changes

---

## 📝 Future Git Workflow

After initial setup, use this workflow for updates:

### Making Changes

```bash
# 1. Check status
git status

# 2. Add changed files
git add .
# Or add specific files
git add src/train.py README.md

# 3. Commit with message
git commit -m "Add feature: improved threshold optimization"

# 4. Push to GitHub
git push
```

### Common Git Commands

```bash
# View commit history
git log --oneline --graph

# View changes before committing
git diff

# Undo unstaged changes
git restore filename.py

# View remote URL
git remote -v

# Pull latest changes (if working across devices)
git pull

# Create new branch
git checkout -b feature-name

# Switch branches
git checkout main
```

---

## 🔒 Handling Sensitive Data

**Never commit:**
- API keys
- Passwords
- `.env` files with secrets
- Personal data

**If you accidentally committed secrets:**

```bash
# Remove from last commit (before pushing)
git reset --soft HEAD~1
git restore --staged .env
git commit -m "Your message"

# If already pushed, rotate the credentials immediately!
```

---

## 📊 Adding Large Files (Models)

If your model files are >100MB, use Git LFS:

```bash
# Install Git LFS
brew install git-lfs

# Initialize
git lfs install

# Track large files
git lfs track "models/*.pkl"

# Commit .gitattributes
git add .gitattributes
git commit -m "Add Git LFS tracking"

# Add models
git add models/
git commit -m "Add trained models"
git push
```

---

## 🌟 Make Your Repo Stand Out

### Add Badges to README

Already included in your README:
- ✅ Python version
- ✅ Framework badges (FastAPI, Streamlit, XGBoost)
- ✅ License badge

### Add Screenshots

```bash
# Take screenshots of:
# 1. Streamlit UI
# 2. API docs (Swagger)
# 3. Model visualizations

# Create screenshots folder
mkdir -p screenshots

# Add to README
# ![Streamlit UI](screenshots/frontend.png)
```

### Create GitHub Profile README

Make your profile stand out:

1. Create repo: `https://github.com/YOUR_USERNAME/YOUR_USERNAME`
2. Add `README.md` with:
   - Your intro
   - Featured projects (link to this one!)
   - Skills & technologies
   - Contact info

---

## 🐛 Troubleshooting

### Problem: "remote origin already exists"

```bash
# Remove old remote
git remote remove origin

# Add correct remote
git remote add origin https://github.com/YOUR_USERNAME/credit-risk-scoring-system.git
```

### Problem: "failed to push some refs"

```bash
# Pull first (if repo has files)
git pull origin main --allow-unrelated-histories

# Then push
git push -u origin main
```

### Problem: "Permission denied (publickey)"

```bash
# Use HTTPS instead of SSH
git remote set-url origin https://github.com/YOUR_USERNAME/credit-risk-scoring-system.git

# Or set up SSH key: https://docs.github.com/en/authentication/connecting-to-github-with-ssh
```

### Problem: Large file rejected (>100MB)

```bash
# Use Git LFS (see above)
# Or exclude from git:
echo "models/*.pkl" >> .gitignore
git rm --cached models/*.pkl
git commit -m "Remove large files"
```

---

## ✅ Checklist

Before pushing to GitHub, verify:

- [ ] `.gitignore` is configured correctly
- [ ] No sensitive data (API keys, passwords) in code
- [ ] README.md is complete and formatted
- [ ] Requirements.txt has all dependencies
- [ ] Code is well-commented
- [ ] All scripts run without errors
- [ ] Models are trained and saved (if including)
- [ ] Documentation is up-to-date

---

## 🎓 Next Steps After Pushing

1. **Add GitHub Actions**: Automate testing/deployment
2. **Enable GitHub Pages**: Host documentation
3. **Add Contributing Guide**: `CONTRIBUTING.md`
4. **Add License**: `LICENSE` file (MIT recommended)
5. **Star Your Repo**: Helps with visibility
6. **Share on LinkedIn**: Showcase your work!

---

## 📚 Resources

- **Git Documentation**: https://git-scm.com/doc
- **GitHub Guides**: https://guides.github.com
- **Git Cheat Sheet**: https://education.github.com/git-cheat-sheet-education.pdf
- **GitHub Actions**: https://docs.github.com/en/actions

---

## 💡 Pro Tips

1. **Commit often**: Small, frequent commits are better
2. **Write clear messages**: Describe what AND why
3. **Use branches**: For features, keep main clean
4. **Review before push**: `git diff`, `git status`
5. **Pull before push**: Avoid conflicts
6. **Tag releases**: `git tag v1.0.0` for versions

---

**Need help? Check your git status: `git status`**
