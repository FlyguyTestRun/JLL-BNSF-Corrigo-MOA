# Syncing Your Local VS Code Folder with GitHub

## Your Setup
- **Local folder**: `C:\Shaw\GitShawHuB\Shaw\JLL-BNSF-Corrigo-MOA`
- **GitHub repo**: `https://github.com/FlyguyTestRun/JLL-BNSF-Corrigo-MOA`

## Initial Setup (Run Once)

Open a terminal in VS Code (Ctrl+`) and run these commands:

### Step 1: Initialize Git (if not already a git repo)
```bash
cd "C:\Shaw\GitShawHuB\Shaw\JLL-BNSF-Corrigo-MOA"
git init
```

### Step 2: Connect to GitHub
```bash
git remote add origin https://github.com/FlyguyTestRun/JLL-BNSF-Corrigo-MOA.git
```

If the remote already exists, update it:
```bash
git remote set-url origin https://github.com/FlyguyTestRun/JLL-BNSF-Corrigo-MOA.git
```

### Step 3: Pull existing content from GitHub
```bash
git fetch origin
git checkout main
git pull origin main
```

### Step 4: If you have existing local files to keep
If your local folder already has files you want to keep AND the GitHub repo also has files:
```bash
git fetch origin
git checkout -b main origin/main
```
Then copy your local files into the folder, stage and commit:
```bash
git add .
git commit -m "Add existing local project files"
git push -u origin main
```

## Daily Workflow

### Pull latest changes (start of session)
```bash
git pull origin main
```

### Push your changes
```bash
git add .
git commit -m "Describe your changes here"
git push -u origin main
```

### If conflicts arise
```bash
git pull origin main
# Resolve any conflicts in VS Code (it highlights them)
git add .
git commit -m "Resolve merge conflicts"
git push -u origin main
```

## VS Code Git Integration
Once connected, VS Code's built-in Source Control panel (Ctrl+Shift+G) will:
- Show changed files
- Let you stage, commit, and push with clicks
- Show branch status in the bottom-left corner
- Auto-detect the GitHub remote

## Recommended VS Code Extensions
- **GitHub Pull Requests and Issues** - manage PRs directly in VS Code
- **GitLens** - enhanced git history and blame annotations
