![Alt text](./SJSU_Primary_mark_Web.png)

# Git Workflow

## **Branch Naming Format**

main ← stable, working code only
dev ← integration branch, merge here first
rohit  
dania  
neha

---

## **INITIAL SETUP (Do This Once)**

### **Step 1: Navigate to where you want the project**

Open VS Code terminal or any terminal, then navigate to your desired folder:

```
cd ~/<folder-name>
```

- Replace `~/<folder-name>` with wherever you want to store the project.
- Common locations: `~/repos` or `~/projects`

### **Step 2: Clone the repository**

```
git clone https://github.com/yourusername/BME-133.git
cd BME-133
```

- `git clone` copies the GitHub repo to your computer.
- `cd` changes into the repo folder.

**✅ Setup complete!** You only need to do this once. Now follow the "Regular Workflow" below every time you work on the project.

---

## **REGULAR WORKFLOW (Do This Every Time You Work)**

### **Step 1: Navigate to your project folder**

```bash
cd path/to/BME-133
```

- Replace `path/to/BME-133` with the actual path to your project folder.
- Example: `cd ~/repos/BME-133`

---

### **Step 2: Switch to main and pull latest changes**

```bash
git checkout main
git pull origin main
```

- This ensures you have the latest code from your team before starting new work.
- **Always do this first!**

---

### **Step 3: Create a new branch**

```bash
git checkout -b <branch-name>
```

- Replace `<branch-name>` with your name
- This creates a new branch and switches to it.

Check your branch:

```bash
git branch
```

- `*` indicates your current branch.

---

### **Step 4: Work on your branch**

- Make your changes.

---

### **Step 5: Stage your changes**

```bash
git add .
```

- The command: `git add .` stages all changed files.

---

### **Step 6: Commit your changes**

```bash
git commit -m "Completed xyz"
```

- Write a clear commit message so others know what you changed.
- Commit frequently as you make progress!

---

### **Step 7: Push your branch to GitHub**

**First time pushing this branch:**

```bash
git push origin <branch-name>
```

**Subsequent pushes (after the first):**

```bash
git push
```

- This uploads your work to GitHub so others can see it.

---

### **Step 8: Merge your branch into main (when exercise is complete)**

Once your work is ready:

```bash
git checkout main
git pull origin main
git merge <branch-name>
git push origin main
```

- This merges your branch into the main branch.
- Always pull before merging to avoid conflicts!

---

### **Step 9: Keep working or start a new exercise**

**To start a new branch:**

- Go back to Step 2 (pull main) and create a new branch.

**To continue working on your current branch:**

```bash
git checkout <your-branch-name>
git pull origin main
```

- This updates your branch with any changes others have made to main.

---

## **Quick Reference: Git Commands**

| Command                         | Description                                                                   |
| ------------------------------- | ----------------------------------------------------------------------------- |
| `git clone <repo_url>`          | Make a local copy of the repository from GitHub (one-time setup).             |
| `git status`                    | Check which files have changed, which are staged, and which branch you're on. |
| `git branch`                    | See all local branches. `*` indicates your current branch.                    |
| `git checkout -b <branch_name>` | Create a new branch and switch to it.                                         |
| `git switch <branch_name>`      | Switch to an existing branch.                                                 |
| `git add <file>`                | Stage a file for commit.                                                      |
| `git add .`                     | Stage ALL changed files in the current directory and subdirectories.          |
| `git commit -m "message"`       | Commit staged files with a descriptive message.                               |
| `git push origin <branch_name>` | Push your branch to GitHub (first time).                                      |
| `git push`                      | Push subsequent commits (after upstream is set).                              |
| `git pull origin main`          | Pull the latest changes from the main branch.                                 |
| `git pull`                      | Pull subsequent changes (after upstream is set).                              |
| `git merge <branch_name>`       | Merge another branch into your current branch.                                |
| `git rm <file>`                 | Remove file from your working directory.                                      |
| `git log`                       | See a history of commits for the current branch.                              |
