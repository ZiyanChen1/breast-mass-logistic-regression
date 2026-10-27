# Breast Mass Classification: Logistic Regression and Decision Thresholds

## Overview

- **Question:** Can mean radius and mean texture separate malignant from benign breast mass samples, and how does the decision threshold change the model's mistakes?
- **Data:** Wisconsin Diagnostic Breast Cancer dataset (Wolberg, Street, and Mangasarian, 1995; UCI), bundled with scikit-learn, so no download is needed. Positive class: **1 = malignant**, 0 = benign.
- **Methods:** Logistic regression written in NumPy (gradient descent: learning rate 0.1, 20,000 epochs, no regularization), compared with scikit-learn's `LogisticRegression` and a majority-class baseline. The threshold was chosen on validation data and tested once on the test set.
- **Findings:** The NumPy model matched scikit-learn. At threshold 0.50, it caught 31 of 42 malignant test samples, while the baseline caught none. The validation-selected threshold (0.60) did worse on the test set.
- **Limitations:** Only 2 features, a small historical sample, and no outside validation.
- **Scope:** Non-clinical analysis of a public dataset. Not a diagnostic tool.

📄 **Full report:** [memo.md](memo.md)

---

## Contents

1. [Project Files](#project-files)
2. [How to Run This Project](#how-to-run-this-project)
3. [What Success Looks Like](#what-success-looks-like)
4. [Troubleshooting](#troubleshooting)

---

## Project Files

```
breast-mass-logistic-regression/
├── logistic_regression_analysis.py
├── memo.md
├── requirements.txt
├── .gitignore
├── README.md
└── reports/
    ├── threshold_results.csv
    ├── assignment_results.csv
    └── training_loss.png
```

---

## How to Run This Project

These steps assume you have never used a terminal, Python, or Git. Follow them in order. The first time takes about 15–20 minutes, mostly for installing software.

**Tip:** You can copy each command from this page and paste it into the terminal. Paste with **Ctrl + V** (Windows) or **Command + V** (Mac), then press **Enter**.

### Step 1: Check which computer you have

- **Windows:** the bottom of your screen has a **Start** button (the Windows logo).
- **Mac:** the top-left corner of your screen has an **Apple logo** ().

Some steps below have separate instructions. Follow only the ones for your computer.

### Step 2: Install Python

You need **Python 3.10 or newer**.

1. Go to **https://www.python.org/downloads/**
2. Click the yellow **Download Python** button.
3. Open the downloaded file (usually in your **Downloads** folder).

**Windows:** If the installer shows a checkbox **"Add python.exe to PATH"**, check it. Then click **Install Now** and follow the prompts.

**Mac:** Click **Continue**, **Agree**, and **Install**. Enter your computer password if asked.

### Step 3: Install Git

Git downloads this project to your computer.

**Windows:**
1. Go to **https://git-scm.com/install/windows** and click **Click here to download**.
2. Open the downloaded file and click **Next** on every screen (the default options are fine), then **Install**.

**Mac:** Git is usually installed already. You will check in Step 5. If a pop-up asks you to install **"command line developer tools"**, click **Install** and wait for it to finish.

### Step 4: Open the terminal

The terminal is a window where you type commands.

- **Windows:** Click **Start**, type **PowerShell**, and press **Enter**.
- **Mac:** Press **Command + Space**, type **Terminal**, and press **Enter**.

Keep this window open for all the remaining steps.

### Step 5: Check that Python and Git are installed

**Windows:**
```powershell
python --version
git --version
```

**Mac:**
```bash
python3 --version
git --version
```

You should see something like `Python 3.14.4` and `git version 2.x.x`. Any Python version **3.10 or higher** works. If you see an error, see [Troubleshooting](#troubleshooting).

### Step 6: Download the project

These commands go to your home folder, download the project, and move into the project folder.

**Windows:**
```powershell
cd $HOME
git clone https://github.com/ZiyanChen1/breast-mass-logistic-regression.git
cd breast-mass-logistic-regression
```

**Mac:**
```bash
cd ~
git clone https://github.com/ZiyanChen1/breast-mass-logistic-regression.git
cd breast-mass-logistic-regression
```

**Check you are in the right folder.** Type `dir` (Windows) or `ls` (Mac) and press Enter. You should see `logistic_regression_analysis.py` in the list.

The project is now saved at `C:\Users\<your user name>\breast-mass-logistic-regression` (Windows) or `/Users/<your user name>/breast-mass-logistic-regression` (Mac). `<your user name>` is the name of your account on the computer.

**All remaining commands must be run inside this folder.** If you close the terminal and come back later, open a new terminal and run `cd $HOME\breast-mass-logistic-regression` (Windows) or `cd ~/breast-mass-logistic-regression` (Mac) first.

### Step 7: Create and turn on a virtual environment

A virtual environment is a private space for this project's Python packages, so they do not affect anything else on your computer.

**Windows:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If you see a red error saying **running scripts is disabled**, run this line, then run the `Activate.ps1` line again:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
This only affects the current PowerShell window. It does not change your computer's security settings permanently.

**Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Check it worked:** the start of the terminal line now shows **`(.venv)`**.

### Step 8: Install the required packages

**Windows:**
```powershell
python -m pip install -r requirements.txt
```

**Mac:**
```bash
python3 -m pip install -r requirements.txt
```

This installs NumPy, scikit-learn, and Matplotlib. It takes about 1–2 minutes and needs an internet connection. The dataset comes with scikit-learn, so nothing else needs to be downloaded.

### Step 9: Run the program

**Windows:**
```powershell
python logistic_regression_analysis.py
```

**Mac:**
```bash
python3 logistic_regression_analysis.py
```

The program takes about 5 seconds. Wait until it finishes.

When you are done, you can type `deactivate` to turn off the virtual environment.

---

## What Success Looks Like

### Printed output

The program prints six parts. Check these key results:

**Part 5: threshold selection (validation data)**
```
  threshold   TN   FP   FN   TP  accuracy  precision   recall       F1
       0.20   54   17    5   38    0.8070     0.6909   0.8837   0.7755
       0.30   61   10    7   36    0.8509     0.7826   0.8372   0.8090
       0.40   63    8    7   36    0.8684     0.8182   0.8372   0.8276
       0.50   65    6    9   34    0.8684     0.8500   0.7907   0.8193
       0.60   69    2   11   32    0.8860     0.9412   0.7442   0.8312  <- selected
       0.70   69    2   14   29    0.8596     0.9355   0.6744   0.7838
       0.80   71    0   16   27    0.8596     1.0000   0.6279   0.7714
  Selected threshold = 0.60 (highest validation F1; ties go to the smallest threshold)
```

**Final evaluation (test rows)**
```
  split      model    policy    thresh   TN   FP   FN   TP  accuracy  precision   recall       F1  log_loss
  test       baseline default     0.50   72    0   42    0    0.6316     0.0000   0.0000   0.0000    0.6581
  test       numpy    default     0.50   68    4   11   31    0.8684     0.8857   0.7381   0.8052    0.2557
  test       numpy    selected    0.60   69    3   14   28    0.8509     0.9032   0.6667   0.7671    0.2557
  test       sklearn  default     0.50   68    4   11   31    0.8684     0.8857   0.7381   0.8052    0.2557
  test       sklearn  selected    0.60   69    3   14   28    0.8509     0.9032   0.6667   0.7671    0.2557
```

The last lines confirm that three files were saved:
```
Part 6: exported files
  Saved: ...\reports\threshold_results.csv
  Saved: ...\reports\assignment_results.csv
  Saved: ...\reports\training_loss.png
```

**Acceptable differences:** The selected threshold (0.60), all TN/FP/FN/TP counts, and all metrics shown with four decimals should match exactly. Very small numbers written like `1.31e-09` or `-5.55e-17` may be slightly different on your computer. This is normal rounding and does not affect the results.

### Output files

The program saves three files in the `reports` folder. Running it again replaces them.

| File | Contents |
|---|---|
| `threshold_results.csv` | 7 rows: validation results for each threshold |
| `assignment_results.csv` | 10 rows: 5 model/threshold settings on validation and test data |
| `training_loss.png` | Chart of the NumPy model's training loss by epoch |

**To open the folder:**
- **Windows:** `explorer reports`
- **Mac:** `open reports`

Double-click a `.csv` file to open it in Excel or Numbers, or the `.png` file to view the chart.

### Tested environment

| Item | Version |
|---|---|
| Operating system | Windows 11 |
| Python | 3.14.4 |
| NumPy | 2.5.4 |
| scikit-learn | 1.9.1 |
| Matplotlib | 3.11.2 |

`requirements.txt` sets minimum versions, so newer versions should also work.

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| Windows: `'python' is not recognized`, or the Microsoft Store opens | Python is not installed, or not added to PATH | Try `py` instead of `python` in every command. If that also fails, reinstall Python (Step 2), close PowerShell, open a new one, and start again from Step 5. |
| Mac: `command not found: python` | Macs use `python3` | Type `python3` instead of `python`. |
| Mac: `python3 --version` shows 3.9 or older | The Mac's built-in Python is too old | Install Python from python.org (Step 2), then close Terminal, open a new one, and check again. |
| `'git' is not recognized` or `command not found: git` | Git is not installed | Complete Step 3, then close and reopen the terminal. |
| `fatal: destination path 'breast-mass-logistic-regression' already exists` | You already downloaded the project | Skip the `git clone` line and run only `cd breast-mass-logistic-regression`. |
| `running scripts is disabled on this system` | Windows blocks the activation script | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again (Step 7). |
| `ModuleNotFoundError: No module named 'sklearn'` (or `numpy`, `matplotlib`) | The virtual environment is not on, or packages are not installed | Make sure the line starts with `(.venv)`. If not, run the activate command from Step 7. Then run Step 8 again. |
| `can't open file ... No such file or directory` | The terminal is in the wrong folder | Go back into the project folder (see the end of Step 6), then run the program again. |
| `PermissionError: [Errno 13] Permission denied` | A results file is open in Excel, which locks it | Close the file in Excel (choose **Don't Save**) and run the program again. |

---

## Author

**Ziyan Chen** · M.S. in Business Analytics, University of the Pacific
