# Project Setup & Useful Commands

Reusable base library for my Python projects.  
It contains common functionality for fitting, smoothing and model-related utilities that can be shared across multiple applications.

---

## 1. Virtual Environment

### 1.1 Create / check / activate environment

Use the provided PowerShell script:

```powershell
. .\setup_env.ps1
````

This script will:

* Create the `.venv` virtual environment if it does not exist
* Activate the virtual environment
* Install / update the required packages
* Install / update the recommended extensions

### 1.2 Manually activate existing venv (if needed)

If you need to activate the environment manually:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## 2. Updating Requirements and VS Code Extensions

### 2.1 Update Python requirements file

From within the active virtual environment:

```powershell
python -m pip freeze > _requirements.txt
```

This writes all currently installed packages into `_requirements.txt`.

### 2.2 Export VS Code extensions

To save the list of installed VS Code extensions:

```powershell
code --list-extensions > _extensions.txt
```

This writes all installed extensions into `_extensions.txt`.

---

## 3. Git Configuration

Set your Git username and email (once per machine / repository):

```powershell
git config user.name "Git name"
git config user.email "email"
```

Replace `"Git name"` and `"email"` with your actual name and email address.

---

## 4. Running the Applications

All applications are started via Python’s module syntax from the repository root.


```powershell
python -m LabviewToPython.app
```
```powershell
python -m time_scan_app.main
```
```powershell
python -m ionplotter.app
```
```powershell
python -m stft_app.app
```
```powershell
python -m lab_discord_bot.main
```
```powershell
python -m Lab_apps.plot_bot.script
```
```powershell
python -m Lab_apps.scan_averaging.script
```
---

## 5. Notes

* Always run commands from the project root directory (where `setup_env.ps1` is located).
* Make sure the virtual environment is active before running any of the Python modules above.

```
