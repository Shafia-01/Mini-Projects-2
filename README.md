# Mini-Projects-2

A collection of Python projects showcasing machine learning, data analytics, and computational techniques.
## Overview

This repository brings together four Python-based mini-projects along with supporting presentations. The projects illustrate core concepts in:

- Machine learning and model building  
- Data preprocessing and transformation  
- Text analytics and regression techniques  
- Visualization of analytical results    

## Projects
Here are the main directories:

| Directory      | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| `Task 1`       | Neural Network Regression (NumPy from Scratch)                              |
| `Task 2`       | NLTK-Powered Text Analytics Web App                                         |
| `Task 3`       | High-Performance Time Series Transformation with NumPy & pandas             |
| `Task 4`       | Complex Data Munging & Statistical Modeling in pandas                       |
| `Presentation` | Contains mathematics presentations, output screenshots, and Task 4 summary. |

## Presentations

The Presentation/ folder includes:

- **Linear Algebra Presentation** – definitions, theorems, and worked examples.  
- **Calculus Presentation** – core concepts, illustrative examples, and solved problems.  
- **Screenshots** – outputs from different task runs for reference.  
- **Task 4 Project Presentation** – summarizing data challenges, applied solutions, and final model results.

These materials provide theoretical background and practical demonstrations.

## Features
- Projects implemented primarily in **Python**, often using **Jupyter Notebooks**. 
- Data analysis / visualization components  
- Use of computational or ML techniques (depending on the project)  
- Clean folder structure, one project per folder  

## Environment Setup

To run any of the projects locally, follow these steps:

1. **Install Python**  
Make sure you have Python 3.7+ installed (or the version required by the notebooks).  

2. **Create a virtual environment** (recommended)  
```bash
python3 -m venv venv
# or on Windows
python -m venv venv
```

3. Activate the virtual environment
- On macOS / Linux:
```bash
source venv/bin/activate
```
- On Windows (PowerShell):
```bash
.\venv\Scripts\Activate
```

4. Install dependencies
If there is a requirements.txt, run:
```bash
pip install -r requirements.txt
```
If not, you’ll need to install common packages manually, for example:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

5. Ensure Jupyter Notebook / JupyterLab is installed if you want to run .ipynb files.

## Installation & Usage
Here’s how you can get started:
1. Clone the repository
```bash
git clone https://github.com/Shafia-01/Mini-Projects-2.git
cd Mini-Projects-2
```

2. Setup environment (see previous section)

3. Run or open a project
- For notebooks:
```bash
jupyter notebook
```
then open the relevant .ipynb file under a Task folder.

- For Python scripts (if any):
```bash
python path/to/script.py
```

4. Explore data / outputs
- Look for input data files (if included)
- Visualizations or results might be in notebook output cells or separate result files

5. Presentations / write-ups
- The Presentation folder contains slides / documents summarising the work. You can view these for process, results, insights.