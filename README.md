<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter Notebook" />
  <img src="https://img.shields.io/badge/Node.js-339933?style=for-the-badge&logo=nodedotjs&logoColor=white" alt="Node.js" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/SciPy-8CAAEE?style=for-the-badge&logo=scipy&logoColor=white" alt="SciPy" />
</div>

<h1 align="center">🌞 Perovskite Solar Cell (PSC) Simulation & Analysis 🔬</h1>

<p align="center">
  <b>Comprehensive analysis and visualization of 1D solar cell simulations (SCAPS-1D) focusing on Perovskite Solar Cells (PSCs) and the effect of different parameters on device performance.</b>
</p>

---

## 📖 Overview

This repository contains my final year project work dedicated to simulating, analyzing, and visualizing the electrical and optical properties of **Perovskite Solar Cells (PSCs)**. By employing SCAPS-1D simulation data, this project deeply investigates how structural changes, varying physical parameters, and environmental factors influence the overall Power Conversion Efficiency (PCE) and stability of the solar cells.

The project utilizes a robust Python data science stack to process simulation outputs and generates publication-ready visualizations, alongside an automated **Node.js report generation tool** to compile and document findings programmatically.

## 🎯 Key Objectives

| Objective | Description |
| :--- | :--- |
| 🌑 **Dark I-V Analysis** | Investigating the dark current-voltage characteristics to understand recombination mechanisms and shunt/series resistance effects. |
| ⚡ **J-V Plotting** | Plotting and extracting key performance metrics ($J_{sc}$, $V_{oc}$, FF, PCE) from simulated illuminated J-V curves. |
| 📏 **Thickness Optimization** | Analyzing the impact of Perovskite Active Layer (PAL), Electron Transport Layer (ETL), and Hole Transport Layer (HTL) thicknesses. |
| 🌡️ **Temperature Stability** | Simulating and visualizing how varying operating temperatures affect solar cell performance metrics. |
| 🌈 **Quantum Efficiency (QE)** | Examining the internal and external quantum efficiency across the light spectrum. |

## 📂 Repository Structure

```text
📁 Final Year Project
│
├── 📁 ETL HTL Sweep                  # Analysis of varying ETL & HTL properties
│   └── 📁 notebook                   # Jupyter notebooks for sweep analysis
│
├── 📁 IV Thickness                   # Impact of layer thickness on I-V curves
│   └── 📁 notebooks                  # plot_IV_thickness.ipynb
│
├── 📁 J-V Plot                       # Illuminated Current Density-Voltage analysis
│   └── 📁 notebook                   # JV_Plot_Final.ipynb, J_V_Plot_Code.ipynb
│
├── 📁 PAL vs PCE Thickness           # Contour analysis: Perovskite layer thickness vs PCE
│   └── 📁 notebooks                  # PAL_Thickness_vs_PCE_Contour_Analysis.ipynb
│
├── 📁 PSC_Dark_IV_Analysis_Project   # Deep dive into dark current-voltage behavior
│   └── 📁 notebooks                  # plot_fixed_graphs.ipynb
│
├── 📁 QE                             # Quantum Efficiency analysis
│   └── 📁 notebook                   # QE_Plot.ipynb
│
├── 📁 Temperature Sweep              # Performance degradation/variation with temperature
│   └── 📁 notebook                   # SCAPS_Temperature_Sweep_Analysis.ipynb
│
└── 📁 Others                         # Report generation scripts (generate_report.js) and related documents
```

> [!NOTE]
> Each sub-directory generally contains its own `data/` (for raw SCAPS outputs) and `graphs/` (for exported figures) folders. Generated documents (PDFs, Word reports) and Node modules are ignored via `.gitignore` to keep the repository clean.

## 🛠️ Technologies & Libraries

*   **Languages:** Python 3.x, JavaScript (Node.js)
*   **Environment:** Jupyter Notebooks
*   **Data Science Core:** `numpy`, `pandas`, `scipy`
*   **Data Visualization:** `matplotlib`, `seaborn`
*   **Report Generation:** `docx` (Node.js)

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Asphane/perovskite-cell-scaps-simulation-analysis.git
cd perovskite-cell-scaps-simulation-analysis
```

### 2. Python Environment (Data Analysis)
It is recommended to use a virtual environment for the Python dependencies.
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r Others/req.txt 
```

### 3. Node.js Environment (Automated Reports)
To utilize the automated report generation tools:
```bash
cd Others
npm install
node generate_report.js
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook
```
*Navigate to the respective directory and open the desired `.ipynb` file to view the analysis.*

## 🤖 Machine Learning Surrogate & Optimization Dashboard

To move beyond static data visualization, this project includes a highly advanced, AI-driven web application built with **Streamlit**. Traditional SCAPS-1D physical simulations can take minutes or hours when running large sweeps. This dashboard solves that by using trained machine learning models to provide **instantaneous** feedback and design optimization.

### 🧠 Core AI Models
- **Random Forest Regressor**: Trained on a dataset of 1,181 SCAPS-1D simulations. It predicts the summary metrics ($PCE$, $V_{oc}$, $J_{sc}$, $FF$) with over **98% accuracy** ($R^2$).
- **PyTorch Multi-Layer Perceptron (ANN)**: A custom neural network that takes the physical parameters and a voltage point as input and predicts the corresponding current density. This allows the app to generate the **entire continuous J-V curve** dynamically in real-time.

### 🔬 Physics Context & Parameter Significance
The dashboard allows manipulation of critical physical parameters that govern solar cell physics:
- **Operating Temperature ($T$)**: Higher temperatures typically increase reverse saturation current and reduce $V_{oc}$ due to increased intrinsic carrier concentration.
- **Thickness ($d$)**: Optimizing thickness is a trade-off between light absorption (thicker is better) and charge carrier collection (thinner reduces bulk recombination).
- **Defect Density ($N_t$)**: Higher defect densities introduce Shockley-Read-Hall (SRH) recombination centers, drastically reducing carrier lifetime and overall efficiency.
- **Acceptor Density ($N_a$)**: Influences the built-in potential and electric field distribution across the perovskite layer.

### 🏋️‍♂️ Model Training & Dataset
- **Dataset**: The models were trained on data extracted from **1,181 SCAPS-1D simulations** with randomized parameters within realistic physical bounds.
- **Feature Engineering**: Input features were standardized using `StandardScaler` to ensure stable gradients during PyTorch training. Logarithmic transformation was applied to density parameters to handle the wide dynamic range ($10^{13}$ to $10^{17} \text{ cm}^{-3}$).
- **PyTorch Training**: The MLP was trained using Mean Squared Error (MSE) loss and the Adam optimizer to learn the continuous mapping $f(\text{Parameters}, \text{Voltage}) \to \text{Current Density}$.

### 🌟 Key Features
1. **Interactive Device Design**: Adjust operating temperature and layer properties (thickness, defect density, acceptor density) using intuitive sliders. Density sliders use logarithmic scaling for easy handling of wide ranges.
2. **Instant Performance Prediction**: See the predicted Power Conversion Efficiency (PCE), Open-Circuit Voltage ($V_{oc}$), Short-Circuit Current ($J_{sc}$), and Fill Factor (FF) update in real-time as you move the sliders.
3. **Dynamic J-V Curve Visualization**: The PyTorch model generates the curve shape, providing insight into the cell's behavior under load.
4. **Radar Balance Chart**: Visualizes how well-rounded the design is across the 4 key metrics.
5. **Feature Importance**: See which physical parameters have the most impact on efficiency, derived directly from the Random Forest model.
6. **Inverse Design Optimizer**: Don't guess the best parameters. Click the "Find Optimal Design" button to run a **Genetic Algorithm** (`scipy.optimize.differential_evolution`) that searches the multi-dimensional space to find the global maximum efficiency configuration.
7. **Sensitivity Analysis**: Select any parameter to see a line plot of how efficiency varies across its entire range, helping identify sweet spots or degradation trends.

### 🚀 How to Run
```bash
.venv/bin/streamlit run Machine_Learning_Dashboard/app.py
```

### 🖼️ App Screenshots
*(Note: Please replace the files in `Machine_Learning_Dashboard/assets/` with your correct screenshots!)*

<div align="center">
  <img src="./Images/app-1.png" alt="Dashboard Header" width="750"/>
  <br><br>
  <img src="./Images/app-2.png" alt="JV Curve" width="750"/>
  <br><br>
  <img src="./Images/app-3.png" alt="Optimizer" width="750"/>
</div>

## 📊 Visualizations & Outputs

<table align="center">
  <tr>
    <td align="center"><b>Current Density-Voltage (J-V)</b></td>
    <td align="center"><b>Efficiency Contour Analysis</b></td>
  </tr>
  <tr>
    <td align="center"><img src="./J-V%20Plot/graph/JV_final_annotated.png?v=2" alt="J-V Curve" width="350"/></td>
    <td align="center"><img src="./PAL%20vs%20PCE%20Thickness/graphs/PSC_PAL_thickness_contour.png?v=2" alt="Thickness Contour" width="350"/></td>
  </tr>
  <tr>
    <td><i>Illuminated J-V curve simulation highlighting key parameters ($V_{oc}$, $J_{sc}$).</i></td>
    <td><i>Contour map showing optimal Perovskite Active Layer (PAL) thickness.</i></td>
  </tr>
  <tr>
    <td align="center"><b>Quantum Efficiency (QE)</b></td>
    <td align="center"><b>Temperature Impact</b></td>
  </tr>
  <tr>
    <td align="center"><img src="./QE/graph/QE_dual_axis.png?v=2" alt="Quantum Efficiency" width="350"/></td>
    <td align="center"><img src="./Temperature%20Sweep/graph/temperature_params.png?v=2" alt="Temperature Impact" width="350"/></td>
  </tr>
  <tr>
    <td><i>Internal and external quantum efficiency across the light spectrum.</i></td>
    <td><i>Variation of core performance metrics under differing temperatures.</i></td>
  </tr>
  <tr>
    <td align="center" colspan="2"><b>Dark I-V Analysis</b></td>
  </tr>
  <tr>
    <td colspan="2" align="center"><img src="./PSC_Dark_IV_Analysis_Project/graphs/IV_log.png?v=2" alt="Dark IV" width="350"/></td>
  </tr>
  <tr>
    <td colspan="2" align="center"><i>Logarithmic plot of dark current-voltage to study recombination mechanisms.</i></td>
  </tr>
</table>

## 🎓 Academic Context

This repository represents the practical implementation and data analysis portion of my Final Year Project. The theoretical background, methodology, and full conclusions are synthesized using the automated tools and documented in the generated reports.

---
<div align="center">
  <i>Developed by Bisakh Patra</i>
</div>
