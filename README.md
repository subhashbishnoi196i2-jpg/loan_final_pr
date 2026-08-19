# 📊 Loan Default Risk – Probability & Statistics Project

A comprehensive Python-based exploratory data analysis, statistical modeling, and probability investigation evaluating **Loan Default Risk** across 5,000 customer loan applications.

---

## 📌 Project Overview

Understanding credit risk and predicting loan default probability is critical for financial institutions to minimize non-performing assets (NPAs) while expanding credit access. 

This project bridges **theoretical mathematical concepts** (probability theory, linear algebra, statistical distributions) with **practical Python data analytics** to assess customer risk profiles, calculate conditional probabilities, evaluate distribution shapes, and analyze vector relationships in customer feature space.

---

## 📁 Repository Structure

```text
Loan_Default_Probability_Project/
├── Loan_Default_Probability_Project.ipynb   # Main Jupyter notebook with EDA, math & code
├── Loan_Default_Project_Report.docx         # Executive summary report (Word format)
├── loan_applications.csv                    # Dataset containing 5,000 applicant records
└── README.md                                # Project documentation
```

---

## 📊 Dataset Schema (`loan_applications.csv`)

The dataset consists of **5,000 synthetic/sample customer loan applications** across 7 key variables:

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `Customer_ID` | String | Unique identifier for each application (e.g., `CUST00001`) |
| `Age` | Integer | Age of applicant in years |
| `Income` | Numeric | Annual income in USD (\$) |
| `Loan_Amount` | Numeric | Total requested loan amount in USD (\$) |
| `Credit_Score` | Integer | FICO credit score ranging from 300 to 850 |
| `Loan_Term` | Integer | Loan repayment duration in months (e.g., 24, 48, 72) |
| `Default_Status` | Categorical | Binary outcome (`Yes` = Defaulted, `No` = Non-Defaulted) |

---

## 🧠 Key Theoretical Concepts Covered

1. **Central Tendency & Dispersion**: Understanding Mean vs. Median in skewed data, Standard Deviation vs. Variance as risk metrics.
2. **Random Variables**: Modeling continuous features (Income, Loan Amount) and discrete outcomes (Default Status).
3. **Conditional Probability & Bayes' Theorem**: Computing $P(\text{Default} \mid \text{Credit Score} < 600)$ and updating prior risk beliefs based on customer segmentation.
4. **Empirical vs. Theoretical Distributions**: Fitting Normal (Gaussian) density curves to credit scores, assessing skewness, kurtosis, and validating normality with Quantile-Quantile (Q-Q) plots.
5. **Linear Algebra & Feature Space Geometry**: Representing applicants as multi-dimensional vectors, computing vector norms, dot products, and cosine angular similarity $\theta$.
6. **Eigenvalues & Eigenvectors**: Conceptual grounding for Dimensionality Reduction via Principal Component Analysis (PCA).

---

## 📈 Key Findings & Statistical Summary

### 1. Overall & Conditional Default Probabilities
* **Overall Default Probability $P(\text{Default})$**: **24.62%** (~1 in 4 applications end in default).
* **High-Risk Segment $P(\text{Default} \mid \text{Credit Score} < 600)$**: **35.71%** (a 45% relative increase in default risk compared to baseline).

### 2. Credit Score Tier Breakdown (Contingency Analysis)

| Credit Score Tier | Non-Default (`No`) | Default (`Yes`) | Total Applicants | Default Rate ($P$) |
| :---: | :---: | :---: | :---: | :---: |
| **< 600 (Poor)** | 468 | 260 | 728 | **35.71%** |
| **600 – 699 (Fair)** | 1,688 | 600 | 2,288 | **26.22%** |
| **700 – 799 (Good)** | 1,351 | 341 | 1,692 | **20.15%** |
| **800 – 850 (Exceptional)** | 262 | 30 | 292 | **10.27%** |
| **Total** | **3,769** | **1,231** | **5,000** | **24.62%** |

### 3. Distribution Metrics

| Metric | Income (\$) | Loan Amount (\$) | Credit Score |
| :--- | :---: | :---: | :---: |
| **Mean** | \$99,495.31 | \$52,492.50 | 679.52 |
| **Median** | \$99,872.00 | \$52,500.00 | 680.00 |
| **Std. Deviation** | \$28,841.12 | \$27,764.14 | 74.82 |
| **Variance** | 831,809,923 | 770,847,682 | 5,598 |
| **Skewness** | -0.0012 | **0.0039** *(Symmetric)* | 0.015 |
| **Excess Kurtosis** | -1.198 | **-1.2231** *(Platykurtic)* | -0.210 |

### 4. Vector Geometry in Feature Space
Evaluating applicant feature vectors $\mathbf{v_1} = [162972, 46792]$ and $\mathbf{v_2} = [145992, 85210]$ (Income, Loan Amount):
* **Dot Product ($\mathbf{v_1} \cdot \mathbf{v_2}$)**: $27,779,754,544$
* **Vector Norm $\|\mathbf{v_1}\|$**: $169,540.85$
* **Vector Norm $\|\mathbf{v_2}\|$**: $169,045.18$
* **Angular Separation ($\theta$)**: **14.25°** (High geometric similarity in borrowing ratio).

---

## 🛠️ Setup & Requirements

### Prerequisites
* Python 3.8 or higher
* Jupyter Notebook or VS Code Jupyter extension

### Required Python Libraries
Install the required packages using `pip`:

```bash
pip install pandas numpy matplotlib scipy
```

---

## 🚀 How to Run the Analysis

1. **Clone or Download the Repository**:
   Ensure `Loan_Default_Probability_Project.ipynb` and `loan_applications.csv` are in the same directory.

2. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook Loan_Default_Probability_Project.ipynb
   ```

3. **Execute Notebook Cells**:
   Run all cells sequentially (`Cell -> Run All` or `Shift + Enter`) to reproduce all statistical metrics, contingency tables, and visualization plots (Histograms with Gaussian overlays and Q-Q plots).

---

## 📐 Mathematical Formulas Applied

* **Conditional Probability**:
  $$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

* **Bayes' Theorem**:
  $$P(\text{Default} \mid \text{Score Tier}) = \frac{P(\text{Score Tier} \mid \text{Default}) \cdot P(\text{Default})}{P(\text{Score Tier})}$$

* **Cosine Similarity & Vector Angle**:
  $$\cos(\theta) = \frac{\mathbf{v_1} \cdot \mathbf{v_2}}{\|\mathbf{v_1}\| \|\mathbf{v_2}\|}, \quad \theta = \arccos\left(\frac{\mathbf{v_1} \cdot \mathbf{v_2}}{\|\mathbf{v_1}\| \|\mathbf{v_2}\|}\right)$$
