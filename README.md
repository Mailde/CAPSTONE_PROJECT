# 🧠 Black-Box Optimization (BBO) Capstone Project

## **1. Project Overview**

This capstone project focuses on the challenge of **optimising unknown black-box functions using limited information**, simulating a real-world **Bayesian optimisation** scenario.  
Each function represents a practical machine learning or scientific problem, such as radiation detection and drug discovery, where the underlying process is hidden, and each evaluation is expensive.

The overall goal is to **maximise eight unknown functions** by proposing one new query point per function in each round.  
Because the internal structure of these functions is not provided, success depends on designing intelligent query strategies that balance **exploration** (discovering new regions) and **exploitation** (refining known promising areas).

This project builds practical and transferable skills in:
- Bayesian optimisation and surrogate modelling  
- Exploration–exploitation trade-offs  
- Iterative, data-driven decision making under uncertainty  

These are essential techniques for modern data scientists and ML engineers who work with incomplete information, hyperparameter tuning, or experimental design in real-world applications.

---

## **2. Inputs and Outputs**

Each black-box function receives a vector of continuous inputs in the range `[0, 1]`.

- **Input format:**  
  A query point represented as `x1-x2-x3-...-xn`, each value rounded to **six decimal places**.  
  Example formats:
  - 2D: `0.123456-0.654321`
  - 4D: `0.211000-0.502340-0.761222-0.122599`

- **Input dimensionality:**  
  Varies across the eight functions, ranging from **2D to 8D**.

- **Output:**  
  A single real-valued response (`y`) representing the performance or objective value for that query.

- **Goal:**  
  Maximise the output value for each function.

Each week, one new query result is received per function, expanding the dataset and enabling refinement of the surrogate model and query strategy.

---

## **3. Challenge Objectives**

- **Primary objective:**  
  Maximise the outputs of eight unknown functions through iterative, data-driven optimisation.

- **Constraints and conditions:**
  - Only **one query per function per round**.  
  - The mathematical form of each function is **unknown** (no access to equations or visualisations).  
  - Must rely entirely on **observed input–output pairs** to guide future queries.  
  - **Response values are delayed** until after each round submission.  
  - Limited evaluation budget encourages efficient exploration.

Each function varies in complexity and dimensionality. Some are **multimodal** (with several local optima), while others are **unimodal**, requiring different exploration strategies.

---

## **4. Technical Approach**

### **4.1 Early Rounds (1–3): Exploratory Phase**

In the early rounds, I adopted a **broad exploratory approach** to capture the general landscape of each function.

- For **functions with multiple local optima**, I maintained a **strong exploratory component** to avoid premature convergence.  
- For **unimodal functions**, I began balancing exploration and exploitation, focusing more heavily on regions that appeared promising based on early model predictions.  
- I started **tuning hyperparameters** (e.g., kernel width, learning rate, acquisition function parameters) rather than relying solely on heuristics.

### **4.2 Modelling Approach**

- Built a **Gaussian Process (GP)** surrogate model to approximate each unknown function.  
- Used acquisition functions such as:
  - **Expected Improvement (EI)**
  - **Upper Confidence Bound (UCB)**  

These functions balance uncertainty and performance prediction, guiding where to query next.

- Hyperparameter tuning helps improve:
  - Model smoothness and adaptability  
  - Prediction reliability across uncertain regions  
  - Convergence speed toward the optimal region  

### **4.3 Exploration vs. Exploitation**

The **acquisition function** governs the exploration–exploitation trade-off:

- **Exploration:** Selects points with high uncertainty (large variance in model predictions) to gain new information.  
- **Exploitation:** Focuses on regions with high predicted performance to refine and exploit known optima.  

As the model matures, I progressively shift toward exploitation while maintaining a **small percentage of uncertainty-driven exploration** to improve model robustness.

### **4.4 Alternative Modelling Considerations**

Although not implemented initially, incorporating **Support Vector Machines (SVMs)** or **Neural Networks (NNs)** can enhance learning:

- A **soft-margin SVM** could separate high- vs. low-performing regions.  
- A **kernel SVM** or **deep neural network** could model nonlinear, high-dimensional relationships.  
- These models can complement regression-based surrogates by identifying promising “zones” within the search space.

---

## **5. Observations and Limitations**

As data accumulates, several challenges became evident:

- **Overfitting risk:** The surrogate model may overfit as complexity grows faster than data diversity.  
- **Irrelevant features:** Not all dimensions contribute equally to performance, especially in higher dimensions.  
- **Computational cost:** Updating Gaussian Processes becomes expensive as the dataset grows, but it was not a problem since we had few data points for each function. 

To address these issues:
- Maintain some **exploratory diversity** to prevent stagnation.  
- Conduct feature relevance analysis where possible.  
- Introduce **regularisation** and **adaptive acquisition functions** to balance accuracy and efficiency.

---

## **6. Future Directions (Pre–Query 5 Plan)**

In future rounds, I planned to:
- Explore **adaptive acquisition functions** that dynamically adjust exploration weight.  
- Test **alternative surrogate models**, such as Random Forests or Neural Networks.  
- Integrate **cross-validation** to prevent overfitting.  
- Evaluate **ensemble approaches** that blend GP and NN predictions for greater robustness.

---

## **7. Learning Outcomes**

This black-box setup mirrors the uncertainty and constraints typical of **real-world machine learning** projects.  

Through this process, I learned to:
- Make **strategic, evidence-based decisions** under uncertainty.  
- Balance **model trust and data-driven exploration**.  
- Iteratively refine models based on performance feedback.  
- Mitigate risks of **overfitting** and **model instability**.

These skills are directly transferable to:
- ML hyperparameter tuning  
- Experimental design  
- Process optimisation in applied research  

---

## **8. Repository Structure (Updated for Query 5)**

To improve clarity and reproducibility, the repository is now organised into well-defined folders:

```bash
CAPSTONE_PROJECT/
├── data/                 # Input/output .npy files
├── src/                  # Optimisation scripts and surrogate models
├── config/               # Parameter settings for reproducible runs
├── notebooks/            # Exploratory diagnostics and analysis
├── results/              # Timestamped output folders with plots and summaries
└── requirements.txt      # Dependency list``


Each optimisation run automatically generates timestamped folders under `data/` and `data_viz/`, containing:
- Surrogate model plots (GP or NN ensemble)  
- Batch suggestions and predicted next queries  
- JSON summaries for traceability  

This structure enhances **clarity**, **navigability**, and **reproducibility**, supporting both experimentation and formal reporting.

---

## **9. Coding Libraries and Packages**

The project relies primarily on **scikit-learn**, **NumPy**, **SciPy**, **pandas**, and **Seaborn**.

- **scikit-learn** provides the **Gaussian Process Regressor (Matern kernel)** and **MLPRegressor** for hybrid surrogate modelling.  
- **SciPy’s Sobol sampling** ensures efficient coverage of the search space.  
- **pandas** and **Seaborn** power data analysis, correlation visualisation, and convergence tracking.  

These libraries balance **stability**, **interpretability**, and **efficiency**.  
The main trade-off is between **accuracy** (from GPs) and **scalability** (from NNs).  
This hybrid GP–NN ensemble approach combines the precision of GPs in early rounds with the flexibility and scalability of neural networks as data increases.

---

## **10. Documentation Updates**

The documentation now includes:
- A full description of the **hybrid surrogate strategy** (GP ↔ NN switching)  
- Clear input/output definitions and data flow  
- Auto-generated plots and summaries for each round  
- Directory structure and reproducibility instructions  

These updates were implemented and validated as part of the **Query 5 submission**, aligning documentation with the latest modelling strategy.

---

## **11. Reflection and Future Plans**

**Repository improvements:**  
I plan to modularise the optimisation code under `src/` and provide config templates for reproducibility.  
Adding Jupyter notebooks for exploratory analysis will make the process more transparent and interactive.  

**Framework direction:**  
While **scikit-learn** remains optimal for fast prototyping, future iterations may incorporate **PyTorch-based ensembles** for richer uncertainty modelling in higher-dimensional spaces.  

**Documentation:**  
Future README updates will add quick-start instructions, improved visual interpretation guides, and reproducibility scripts.  

**Completion status:**  
The current repository meets the capstone’s expectations for structure, reproducibility, and transparency.  
This analysis and documentation update are officially included as part of the **Query 5 submission**.

---

