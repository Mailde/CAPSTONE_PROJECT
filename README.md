# 🧠 Black-Box Optimization (BBO) Capstone Project

## **1. Project Overview**

This capstone project focuses on the challenge of **optimising unknown black-box functions using limited information**, simulating a real-world **Bayesian optimisation** scenario.  
Each function represents a practical machine learning or scientific problem, such as radiation detection, robot control, or drug discovery, where the underlying process is hidden, and each evaluation is expensive.

The overall goal is to **maximise eight unknown functions** by proposing one new query point per function in each round. Because the internal structure of these functions is not provided, success depends on designing intelligent query strategies that balance **exploration** (discovering new regions) and **exploitation** (refining known promising areas).

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
- I began **tuning hyperparameters** (e.g., kernel width, learning rate, and acquisition function parameters) rather than relying solely on heuristics.

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

As the model matures, I progressively shift toward exploitation while maintaining a **small percentage of uncertainty-driven exploration** to continue improving model robustness.

### **4.4 Alternative Modelling Considerations**

Although not implemented yet, incorporating **Support Vector Machines (SVMs)** could enhance classification of performance regions:

- A **soft-margin SVM** could separate high- vs. low-performing regions, even in the presence of noisy evaluations.  
- If the response surface is **nonlinear**, a **kernel SVM** (RBF or polynomial) could capture more complex relationships and boundaries.  
- SVMs could therefore complement regression models by identifying promising “zones” within the search space for further optimisation.

---

## **5. Observations and Limitations**

As data accumulates, several challenges become apparent:

- **Overfitting risk:** The surrogate model may overfit as complexity grows faster than the diversity of available data.  
- **Irrelevant or redundant features:** In higher-dimensional functions, not all variables may contribute meaningfully to performance.  
- **Computational cost:** Updating surrogate models (especially GPs) becomes more expensive with larger datasets.  

To mitigate these issues:
- Continue incorporating exploratory queries even when certain regions appear optimal.  
- Even with a small number of input dimensions, performing feature importance analysis could help clarify which variables most strongly influence performance
- Apply **regularisation** and **adaptive acquisition functions** to maintain balance.

---

## **6. Future Directions**

In the next rounds, I plan to:
- Explore **adaptive acquisition functions** that adjust exploration weight dynamically.  
- Investigate **alternative surrogate models**, such as Random Forest regressors or Neural Networks, for higher-dimensional tasks.  
- Integrate **cross-validation** to monitor generalisation and prevent overfitting.  
- Evaluate **ensemble strategies** that combine GP and SVM predictions to improve robustness.

---

## **7. Learning Outcomes**

This black-box setup mirrors the uncertainty and constraints typical of **real-world machine learning** and **data science projects**.  

Through this process, I have learned to:
- Make **strategic decisions under uncertainty**, guided by model predictions and evidence.  
- Balance **data collection, model trust, and uncertainty management**.  
- Apply iterative reasoning — using each round’s outcome to refine both the model and the search strategy.  
- Recognise and mitigate **model limitations** such as overfitting and poor generalisation.

These skills directly translate to real-world scenarios such as:
- Hyperparameter tuning of ML models  
- Experimental design in AI research  
- Process optimisation in engineering and applied sciences  

---


