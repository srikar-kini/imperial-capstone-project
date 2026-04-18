# Black‑Box Optimisation (BBO) Capstone Project

## 1. Project Overview

This project is part of the Black‑Box Optimisation (BBO) capstone, where the goal is to optimise several unknown functions without ever seeing their internal structure. Instead of having a formula, I only receive a numeric output for each query I submit. This setup reflects many real‑world machine‑learning challenges, where we often must tune systems or models without full transparency.

The core purpose of the project is to learn how to make intelligent decisions under uncertainty. Each function behaves differently, may be smooth or noisy, and may respond strongly or weakly to certain inputs. Because I cannot see the underlying equations, optimisation becomes a strategic problem: deciding where to sample next, using limited feedback and imperfect information.

The project also strengthens skills that are highly valuable in real‑world ML and engineering roles: dealing with incomplete information, building surrogate models, balancing exploration and exploitation, and iteratively refining predictions. These are the same challenges involved in hyperparameter tuning, A/B experimentation, and optimising systems in production.

For my career, especially in engineering and ML‑driven work, the BBO capstone helps build intuition for optimisation workflows, GP modelling, and how to make systematic decisions when the “ground truth” is not available.

## 2. Inputs and Outputs

Each function accepts a vector of values in the format: `x1-x2-x3-...-xn`

Where each value:

- Is a decimal between 0 and 1
- Is written with six decimal places
- Must follow strict formatting rules

Different functions have different dimensions:

- Function 1 → 2‑D
- Function 2 → 2‑D
- Function 3 → 3‑D
- Function 4 → 4‑D
- Function 5 → 4‑D
- Function 6 → 5‑D
- Function 7 → 6‑D
- Function 8 → 8‑D

Example input (3‑D):

```
0.123456-0.654321-0.111111
```

Output: A single numeric value that represents the function’s response. Higher values usually indicate better performance, but this depends on the task.

Because the submission interface requires a vector format even for outputs, I repeat the predicted value `n` times for an `n`‑dimensional function.

## 3. Challenge Objectives

The main objective is to maximise the output of each black‑box function while respecting strict constraints.

Key constraints:

- I only get a limited number of queries per week.
- I cannot see the function’s shape or behaviour.
- I must find good inputs even when the function may be noisy or multi‑modal.
- Each week’s decision must be based only on previous inputs and outputs.
- The format of each submission must match precise rules.

This setup challenges me to design a strategy that learns efficiently and avoids wasting queries on uninformative areas. It also mirrors real situations where experiments are expensive, noisy, or time‑limited.

## 4. Technical Approach (Iterations 1–3)

My technical approach has evolved across the first three iterations.

### Iteration 1 — Broad Exploration

- I tried points spread across the space.
- The goal was to understand rough behaviour.
- I used simple heuristics, since no patterns existed yet.

### Iteration 2 — Local Refinement

- I focused more on areas that produced better results.
- I used directional adjustments based on previous outputs.
- Still kept some exploration to avoid getting stuck.

### Iteration 3 — Model‑Driven Predictions

By the third iteration:

- I built a Gaussian Process (GP) surrogate model per function.
- The model predicts the mean and uncertainty for new points.
- This allowed me to make more informed decisions.
- Function 5 required output scaling due to formatting rules.

I am now moving toward a Bayesian Optimisation style workflow, using the GP as a guide to choose promising query points.

### Balancing Exploration and Exploitation

- Exploration: still necessary early on or when uncertainty is high.
- Exploitation: important near high‑performing regions to refine them.
- I combine both by adjusting step sizes and using the GP’s uncertainty.

### Possible Future Methods

- SVMs: I could label outputs as “high” or “low” and train a soft‑margin SVM to identify promising regions. A kernel SVM may help if the space is non‑linear.
- Linear or Logistic Regression: Could help identify simple feature relationships, though most functions appear non‑linear.
- Full Bayesian Optimisation: Integrating Expected Improvement (EI) or Probability of Improvement (PI) for automated suggestion of next points.

---

