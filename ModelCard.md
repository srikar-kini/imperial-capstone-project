# Datasheet and Model Card
## Black‑Box Optimisation (BBO) Capstone Project

---

# Part 1 – Datasheet for the BBO Optimisation Dataset

## Motivation
This dataset was created to support a **black‑box optimisation (BBO)** task where the objective is to iteratively improve unknown functions under strict input constraints. The dataset enables analysis of how query strategies evolve over time and supports reflection on optimisation behaviour, scaling effects, robustness, transparency, and interpretability.

## Composition
The dataset contains:
- **10 weekly iterations** (Rounds 1–10)
- **8 independent black‑box functions**
- **Inputs:** Vectors with dimensionality ranging from 2 to 8
    - Each input value is constrained to `[0, 1)` and formatted as `0.xxxxxx`
- **Outputs:** Single real‑valued scalar per function per week

The data is stored in a **single Excel workbook**, organised sequentially by week with clearly labelled input and output sections.

### Gaps and Irregularities
- Early rounds contain **invalid or extreme inputs** (e.g. Week‑3) that violate constraints
- Later rounds show **dense clustering** around small regions of the input space
- Large portions of the global search space remain unexplored

## Collection Process
Queries were generated **manually but systematically**, one per function per week, across ten rounds.

The strategy evolved over time:
- **Early rounds:** exploratory probing to understand stability and constraints
- **Middle rounds:** elimination of invalid regions and identification of promising neighbourhoods
- **Later rounds (6–10):** cautious local refinement around high‑performing inputs

All decisions were informed by observed historical performance rather than random sampling.

## Preprocessing and Intended Uses
Preprocessing was minimal:
- Invalid inputs were **identified and excluded from reasoning**
- No scaling, normalisation, or transformation was applied to valid inputs or outputs

### Intended Uses
- Studying black‑box optimisation dynamics
- Analysing transparency and interpretability in decision‑making
- Comparing exploration vs exploitation strategies

### Inappropriate Uses
- Training general predictive or surrogate models
- Claiming global optimality
- Using the dataset outside its constrained optimisation context

## Distribution and Maintenance
The dataset is maintained within the project repository and distributed for **educational and assessment purposes only**.  
There are no external usage rights. Maintenance and updates are the responsibility of the project author.

---

# Part 2 – Model Card for the BBO Optimisation Approach

## Overview
- **Model name:** Incremental Local Refinement BBO
- **Type:** Human‑guided black‑box optimisation strategy
- **Version:** v1.0 (10‑round implementation)

This is not a predictive model, but a **decision‑making strategy** for selecting queries under uncertainty.

## Intended Use
### Suitable For
- Low‑budget black‑box optimisation
- Educational settings emphasising interpretability
- Problems with strict input constraints

### Not Suitable For
- High‑dimensional global optimisation
- Automated large‑scale optimisation
- Tasks requiring guaranteed convergence to a global optimum

## Strategy Details
The approach evolved across ten rounds:
- Early exploration identified unstable and invalid regions
- Extreme failures (e.g. Week‑3) highlighted the importance of constraint awareness
- Later rounds focused on **small, local refinements** around high‑performing regions
- Each function was treated independently, allowing per‑function adaptation

Decisions were guided by:
- Best historical performance
- Directional trends between recent rounds
- Risk management over aggressive optimisation

## Performance Summary
Performance was evaluated using:
- Best observed output per function
- Stability of outputs over time
- Absence of catastrophic regressions

Functions such as **F5 and F8** showed steady improvements with diminishing returns, while **F1 and F7** saturated early. Other functions stabilised after early volatility.

## Assumptions and Limitations
### Key Assumptions
- Local smoothness near high‑performing regions
- Incremental refinement is safer than large jumps
- Recent performance is informative for near‑future queries

### Limitations
- Strong bias toward exploitation
- Limited global exploration
- Reduced ability to discover distant optima late in the process

## Ethical and Transparency Considerations
Transparency is a core strength of this approach:
- All decisions are traceable to recorded data
- No hidden parameters or opaque optimisation routines are used
- Another researcher could reproduce the strategy using the dataset and documented rules

Adding more complexity (e.g. advanced surrogate models) would reduce clarity without improving reliability in this low‑data setting. The current structure is therefore intentionally simple and sufficient.

---

``