## NightHawk: DBMS Silent Fix Analysis

This repository contains the implementation of the **NightHawk** project, which supports the analysis of *silent fixes* in DBMSs. It includes related datasets, source code, and pretrained models. The repository is organized as follows:

- **Datasets**
  This directory contains the data used throughout the analysis:
  - `upstream_DBMS.numbers`: Metadata of upstream DBMSs, including names, CPEs, GitHub links, DBMS types, and GitHub star counts;
  - `downstream_DBMS.numbers`: Metadata for selected downstream DBMSs;
  - `dbms_cve_fix.json`: Mapping between CVEs and their corresponding fix commits;
  - `train_dataset.jsonl`: General-purpose fix/non-fix training data, including GitHub commit links, commit messages, code diffs, and labels (1 = fix, 0 = non-fix);
  - `train_dbms_dataset.jsonl`: Domain-specific training data from DBMS projects;
  - `test_dbms_dataset.json`: Testing dataset for DBMS-specific evaluation.
- **Code (under the root folder)**
  Source code for model training and inference:
  - `main.py`: Main pipeline for silent fix detection;
  - `adv_perturb.py`: Script to inject adversarial perturbations into training data;
  - `utils.py`: Auxiliary utility functions.
- **Models**
  The `trained_models` directory contains pretrained models used in experiments:
  - `nighthawk/`: The final NightHawk model with adversarial training;
  - `nighthawk_without_adv/`: An ablation variant trained without adversarial perturbation.

* **Environment:** We conduct our experiments on Ubuntu 22.04 with Python 3.10.12. The complete list of dependencies is provided in the accompanying `requirements.txt` file.
* **Additional Functionality**
  NightHawk integrates several existing tools, including modules for code reuse detection and CVE patch collection. For details on the usage and implementation of these components, please refer to the following related studies:
  * CENTRIS: A precise and scalable approach for identifying modified open-source software reuse, ICSE'23
  * Enhancing Security in Third-Party Library Reuse--Comprehensive Detection of 1-day Vulnerability through Code Patch Analysis, NDSS'25