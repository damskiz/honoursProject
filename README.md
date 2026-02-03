# Lightweight ML-Based Intrusion Detection System for SME Networks

Honours project developing a machine learning-based IDS tailored for small/medium enterprises, focusing on detecting attacks in HTTPS traffic, DDoS scenarios and brute force attacks.

## Project Overview

This project aims to design, implement, and evaluate a lightweight intrusion detection system that applies ML techniques to detect anomalous traffic within SME networks, with emphasis on:
- DDoS attack detection
- Brute force attempts
- Resource-efficient deployment for constrained environments
- HTTPS anomaly detection (without decryption)

## Setup

### Prerequisites
- Python 3.12+
- Ubuntu/WSL recommended

### Installation
```bash
# Clone the repository
git clone git@github.com:damskiz/honoursProject.git
cd honoursProject

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install pandas numpy matplotlib scikit-learn
```

## Project Structure
```
honoursProject/
├── data/
│   ├── raw/              # Original datasets (not in git)
│   └── processed/        # Cleaned datasets
├── scripts/              # Python scripts for data processing and training
├── notebooks/            # Jupyter notebooks for exploration
├── models/               # Trained model files
├── results/              # Output metrics, plots, reports
└── docs/                 # Documentation and dissertation chapters
```

## Datasets

- **CIC-IDS2017**: Modern attack scenarios with TLS flows
- **UNSW-NB15**: Diverse attack categories with realistic traffic

*Datasets not included in repository due to size. Download from:*
- CIC-IDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
- UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset

## Usage
```bash
# Activate virtual environment
source venv/bin/activate

# Run data exploration
python scripts/explore.py

# (More scripts to be added as project develops)
```

## Current Status

- [x] Environment setup
- [x] Dataset acquisition (CIC-IDS2017)
- [x] Initial data exploration
- [ ] Data preprocessing and cleaning
- [ ] Feature selection
- [ ] Model training and evaluation
- [ ] Testbed deployment

## Author

Damian Wright  
Honours Project - SOC10101  
Edinburgh Napier University

## License

Academic project - not for commercial use
