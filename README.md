# honoursProject

**Dataset Setup & Initial Exploration**

- Added all CIC-IDS2017 CSVs to the /data directory.

- Created initial notebook: notebooks/01_explore_dataset.ipynb for dataset inspection.

- Verified that all files load correctly and display basic structure (df.info(), df.columns, df[' Label'].value_counts()).

- Confirmed approach: each dataset will be processed individually (load → clean → train → save results), then concatenated for a general IDS evaluation.

**Established next steps:**

- Clean one dataset (Friday-WorkingHours-Afternoon-DDos)

- Train a baseline RandomForest model

- Save predictions to /results

- Document performance metrics
