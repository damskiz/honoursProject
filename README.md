# honoursProject

08/10/25 - **Dataset Setup & Initial Exploration**

- Added all CIC-IDS2017 CSVs to the /data directory.

- Created initial notebook: notebooks/01_explore_dataset.ipynb for dataset inspection.

- Verified that all files load correctly and display basic structure (df.info(), df.columns, df[' Label'].value_counts()).

- Confirmed approach: each dataset will be processed individually (load → clean → train → save results), then concatenated for a general IDS evaluation.

**Established next steps:**

- Clean one dataset (Friday-WorkingHours-Afternoon-DDos)

- Train a baseline RandomForest model

- Save predictions to /results

- Document performance metrics

09/10/25
- Project scope altered to be more research oriented, 
putting practical work on ice and priotitising chapter 2 of dissertation,
gathering recources and research


15/10/25
- IPO revised:
- Title updated
- project to be much more research focused
- focusing on 1 or 2 applications rather than a general "giga ML model" (scope was far too ambitious) 
- HTTPS/SMTP/general Web-based IDS to cater toward industry adoption
- dropping forensic timeline to cater for refined scope
