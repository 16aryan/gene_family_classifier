# gene_family_classifier
Gene ML Model

This project is a machine learning pipeline that classifies DNA sequences into gene families using `k-mer encoding` and a `RandomForestClassifier`.

---
🧬 Gene Family Classification using Machine Learning

This project builds an end-to-end machine learning pipeline to classify gene sequences into their respective gene families using DNA sequence data and k-mer encoding.

🔍 Problem Statement

Gene families are groups of related genes that share similar sequences and biological functions. Automatically classifying new sequences into gene families helps in genomic research, drug discovery, and evolutionary studies.


OUTPUT:

🚀 Step 1: Loading and preprocessing data...
✅ Loaded 455796 sequences
🔤 Step 2: Vectorizing sequences...
✅ Vectorized shape: (455796, 228)
✂️ Step 3: Splitting data...
✅ Training samples: 364636, Test samples: 91160
🧠 Step 4: Training model...
⚙️ Training RandomForestClassifier...
✅ Accuracy: 0.7876
💾 Model saved to: models/model.pkl
🎉 All done!



🛠 Tech Stack
	•	Python
	•	scikit-learn (RandomForest, LabelEncoder, train_test_split)
	•	pandas
	•	joblib (for model saving)
	•	CountVectorizer (text vectorization)



gene_family_classifier-main/
├── data/
│   └── gene_families.csv
├── models/
│   └── model.pkl
├── src/
│   ├── data_preprocessing.py
│   └── train_test_split.py
├── train_model.py
├── main.py
└── README.md










git init
git add .
git commit -m "Mid"
git push origin main

