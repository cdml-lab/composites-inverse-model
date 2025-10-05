# 🚧 WORK IN PROGRESS – CONTENT NOT FINALIZED  🚧

# 🔬 Inverse Model for Frustrated Composites

## 📖 Overview
This project builds an inverse model for **frustrated composites** using deep learning. It consists of three main steps: **data preparation**, **model training**, and **prediction**. 
it supports two seperate options to get the inverse solution: 
1. train a direct inverse model and predict the fiber orientation of a surface.
2. train a forward model and then optimize to find the fiber orientation of a surface.


---

## 📂 Project Structure
```
inverse-model-frustrated-composites/
│── main/
│   ├── prepare_dataset.py   # Prepares dataset for training
│   ├── train_model.py       # Trains the ML model
│   ├── predict_model.py     # Uses the trained model for predictions
│── modules/                 # Helper modules for main files
│   ├── s1_convert_excel_to_h5.py
│   ├── s2_clean_and_reshape_h5.py
│   ├── s3_merge_h5_files.py
│── utils/                   # Utility scripts (optional tasks)
│   ├── analyze_dataset.py
│   ├── view_random_samples.py
│── data/                    # Example datasets (link provided below)
│── models/                  # Trained models
│── results/                 # Output results
│── README.md                # Documentation
│── requirements.txt         # Dependencies
│── config.yaml              # Project configuration file
```

---

## 🚀 Installation
### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/your-username/inverse-model-frustrated-composites.git
cd inverse-model-frustrated-composites
```

### **2️⃣ Set Up a Virtual Environment**
```sh
python -m venv .venv
source .venv/bin/activate   # MacOS/Linux
# OR
.venv\Scripts\activate      # Windows
```

### **3️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```

---

## 📂 Example Dataset
The example dataset can be downloaded from OneDrive:  
📂 **[Download Example Dataset](https://onedrive.live.com/your-link-here)**

Once downloaded, place the dataset inside the `data/` folder.

---

## 🔧 Changing Project Parameters
### **Dataset Normalization & Preprocessing**
Modify `config.yaml` to adjust **normalization settings**:
```yaml
normalization:
  method: "min-max"  # Options: "min-max", "z-score", "none"
  min: 0
  max: 1
```

### **Hyperparameter Tuning**
Modify the hyperparameters in `config.yaml`:
```yaml
hyperparameters:
  learning_rate: 0.001
  batch_size: 32
  epochs: 50
  optimizer: "adam"  # Options: "sgd", "adam"
```

### **W&B Integration**

https://wandb.ai/kapon-gal-technion/forward_model?nw=nwuserkapongal

Ensure W&B tracking is enabled by setting up your API key:
```sh
wandb login your-api-key
```
Modify `config.yaml` to log experiments:
```yaml
wandb:
  enable: true
  project: "frustrated-composites"
```



---

## 🔧 Usage
### **1️⃣ Prepare Dataset**
Convert `.xlsx` data to HDF5 format, clean, reshape, and merge.
```sh
python main/prep_dataset.py
```
Manually change the names and sizes of the datasets you wish to use in order to convert them from an excel to the appropriate formating. 
for example xlsx files named "Dataset_Input_82.xlsx" and "Dataset_Outut_82.xlsx" will be described as part of the dictionary as:
   "82": (y, x, n)
where y and x are the y and x size of the sheet and n is the number of channels you plan to use.
if you wish to change the type of data used from x,y,z to something else you need to change it in "modules/s2_clean_and_reshape_h5.py"
and change the variable "remove_split_columns" according to the columns you wish to keep.

list of columns collected by our GH dataset creation code:



### **2️⃣ Train Model**
Train a deep learning model on the dataset.
```sh
python main/train_model.py
```

### **3️⃣ Predict Results**
Use the trained model to make predictions.
```sh
python main/predict_model.py
```

---

## 📂 File & Folder Descriptions
| **File/Folder** | **Description** |
|----------------|----------------|
| `main/prepare_dataset.py` | Converts & preprocesses dataset. |
| `main/train_model.py` | Trains the deep learning model. |
| `main/predict_model.py` | Runs the model to make predictions. |
| `modules/` | Contains helper scripts for processing. |
| `utils/` | Additional scripts for dataset analysis. |
| `data/` | Stores raw and processed datasets. |
| `models/` | Stores trained models. |
| `results/` | Saves experiment outputs. |

---

## 🛠️ Dependencies
- **Python 3.8+**
- **Required Libraries:**
  ```sh
  pip install torch h5py pandas numpy openpyxl wandb
  ```

---

## 🏆 Contributing
1. **Fork the repo** and clone it locally.
2. Create a **new branch** for your feature.
3. Commit your changes with a **descriptive message**.
4. Push your branch and create a **Pull Request**.

---

## 🔍 Troubleshooting


---

## 📜 License


