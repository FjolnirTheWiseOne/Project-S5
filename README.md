<h1 align="center">DeepCareX: AI-based Healthcare System</h1>

<h3 align="center">Project S5 | Capstone Project 2025-2026</h3>
<h4 align="center">Faculty of Computer Science and Applied Mathematics</h4>

---

## 👥 Team Members & Roles

- **Artur Gevorgyan** - Team Lead, Cyber Security, ML Pentesting
- **Emil Hovhannisyan** - Backend Development, ML Optimization
- **Eliza Manukyan** - Frontend Development (UI/UX), Backend Integration
- **Artur Torosyan** - ML Validation, QA (API & Model Testing)

---

## 📝 Table of Contents

- [Introduction](#intro)
- [Objective](#obj)
- [Literature Review](#LR)
- [Methodology and Architecture](#MID)
- [Algorithms Implemented](#Algo)
- [Experimentation Setup and Results](#Exp)
- [System Features](#features)
- [User Interface (UI)](#ui)
- [Future Work](#future)
- [Installation & Usage](#install)
- [Docker Deployment](#docker)
- [Contribution](#contri)

---

## **Introduction** <a name="intro"></a>

Effective diagnosis of a disease is a significant need on a large scale. The development of a tool for early diagnosis and an efficient course of therapy is extremely difficult due to the complexity of the many disease mechanisms and underlying symptoms of the patient population. DeepCareX addresses these challenges by integrating **Machine Learning (ML)** and **Deep Learning (DL)** into a user-friendly web application.

Artificial intelligence (AI) in the medical field largely focuses on creating algorithms and methods to assess if a system's behavior in diagnosing diseases is accurate. This project serves as a decision-support tool for both patients and doctors to identify potential health risks early.

## **Objectives** <a name="obj"></a>

The primary objectives of this project are:

1. **Identify diseases by analyzing symptoms:** Users input required clinical data (e.g., glucose levels, X-ray scans, BMI) specific to the disease module.
2. **Generate specific diagnostic reports:** The system provides real-time prediction (Positive/Negative) with probability confidence scores.
3. **Deep Learning Integration:** Utilize advanced models (CNNs, Transfer Learning) to distinguish between similar symptoms caused by different pathologies.
4. **Data Persistence:** Provide a secure platform where users can save their diagnostic history for future reference.
5. **Accessibility:** Democratize access to initial health screening via a responsive web interface.

## **Literature Review** <a name="LR"></a>

Our research focused on the efficacy of **Transfer Learning models** (like VGG19 and DenseNet201) for medical image processing versus **Ensemble methods** (Random Forest, XGBoost) for tabular clinical data.

- **Tabular Data:** Studies suggest that for structured datasets (Heart Disease, Diabetes), ensemble techniques often outperform single classifiers. We utilized **Random Forest** (Bagging) and **XGBoost** (Boosting) based on their proven robustness in handling outliers and complex feature interactions.
- **Medical Imaging:** For unstructured data (MRI, X-Rays), we referenced works utilizing **ResNet** and **VGG** architectures. Research demonstrates that enhanced VGG16 models can detect COVID-19 with high accuracy. Similarly, we applied **ResNet152V2** and **DenseNet201** to overcome data scarcity issues through transfer learning.

## **Methodology and Implementation Details** <a name="MID"></a>

The team adopted an **Agile development methodology**, allowing for iterative development of the predictive models and the web interface.

1.  **Data Collection:** Datasets were aggregated from open-source repositories (Kaggle) to ensure diverse training data for the 8 target diseases.
2.  **Pre-processing:**
    - **Cleaning:** Handling missing values via statistical imputation.
    - **Encoding:** Using Label Encoding and One-Hot Encoding for categorical variables.
    - **Normalization:** Applying `StandardScaler` for numerical consistency.
    - **Augmentation:** Rotating and flipping images to prevent overfitting in CNNs.

## **Algorithms Implemented** <a name="Algo"></a>

We employed a specific algorithm for each disease to maximize accuracy:

### **1. Random Forest (Ensemble)**

Used for **Breast Cancer** and **Heart Disease**. It constructs a multitude of decision trees at training time and outputs the class that is the mode of the classes (classification) of the individual trees.

### **2. XGBoost (Extreme Gradient Boosting)**

Used for **Diabetes** and **Hepatitis C**. It is an optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable.

### **3. Convolutional Neural Networks (CNNs)**

Used for **Alzheimer's** and **Kidney Disease**. We designed custom CNN architectures with convolutional, max-pooling, and fully connected layers to extract features from medical images.

### **4. Transfer Learning Models**

- **VGG19:** Used for **Brain Tumor** detection. A 19-layer deep network pre-trained on ImageNet.
- **ResNet152V2:** Used for **COVID-19** detection. Utilizes skip connections to solve the vanishing gradient problem.
- **DenseNet201:** Used for **Pneumonia**. Connects each layer to every other layer in a feed-forward fashion.

## **Experimentation Setup and Results** <a name="Exp"></a>

The system currently supports 8 diseases. Below are the final testing accuracies achieved by our models:

| Disease             | Model              | Testing Accuracy |
| :------------------ | :----------------- | :--------------- |
| **Breast Cancer**   | Random Forest      | 94%              |
| **Diabetes**        | XGBoost Classifier | 97%              |
| **Hepatitis C**     | XGBoost Classifier | 97%              |
| **Heart Disease**   | Random Forest      | 99%              |
| **Brain Tumor**     | VGG19              | 97%              |
| **COVID-19**        | ResNet152V2        | 95%              |
| **Alzheimer's**     | Custom CNN         | 98%              |
| **Kidney Disorder** | Custom CNN         | 97%              |
| **Pneumonia**       | DenseNet201        | 83%              |

### **Tools Used**

- **Backend:** Python 3.9, Flask
- **Frontend:** HTML5, CSS3, Bootstrap v5.3
- **ML Libraries:** TensorFlow 2.9, Keras, Scikit-learn, XGBoost, Pandas, NumPy
- **Database:** SQLite3

## **User Interface (UI)** <a name="ui"></a>

### Home Page

![](./media/image38.png)

### Diagnostic Input (Breast Cancer Example)

![](./media/image45.png)

### Prediction Result

![](./media/image53.png)

## **Future Work** <a name="future"></a>

While the current system is functional, the following enhancements are planned:

1.  **Expanded Disease Database:** Adding support for dermatological conditions (Skin Cancer) and Malaria.
2.  **Professional Integration:** A portal for doctors to view patient-saved reports directly with Role-Based Access Control (RBAC).
3.  **Advanced Security:** Implementation of **AES-256 encryption** and HIPAA compliance standards for medical data.
4.  **Mobile Application:** Developing a **React Native** version for mobile access (iOS/Android).

## **Installation & Usage** <a name="install"></a>

To run DeepCareX locally for development or testing:

### 1. Clone the Repository

```sh
git clone https://github.com/FjolnirTheWiseOne/Project-S5
cd DeepCareX
2. Install Dependencies

Ensure you have Python 3.9+ installed.

pip3 install -r requirements.txt

Alternatively, install manually:

pip3 install numpy pandas scikit-learn matplotlib os scipy seaborn xgboost joblib pickle sqlite3 tensorflow flask
3. Initialize the Database

You must create the SQLite database before running the app.


cd Website/database
python3 database.py

This creates the database.db file.

4. Run the Application

Navigate back to the website directory and start the Flask server.

cd ..
python3 main.py
5. Access the System

Open your web browser and go to:

http://127.0.0.1:5000/

Access: http://localhost:5000

Contribution <a name="contri"></a>

This project was developed for the Project S5 Capstone.

Artur Gevorgyan (Team Lead)

Emil Hovhannisyan

Eliza Manukyan

Artur Torosyan

<p align="center">Academic Year 2025-2026</p>
```
