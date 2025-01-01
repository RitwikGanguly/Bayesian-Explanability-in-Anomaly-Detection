# Bayesian Explainability for Real-Time Anomaly Detection in Medical Diagnostics
## Introduction:
In the modern landscape of healthcare, patient safety and quality of care have become 
paramount. Advanced machine learning models are increasingly being adopted to analyze 
patient data, identify anomalies, and assist clinicians in decision-making. However, 
despite the impressive accuracy of these models, their black-box nature poses significant 
challenges. Clinicians require interpretable and trustworthy insights, especially when 
these models flag anomalous behaviors in critical systems such as Intensive Care Units 
(ICUs). 
This project focuses on combining anomaly detection with Bayesian explainability, creating 
a transparent and reliable framework for identifying and understanding anomalies in real
time medical diagnostics. By leveraging Bayesian principles, this approach quantifies the 
uncertainty in explanations, providing healthcare practitioners with not just predictions 
but also the confidence levels associated with them. 
In the era of Explainable A.I, I will be using the Bayesian Explainability to Anomaly 
Detection output and making the prediction more trustworthy to the end user. 
The project is titled "Bayesian Explainability for Real-Time Anomaly Detection in 
Medical Diagnostics", and it employs the MIMIC-III (Medical Information Mart for 
Intensive Care) dataset to evaluate and showcase the framework's efficacy.

## Problem Statement: 
Anomaly detection is a critical component of real-time medical diagnostic systems. It 
involves identifying unusual patterns in patient data that may indicate potential risks or 
abnormal conditions. While deep learning models like LSTM Autoencoders are highly 
effective at detecting such anomalies, their predictions are often opaque and lack 
interpretability. This black-box nature makes it difficult for clinicians to trust and act upon 
these predictions, especially in life-critical scenarios. 
Moreover, the healthcare domain is characterized by high uncertainty, variability in patient 
conditions, and data sparsity. Traditional explainability techniques such as LIME (Local 
Interpretable Model-Agnostic Explanations) and SHAP (SHapley Additive exPlanations) fail 
to adequately quantify this uncertainty, further limiting their utility in critical medical 
applications. 

To address and solve these issues, this project integrates, 
1) **Dataset Chosen:** Have chosen the MIMIC-III diagnostic dataset and after 
transformation make this in a format on which I can do the work. 
2) **Bayesian explainability techniques (BayesLIME and BayesSHAP):**  To provide 
interpretable and uncertainty-aware explanations for the flagged anomalies. 
The main agenda is that, if from this interpretable technique, the trust, 
transparency, and clinical utility of anomaly detection systems in real-time 
healthcare settings can be enhanced.

## Methodology:  
### Dataset Description: 
The MIMIC-III (Medical Information Mart for Intensive Care) database is an extensive, 
publicly available dataset containing de-identified health-related data of over 40,000 
critical care patients admitted to intensive care units (ICUs) at the Beth Israel Deaconess 
Medical Center between 2001 and 2012. It integrates diverse types of data, including: 

1. Demographics: Information about patient age, gender, and ethnicity. 
2. Clinical Notes: Unstructured text data documenting patient history, treatment, and 
progress. 
3. Laboratory Tests: Time-stamped measurements of biochemical markers like 
glucose levels, creatinine, and hemoglobin. 
4. Vitals and Physiological Signals: Parameters such as heart rate, blood pressure, 
and oxygen saturation recorded at regular intervals. 
5. ICU Stays: Details about ICU admissions, durations, and associated interventions.

![mimic_dataset](https://github.com/RitwikGanguly/Bayesian-Explanability-in-Anomaly-Detection/blob/main/data_snaps/all_files_mimic.png)

### NORMAL & ABNORMAL REPORTS:

As I will be fit my data to the LSTM Autoencoder model and will be getting some 
reconstruction. For training the data I will taking the normal_flagged lab tests, on which I 
will build the model, and through the abnormal_flagged data I will check the anomaly in the 
model. For more better data view, I transformed the datasets as such, the lab test will be on the columns and the different patients lab tests with their respective charttime will be at the Row. The datasets look like, like the below –  

![mimic_normal_un](https://github.com/RitwikGanguly/Bayesian-Explanability-in-Anomaly-Detection/blob/main/data_snaps/mimic_normal_data.png)

### NULL value IMPUTATION: 
Now the main problems arise, the null value imputation. Fow the correct and diverse null 
value imputation, I have used the K-Nearest Neighbors Imputation (KNNImputer). 
The K-Nearest Neighbors (KNN) Imputer is an advanced imputation technique that 
estimates missing values based on the values of the nearest neighbors in the dataset. 
Unlike simpler methods like mean or median imputation, KNN Imputer retains the 
relationships between features, which is crucial for preserving the integrity of multivariate 
healthcare data. 
The missing value for a feature is replaced with the mean (or another aggregation function) 
of the corresponding feature values from its k nearest neighbors. Now the datasets look – 

![abc](https://github.com/RitwikGanguly/Bayesian-Explanability-in-Anomaly-Detection/blob/main/data_snaps/mimic_normal_processed.png)

Like the same the Abnormal Dataset is also transformed and imputed and are looking like –  

![cvb](https://github.com/RitwikGanguly/Bayesian-Explanability-in-Anomaly-Detection/blob/main/data_snaps/mimic_abn_processed.png)


These two datasets are used at the anomaly detection LSTM Autoencoder Model.


## Main Algorithm: 

Before feeding the data to the LSTM Autoencoder model the is preprocessed (used 
MinmaxScalar) and transform ass such, the unique patients will be at each sample and the 
time-stamp will be at the columns and the lab-tests values will be filled as the tensor for 
each cell entry.

![ty](https://github.com/RitwikGanguly/Bayesian-Explanability-in-Anomaly-Detection/blob/main/data_snaps/mimic_abn_tensor.png)

The shape for the normal sequence becomes (100, 75, 29). Where for 100 unique patients, 
there are 75 each time-stamp(charttime) and each time there are report for 29 unique lab
tests. 

### Introduction to the LSTM Autoencoder:

An LSTM Autoencoder is a type of neural network architecture specifically designed for 
reconstructing sequences. It combines the strengths of Long Short-Term Memory (LSTM) 
networks, which excel at capturing long-term dependencies in sequential data, with an 
autoencoder framework, where: 
1. The **Encoder** compresses the input sequence into a lower-dimensional latent 
representation, capturing the most significant patterns. 
2. The **Decoder** reconstructs the sequence from this compressed representation. 
This process enables the model to learn the normal behavior of the data during training, as 
it minimizes the reconstruction error for normal sequences. 

**Input to the Model:** 
- Each input to the LSTM Autoencoder is a time-series segment corresponding to a 
patient's lab test data. The sequence length represents multiple time stamps (e.g., 
75-time stamps), and the features represent different lab tests (e.g., 29 features). 
- The input shape is (batch_size, sequence_length, num_features).
 
**Encoder:** 
- The LSTM layers in the encoder process the input sequence and compress it into a 
latent representation. This representation captures the essence of the sequence 
while discarding redundant information.

**Decoder:** 
- The decoder reconstructs the original sequence from the latent representation. The 
reconstruction is compared to the original input to calculate the reconstruction 
loss.

**Training:** 
- The model is trained on normal data only, ensuring that it learns to reconstruct 
sequences that follow normal patterns.

**Testing and Anomaly Detection:** 
- When the model encounters a sequence with anomalous patterns during testing, it 
struggles to reconstruct it accurately, leading to a higher reconstruction loss. This 
increase in loss serves as an indicator of anomalies.

### Data Splitting: 
I splitted the data as 70:30, i.e. there are (70, 75, 29) samples are for the training and also I 
split the validation data into also 0.33 ratio as the validation set is of (20, 75, 29) and my 
test set is for (10, 75, 29). 


### Encoder Model: 
The LSTM Autoencoder implemented in this project consists of two main components: an 
Encoder and a Decoder. These modules work together to compress the input sequence 
into a latent representation and then reconstruct the sequence from it. The architecture 
leverages LSTM networks to handle the temporal dependencies in time-series data. 
In my encoder the input shape is (batch_size, seq_len, n_features), here I passed the total 
patients set once and the batch is 1. So, the size is same as the size of input data. 
I have used two LSTM layer –  
**First LSTM Layer:** 
- Maps the input to a hidden dimension of size 2 * embedding_dim.
- Produces an output sequence of shape (batch_size, seq_len, hidden_dim). 

**Second LSTM Layer:**
- Further compresses the sequence into a latent space of size embedding_dim.
- Outputs the final hidden state hidden_n of shape (1, batch_size, embedding_dim). 
The final hidden state (hidden_n) is permuted to have a shape of (batch_size, 1, 
embedding_dim) and returned as the latent representation.


### Decoder Model: 
The decoder takes the latent representation as input with a shape of (batch_size, 1, 
embedding_dim). 
I also used two layers for the decoder – 

**First LSTM Layer:**
- Processes the repeated latent representation and outputs a sequence of shape 
(batch_size, seq_len, input_dim).

**Second LSTM Layer:**
- Further transforms the sequence and produces an output of shape (batch_size, 
seq_len, hidden_dim).

The output layer becomes, A fully connected layer maps the LSTM output to the original 
feature space, producing a reconstructed sequence of shape (batch_size, seq_len, 
n_features). 

During training, the model learns to minimize the reconstruction error for normal 
sequences. 

During testing, anomalous sequences result in higher reconstruction losses because the 
model cannot effectively reconstruct patterns it has not seen before. 

Bayesian explainability techniques are applied post-anomaly detection to attribute 
anomalies to specific features and time stamps.


### Loss Diagram in epochs:

After running the model 1000 epochs and with a initial embedding dimension of 64, we are 
getting the training and the validation loss less, as -  

![fh](https://github.com/RitwikGanguly/Bayesian-Explanability-in-Anomaly-Detection/blob/main/data_snaps/anomaly_loss_function.png)

The loss curve suggests that the model is well-trained and balanced. The consistent 
behavior between training and validation losses indicates that the model captures the 
underlying patterns in the data without overfitting or underfitting.

The minimal gap between the losses shows that the model's performance is consistent on 
unseen data, which is crucial for anomaly detection tasks in medical diagnostics.

### Distribution of Normal Losses:

![jk](https://github.com/RitwikGanguly/Bayesian-Explanability-in-Anomaly-Detection/blob/main/data_snaps/anomaly_loss_curve_normal.png)

From this distribution, we can identify the Threshold, that will be applied to the unseen 
data and to be decided the data is recorded as anomaly or not.

![rty](https://github.com/RitwikGanguly/Bayesian-Explanability-in-Anomaly-Detection/blob/main/data_snaps/anomaly_reconstruction_loss.png)

I was selecting the anomaly in this process 0.01548 as, 99.99% of the normal (non
anomalous) data points in your dataset have a loss less than or equal to this value, or 
leaving only the top 0.01% of normal data points with a loss greater than this threshold.

 
 ## Explainable AI (XAI)

Explainable AI (XAI) refers to techniques and methods that make the decisions and predictions of AI models transparent, interpretable, and understandable to humans. It aims to address the "black-box" nature of complex models, such as deep learning networks, by providing insights into how and why a model arrived at a particular decision.

### XAI in Healthcare or Anomaly Detection
- **Building Trust**: XAI helps explain predictions, such as why a patient's condition is flagged as anomalous.
- **Accountability and Compliance**: Ensures AI systems are interpretable and justifiable, supporting decision-making.

XAI techniques can involve:
- **Local Explanations**: Focusing on specific instances.
- **Global Explanations**: Providing an overall understanding of model behavior.
- Methods like **LIME**, **SHAP**, or **Bayesian approaches** to interpret predictions.

---

### Explainable AI Domains in This Project

### **Local Explanation**
- **Definition**: Focuses on understanding the decision-making process of the model for a specific data instance (e.g., a single patient's timestamp or sample).
- **Purpose**: Answers the question, "Why did the model make this particular prediction?"
- **Example in Context**: Using LIME or BayesLIME to identify which lab results contributed most to an anomaly at a specific time.

---

### **Global Explanation**
- **Definition**: Provides an overview of how the model behaves across the entire dataset or domain.
- **Purpose**: Answers the question, "What patterns or rules does the model use to make decisions overall?"
- **Example in Context**: Understanding how lab features, such as blood pressure or heart rate, influence anomaly detection for all patients.

---

### **Model-Agnostic Explanation**
- **Definition**: Explanations that are independent of the underlying model architecture and can be applied to any type of model (e.g., LSTM Autoencoders, Random Forests).
- **Purpose**: Ensures flexibility by explaining predictions without requiring access to the internal workings of the model.
- **Example in Context**: LIME is a model-agnostic technique that uses surrogate models (e.g., linear regression) to approximate and explain the predictions of an LSTM autoencoder.

---

### **Post-Hoc Explanation**
- **Definition**: Explanations generated after the model has been trained to interpret and justify its predictions.
- **Purpose**: Focuses on making a trained model interpretable without altering its structure or training process.
- **Example in Context**: Using LIME or BayesLIME to explain the reconstruction loss of an anomaly detection model after it has been trained on patient data.

---

## Focus in This Project

In this project, the focus is on **post-hoc explanations** with **local explainability**. Specifically, a comparison study is conducted between:
- **LIME**
- **Bayesian LIME (BayesLIME)**

The results section will present this comparative analysis, highlighting their effectiveness in explaining the model's predictions.


## Feature Explanation with LIME (Local Interpretable Model-Agnostic Explanations): 

LIME is used in this project to provide explanations for the decisions made by your anomaly 
detection model. Specifically, it helps identify which features (e.g., lab results, vital signs, 
etc.) contributed the most to the model's reconstruction loss for each timestamp in the 
patient's data. This is crucial in a medical diagnostics setting, where interpretability is 
essential to trust and actionable insights.

```python
from lime.lime_tabular import LimeTabularExplainer
import numpy as np

# Initialize BayesLIME Explainer
def initialize_explainer(data, feature_names):
    explainer = LimeTabularExplainer(
        data,
        feature_names=feature_names,
        mode='regression',
        discretize_continuous=True
    )
    return explainer
```

- LIME creates small random variations (perturbations) of the data sample 
(num_samples=1000), computes predictions for these, and fits a simple 
interpretable model (e.g., linear regression) to approximate the behavior of 
the complex model locally.

- The output is an explanation of how the features contributed to the 
reconstruction loss for that specific timestamp. 

- Feature Importance: For each timestamp, LIME outputs a ranked list of 
features with their corresponding importance scores (positive or negative). 
These scores quantify how much each feature influenced the model's 
prediction. 

One of the patient of ID – 9, I got this LIME feature explanation – 

![op](https://github.com/RitwikGanguly/Bayesian-Explanability-in-Anomaly-Detection/blob/main/data_snaps/lime_op_1.png)

From the above result of LIME model, I can conclude some points –  
1. The predicted value is 2.69, but for the patient for his entire 75 timestamp it is 
ranging from the 0.88 to the 33.76. So, I can say, that the samples of the 
patient to this timestamp have a little anomaly score and mostly it is closely 
to the normal behavior but has the Anomaly nature overall. 
2. Some key insight from the chart above, the ‘blue’ bar decreases the sample 
be an anomaly nature and the ‘orange’ bar make the data sample to be 
anomaly. 
3. “Alanine Aminotransferase (ALT) Blood Chemistry <= 13.76: -15.0817” – 
significance is ALT levels below or equal to 13.76 significantly decrease the 
anomaly score. Indicates that this lab value is strongly aligned with 
normal/expected levels. So, it has negative score for the anomaly detection. 
4. Similarly, “Chloride Blood Chemistry > 0.63: 0.3899”: Chloride levels above 
this threshold slightly increase the anomaly score, and thus it has the little 
positive anomaly score.

---


## Bayesian View (BayesLIME): 

BayesLIME is an extension of the LIME (Local Interpretable Model-Agnostic Explanations) framework that incorporates Bayesian Ridge Regression to account for uncertainty in feature importance estimates. While traditional LIME provides feature importance values for each input sample using linear regression on perturbations, BayesLIME adds a probabilistic layer to the process, allowing us to quantify the uncertainty of feature weights using posterior distributions.

### Why Bayesian Methods?

Bayesian methods provide not only the point estimates (mean values of the coefficients) but also the associated uncertainties (variance). These uncertainties help:

1. **Improve Trust in Explanations**: Highlight features where the model is confident or uncertain.  
2. **Assess Stability**: Ensure robustness of explanations for different perturbations.  
3. **Model Credible Intervals**: Capture the variability in predictions due to sampling or model uncertainty.

---

### Key Advantages of BayesLIME

### 1. Uncertainty Estimates
- BayesLIME generates posterior distributions for feature weights, providing confidence intervals for each feature's importance.
- This ensures that decision-makers can trust the most influential features with high certainty.

### 2. 95% Credible Intervals
- For each feature, a credible interval (e.g., mean ± 1.96 × posterior standard deviation) is computed, indicating the range within which the true weight lies with high probability.

### 3. Robustness to Noise
- Bayesian Ridge Regression ensures that outliers or noisy perturbations have a lower impact on the model explanation.

---

## BAYESLIME Implementation in the Project

### Perturbing Data Samples
- **Gaussian Noise**: Gaussian noise is added to the original instance (`data_row`) to create perturbed samples for the explanation.
- **Noise Scale**: The scale of the noise is calculated from the training data's standard deviation, ensuring realistic perturbations.

### Predicting Outputs for Perturbed Samples
- The **prediction function** (`predict_fn`) is used to compute the model's outputs for the perturbed samples.
- These outputs serve as the target values for regression.

### Bayesian Ridge Regression
- Instead of traditional linear regression (as in LIME), **Bayesian Ridge Regression** is applied to fit the perturbed samples to their predictions.
- **Key Outputs from Bayesian Regression**:
  - **Feature Weights**: Point estimates for each feature's importance.
  - **Posterior Standard Deviations**: Quantifies uncertainty for each feature's weight.
  - **Posterior Covariance Matrix**: Tracks correlations between features.

### Generating Explanations
- Alongside Bayesian regression, the **traditional LIME explanation** is generated to retain interpretability.
- **Credible Intervals** are computed for feature importance weights using the formula:

95% Credible Interval = $\left[ (\text{mean} - 1.96 \times \text{std}), (\text{mean} + 1.96 \times \text{std}) \right]$


## BayesLIME Result: 

BayesLIME implementation provides feature importance weights along with 95% credible 
intervals (CI) for each feature. 
The weight for each feature represents its estimated importance in the model's decision
making process. Positive weights indicate a positive correlation with the predicted 
outcome, and negative weights indicate a negative correlation.

![uij](https://github.com/RitwikGanguly/Bayesian-Explanability-in-Anomaly-Detection/blob/main/data_snaps/bayeslime_op.png)

As the explanation of this output from the BayesLIME,  

- Alkaline Phosphatase Blood Chemistry has a weight of 0.3332, which means this 
feature is positively correlated with the anomaly detection prediction.

- Bicarbonate Blood Chemistry has a weight of -0.0685, which means this feature has 
a negative correlation with the anomaly detection prediction.
 
For a single patient, I have shown the all timestamp explanation through BayesLIME. 
If I see the reports of a patient (subject_id = 9), then the following conclusion, I can say -  
1. Alanine Aminotransferase (ALT): This test is highly important for most of the 
timestams, with weights ranging from 1.57 to 1.69, suggesting its strong influence 
on the model predictions. 
2. Bilirubin, Total: Similarly, bilirubin levels show high importance, with weights 
ranging from 0.22 to 0.73, further confirming its significant role across patients. 
3. Alkaline Phosphatase: This is moderately important for most patients, with weights 
between 0.13 and 0.33, indicating its role but not as dominant as ALT or bilirubin. 
4. Creatinine: This test is also important across all patients, with weights ranging from 
0.13 to 0.18, showing a significant impact on predictions. 
5. White Blood Cells: This variable is crucial for several patients, especially for patient 
2, where its weight is 0.35, suggesting its importance in differentiating outcomes.

Other blood chemistry and hematology variables like Glucose, Urea Nitrogen, and 
Calcium also appear in the list with varying degrees of importance, but they generally 
exhibit lower weights compared to ALT, bilirubin, and creatinine.


### Data Visualization of Credible Interval:

![dfgb](https://github.com/RitwikGanguly/Bayesian-Explanability-in-Anomaly-Detection/blob/main/data_snaps/final_plot.png)


This plot define the 95% credible interval for the feature importance, defining the 
importance of the feature in the anomaly model output. 

Where the line representing the 95% credible intervals for each feature's importance 
indicate the range of plausible values for that feature's contribution or effect size. 

Specifically: 

- The left end of the horizontal line represents the lower bound of the 95% credible 
interval. 
- The right end of the horizontal line represents the upper bound of the 95% credible 
interval.

The values at the positive x axis are responsible for the sample be anomalous and the 
values or features at the negative x axis are the negative correlated values for the sample be 
anomalous.


## How BayesLIME is Better than LIME in Explainability:

# Comparison of LIME and BayesLIME

| **Aspect**         | **LIME**                              | **BayesLIME**                                      |
|---------------------|---------------------------------------|---------------------------------------------------|
| **Feature Weights** | Point estimates only               | Point estimates + uncertainty (posterior std)   |
| **Uncertainty**     | None                               | Posterior std + credible intervals              |
| **Robustness**      | Sensitive to noisy data            | More robust to noise via Bayesian Ridge         |
| **Interpretability**| Linear explanation with feature ranks    | Linear explanation + confidence intervals       |
|                     |                                 |                                                   |

## Why BayesLIME is Better: 

1. **Uncertainty Quantification:** Adds a probabilistic layer to traditional LIME, making 
explanations more reliable. 
2. **Credible Intervals:** Helps stakeholders trust the most significant features by 
showing confidence ranges. 
3. **Robustness:** Handles noisy or imbalanced data better than LIME. 













