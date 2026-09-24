CUSTOMER CHURN PREDICTION USING ARTIFICIAL NEURAL NETWORKS (ANN)

Author : Sanyukta Saha, MSc Statistics and Computing (BHU)


1. OVERVIEW
------------------------------------------------------------------------------
Customer churn is when a customer stops using a company's products or
services. In subscription-driven industries such as banking, telecom and
SaaS, churn directly reduces revenue, and acquiring a new customer usually
costs far more than retaining an existing one.

This project builds an Artificial Neural Network (ANN) that predicts whether a
bank customer will churn (Exited = 1) or stay (Exited = 0), using demographic,
account and behavioural features. Three ANN training strategies are compared
on an imbalanced dataset (about 20% churn), and the best candidate is tuned
with a decision threshold chosen on validation data and saved for reuse.


2. OBJECTIVES
------------------------------------------------------------------------------
- Explore the data (EDA) and understand customer behaviour patterns.
- Clean and preprocess the data: drop identifiers, encode categoricals,
  scale numeric features, engineer new features.
- Handle class imbalance (class weighting and SMOTE).
- Build and compare ANN variants for churn prediction.
- Evaluate with Accuracy, Precision, Recall, F1-Score and ROC-AUC.
- Identify the factors that most influence churn.
- Save a reusable predictive pipeline for identifying high-risk customers.


3. DATASET
------------------------------------------------------------------------------
File     : Churn_Modelling.csv (place it in the project root folder)
Size     : 10,000 customer records, 14 columns
Target   : Exited (1 = churned, 0 = retained)
Balance  : 7,963 retained / 2,037 churned (churn rate = 20.4%)

Columns  : RowNumber, CustomerId, Surname, CreditScore, Geography, Gender,
           Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember,
           EstimatedSalary, Exited

The dataset is not bundled with this repository. Download it separately and
save it as Churn_Modelling.csv next to the notebook.


4. PROJECT WORKFLOW
------------------------------------------------------------------------------
1) Exploratory analysis
   - Class balance, churn rate by Geography and by number of products,
     and a correlation heatmap of the numeric features.
   - Germany shows the highest churn rate; customers with 3 or 4 products
     churn far more than those with 1 or 2.
   - Numeric features are only weakly correlated with each other.

2) Cleaning and feature engineering
   - Dropped RowNumber, CustomerId and Surname (identifiers, no signal).
   - Engineered features:
       BalanceSalaryRatio   = Balance / (EstimatedSalary + 1)
       TenureByAge          = Tenure / (Age + 1)
       CreditScoreGivenAge  = CreditScore / (Age + 1)
       IsSeniorCustomer     = 1 if Age > 60 else 0
       ZeroBalance          = 1 if Balance == 0 else 0
   - One-hot encoded Geography and Gender (drop_first=True).
   - Final feature matrix: 10,000 rows x 16 features.

3) Train / validation / test split (stratified, random_state = 42)
   - Train: 6,400   Validation: 1,600   Test: 2,000
   - Churn rate is 0.204 in all three splits.

4) Scaling and resampling
   - StandardScaler fitted on the training set only, then applied to the
     validation and test sets (no leakage).
   - SMOTE (k_neighbors = 5) applied to the training set only:
     5,096 / 1,304 -> 5,096 / 5,096. Validation and test sets stay untouched.

5) Model training - three ANNs with the same architecture
   - Basic ANN         : no imbalance handling
   - Class Weight ANN  : class 1 weighted about 3.91x
   - SMOTE ANN         : trained on the SMOTE-balanced training data

6) Evaluation
   - ROC-AUC, Accuracy, Precision, Recall, F1, confusion matrix and
     classification report on the test set.

7) Feature importance
   - Random Forest (300 trees, class_weight = "balanced") importances.

8) Threshold tuning and finalisation
   - The SMOTE ANN is designated as the production model.
   - The decision threshold is chosen to maximise F1 on the validation set
     (best threshold = 0.59), then evaluated once on the test set.

9) Saving artifacts to the models/ folder.


5. MODEL ARCHITECTURE
------------------------------------------------------------------------------
Input (16 features)
  -> Dense(64, ReLU) -> BatchNormalization -> Dropout(0.3)
  -> Dense(32, ReLU) -> BatchNormalization -> Dropout(0.3)
  -> Dense(16, ReLU) -> Dropout(0.2)
  -> Dense(1, Sigmoid)

Optimizer : Adam (learning rate 0.001)
Loss      : Binary cross-entropy
Metrics   : AUC, Accuracy
Training  : up to 150 epochs, batch size 32
Callbacks : EarlyStopping (monitor val_auc, patience 15,
            restore best weights)
            ReduceLROnPlateau (monitor val_auc, factor 0.5, patience 6,
            min_lr 1e-6)


6. RESULTS (TEST SET, 2,000 CUSTOMERS)
------------------------------------------------------------------------------
Model                          ROC-AUC  Accuracy  Precision  Recall     F1
-----------------------------  -------  --------  ---------  ------  ------
ANN - Basic (threshold 0.50)    0.8609    0.8625     0.8235  0.4128  0.5499
ANN - Class Weight (0.50)       0.8621    0.7855     0.4831  0.7740  0.5949
ANN - SMOTE (0.50)              0.8548    0.7820     0.4778  0.7666  0.5887
ANN - SMOTE (tuned, 0.59)       0.8548    0.8160     0.5371  0.6929  0.6052

Confusion matrix of the final model (tuned threshold 0.59):

                      Predicted retained   Predicted churn
  Actual retained             1350                243
  Actual churn                 125                282

Key takeaways:
- All models reach a ROC-AUC of about 0.85-0.86, so they rank customers by
  churn risk almost equally well. The differences come from where the
  decision threshold sits.
- The Basic ANN looks accurate (86%) but catches only about 41% of actual
  churners.
- Class weighting and SMOTE raise recall to about 77% at the cost of
  precision and accuracy, which suits churn use cases where a missed churner
  usually costs more than an unnecessary retention offer.
- Tuning the threshold on validation data gave the best F1 (0.6052) and a
  better precision/recall balance.

Top Random Forest feature importances:
  Age (0.169), CreditScoreGivenAge (0.139), NumOfProducts (0.119),
  CreditScore (0.089), EstimatedSalary (0.087), Balance (0.084),
  TenureByAge (0.082), BalanceSalaryRatio (0.070), Tenure (0.044),
  IsActiveMember (0.037)

7. LIMITATIONS AND FUTURE WORK
------------------------------------------------------------------------------
- Try class weighting and SMOTE together, and focal loss.
- Add precision-recall curves and check probability calibration
  (SMOTE-trained models tend to shift predicted probabilities).
- Repeat training across several random seeds to see whether the differences
  between the three ANN variants are larger than run-to-run noise.
- Replace pd.get_dummies with ColumnTransformer + OneHotEncoder
  (handle_unknown="ignore") so unseen categories do not break feature
  alignment in production.
- Add a predict_customer(raw_record) function that runs a single new record
  end to end.
- Choose the deployment threshold with a business cost matrix (cost of a
  missed churner vs. cost of an unnecessary retention offer) instead of F1
  alone.
- Compare against tree-based ensembles such as Random Forest or gradient
  boosting as full predictive models.


8. CONCLUSION
------------------------------------------------------------------------------
The project shows how data preprocessing, feature engineering, imbalance
handling and threshold tuning affect a churn model far more than the choice
of headline metric alone. By flagging at-risk customers early, a business can
target retention offers, improve satisfaction and reduce revenue lost to
attrition.
