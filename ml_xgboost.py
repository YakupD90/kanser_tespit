import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

df_2 = pd.read_excel('kan_sonuc_dataset.xlsx')
df_2.drop(["id"], axis=1, inplace=True)

A = df_2.drop(["hastalık"], axis=1)
B = df_2["hastalık"]
label_encoder = LabelEncoder()
B_encoded = label_encoder.fit_transform(B)

Atrain, Atest, Btrain, Btest = train_test_split(A, B_encoded, test_size=0.3, random_state=10)
model_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model_xgb.fit(Atrain, Btrain)
tahmin = model_xgb.predict(Atest)

#str yap
Btest_labels = label_encoder.inverse_transform(Btest)
tahmin_labels = label_encoder.inverse_transform(tahmin)

print(classification_report(Btest_labels, tahmin_labels, target_names=label_encoder.classes_))
accuracy = accuracy_score(Btest, tahmin)
print(f"Doğruluk (Accuracy): {accuracy:.2f}")

con_mat=confusion_matrix(Btest,tahmin)
print(con_mat)

n_classes = con_mat.shape[0]
for i in range(n_classes):
   if i==0:tür="Breast Cancer"
   if i==1:tür="Colon Cancer"
   if i==2:tür="Lung Canser"
   if i==3:tür="Prostate Canser"
   tp = con_mat[i, i]
   fn = np.sum(con_mat[i, :]) - tp
   fp = np.sum(con_mat[:, i]) - tp
   tn = np.sum(con_mat) - (tp + fn + fp)
    
   sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
   specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    
   print(f"Sınıf {tür}:")
   print(f"Duyarlılık (Sensitivity): {sensitivity}")
   print(f"Özgüllük (Specificity): {specificity}")
   print()