import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.ensemble import RandomForestClassifier

df_2=pd.read_excel('kan_sonuc_dataset.xlsx')
df_2.drop(["id"],axis=1,inplace=True)
#df_2.drop(["Unnamed: 0"],axis=1,inplace=True)kan_sonuc_dataset
df_1=df_2.copy()
#print(df_1)

#değişkenler
A=df_1.drop(["hastalık"], axis=1)
B=df_1["hastalık"]

#eğitim vr test
Atrain, Atest, Btrain, Btest=train_test_split(A,B, test_size=0.3, random_state=10)
model=RandomForestClassifier(random_state=0)
model.fit(Atrain,Btrain)
tahmin=model.predict(Atest)

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

accuracy = accuracy_score(Btest, tahmin)
print(f"Doğruluk (Accuracy): {accuracy:.2f}")
print(classification_report(Btest,tahmin))
con_mat = confusion_matrix(Btest, tahmin)
print("Confusion Matrix:\n", con_mat)
print(con_mat.shape)

# her birini özgünlk ve hassasiyet
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
print("######################################################################################")