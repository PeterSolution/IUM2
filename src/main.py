from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_wine
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
from model import Model
from torch.optim import Adam
import numpy as np
from sklearn.model_selection import KFold, cross_val_score

# zad1
wineDataset=load_wine()
x=wineDataset.data
y=wineDataset.target

x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.2,random_state=42
)

print(f"x_train shape: {x_train.shape}")
print(f"y train shape:{y_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y test shape:{y_test.shape}")

# zad2
model = Pipeline([

    ("scaler",StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

model.fit(x_train,y_train)
yPrediction=model.predict(x_test)
accuracy=accuracy_score(y_test,yPrediction)
print(f"Accuracy: {accuracy}")

# zad3
torch.manual_seed(42)

scaler = StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


x_train_tensor=torch.tensor(
    x_train, dtype=torch.float32
)

x_test_tensor=torch.tensor(
    x_test, dtype=torch.float32
)

y_train_tensor=torch.tensor(
    y_train, dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test, dtype=torch.float32
)

input=x_train_tensor.shape[1]
Size=16
output=len(wineDataset.target_names)
torchModel=Model(input,Size,output)
criterion=torch.nn.CrossEntropyLoss()
optimizer=Adam(torchModel.parameters(),lr=0.001)

torchModel.train()
for i in range(1000):
    optimizer.zero_grad()
    y_pred=torchModel(x_train_tensor)
    loss=criterion(y_pred,y_train_tensor.long())
    loss.backward()
    optimizer.step()
    if((i)%100==0 or i==999):
        print(f"Epoch {i}, Loss: {loss.item()}")

torchModel.eval()

with torch.no_grad():
    output=torchModel(x_test_tensor)
    prediction=torch.argmax(output,dim=1).numpy()
    accuracy=accuracy_score(y_test,prediction)
    print(f"Accuracy: {accuracy}")

# zad4
kfold=KFold(
    n_splits=5, shuffle=True, random_state=42
)

score=cross_val_score(model, x, y, cv=kfold, scoring="accuracy")

for i,score in enumerate(score):
    print(f"Fold {i+1}: Accuracy: {score}")

mean=np.mean(score)
std=np.std(score)

print(f"accuracy srednie: {mean}")
print(f"odchyenie standardowe: {std}")

# zad5
input_size=x.shape[1]
hidden_size=16
output_size=len(np.unique(y))
epoch=100
learning_rate=0.001

score=[

]
#for fold, (train)
# do dokonczenia

import time

scores_torch=[]
start_torch=time.time()

for fold,(train_idx,test_idx) in enumerate(kfold.split(x)):
    x_tr,x_te=x[train_idx],x[test_idx]
    y_tr,y_te=y[train_idx],y[test_idx]

    # skalowanie osobno w kazdym foldzie (scaler uczony tylko na train)
    fold_scaler=StandardScaler()
    x_tr=fold_scaler.fit_transform(x_tr)
    x_te=fold_scaler.transform(x_te)

    x_tr_t=torch.tensor(x_tr,dtype=torch.float32)
    x_te_t=torch.tensor(x_te,dtype=torch.float32)
    y_tr_t=torch.tensor(y_tr,dtype=torch.long)

    # nowa siec w kazdym foldzie, zeby nie trenowac dalej tych samych wag
    torch.manual_seed(42)
    foldModel=Model(input_size,hidden_size,output_size)
    fold_criterion=torch.nn.CrossEntropyLoss()
    fold_optimizer=Adam(foldModel.parameters(),lr=learning_rate)

    foldModel.train()
    for i in range(epoch):
        fold_optimizer.zero_grad()
        fold_loss=fold_criterion(foldModel(x_tr_t),y_tr_t)
        fold_loss.backward()
        fold_optimizer.step()

    foldModel.eval()
    with torch.no_grad():
        fold_pred=torch.argmax(foldModel(x_te_t),dim=1).numpy()
    fold_acc=accuracy_score(y_te,fold_pred)
    scores_torch.append(fold_acc)
    print(f"Fold {fold+1}: Accuracy: {fold_acc}")

time_torch=time.time()-start_torch
print(f"PyTorch accuracy srednie: {np.mean(scores_torch)}")
print(f"PyTorch odchylenie standardowe: {np.std(scores_torch)}")

# zad6
start_sk=time.time()
scores_sk=cross_val_score(model,x,y,cv=kfold,scoring="accuracy")
time_sk=time.time()-start_sk

sk_mean,sk_std=np.mean(scores_sk),np.std(scores_sk)
t_mean,t_std=np.mean(scores_torch),np.std(scores_torch)

if sk_mean>t_mean:
    wniosek="Lepszy okazal sie scikit-learn (wyzsza srednia accuracy)."
elif t_mean>sk_mean:
    wniosek="Lepszy okazal sie PyTorch (wyzsza srednia accuracy)."
else:
    wniosek="Oba modele maja taka sama srednia accuracy."

tekst=(
    "Porownanie modeli na zbiorze wine (5-fold CV)\n\n"
    f"{'Model':<24}{'Srednia acc':<14}{'Odch. std':<12}{'Czas [s]':<10}\n"
    f"{'scikit-learn (LogReg)':<24}{sk_mean:<14.4f}{sk_std:<12.4f}{time_sk:<10.2f}\n"
    f"{'PyTorch (siec FC)':<24}{t_mean:<14.4f}{t_std:<12.4f}{time_torch:<10.2f}\n\n"
    f"Wniosek: {wniosek}\n"
)
print(tekst)

with open("wyniki.txt","w",encoding="utf-8") as f:
    f.write(tekst)