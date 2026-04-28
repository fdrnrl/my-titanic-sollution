import torch
import pandas as pd
from numpy import where
from sklearn.preprocessing import StandardScaler
from torch.nn import ReLU,Linear,Sequential,BCELoss,Sigmoid
from torch.optim import Adam

df = pd.read_csv('/Users/sasanuralskaa/Documents/newtrain2.csv')
y = torch.tensor(df['Survived'].to_numpy(),dtype=torch.float32)
args = df.iloc[:,2:]
scaler = StandardScaler()
normalized_args = pd.DataFrame(scaler.fit_transform(args))
x = torch.tensor(normalized_args.to_numpy(),dtype=torch.float32)
x_train = x[:445]
x = x[445:890]
y_train = y[:445]
y = y[445:890]

model = Sequential(Linear(4,16),ReLU(),Linear(16,32),ReLU(),Linear(32,1),Sigmoid())
loss_fn = BCELoss()
optimizer = Adam(model.parameters(), lr=0.001)

for i in range(500):
    optimizer.zero_grad()
    y_hat = model(x_train)
    loss = loss_fn(y_hat.reshape(445,1),y_train.reshape(445,1))
    loss.backward()
    optimizer.step()
test_df = pd.read_csv('/Users/sasanuralskaa/Documents/test.csv')
passenger_id = test_df["PassengerId"]
test_df = test_df.drop(['PassengerId','Name','SibSp','Ticket','Fare','Cabin','Embarked'],axis=1)
test_df['Sex'] = where(test_df['Sex'] == 'male', 0, 1)
mean_age_class_1 = round(test_df[test_df['Pclass'] == 1]['Age'].mean())
mean_age_class_2 = round(test_df[test_df['Pclass'] == 2]['Age'].mean())
mean_age_class_3 = round(test_df[test_df['Pclass'] == 3]['Age'].mean())
test_df.loc[test_df['Pclass'] == 1, ['Age']] = test_df.loc[test_df['Pclass'] == 1, ['Age']].fillna(mean_age_class_1)
test_df.loc[test_df['Pclass'] == 2, ['Age']] = test_df.loc[test_df['Pclass'] == 2, ['Age']].fillna(mean_age_class_2)
test_df.loc[test_df['Pclass'] == 3, ['Age']] = test_df.loc[test_df['Pclass'] == 3, ['Age']].fillna(mean_age_class_3)
x_test = pd.DataFrame(StandardScaler().fit_transform(test_df.iloc[::]))
y_test = model(torch.tensor(x_test.to_numpy(),dtype=torch.float32))

answer = pd.DataFrame({
    'PassengerId': passenger_id,
    'Survived': (y_test.detach().numpy().flatten() > 0.5).astype(int)
})
answer.to_csv('answer.csv', index=False)





