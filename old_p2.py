
df = pd.read_csv("scoring_train_small.csv", header=None)
#y = df.iloc[:, 9].values
y = df.iloc[:, 5].values

y = np.where(y == 'O', 0, 1)
counter = 0
for i in y:
    if i == 1:
        counter +=1
print(counter)
#X = df.iloc[:, [0,1,2,3,4,5,6,7,8]].values
depp = []
for x in range(0,5):
    depp.append(x)
#print(depp)
X = df.iloc[:, depp].values

lr = LogisticRegression(n_iter=500, eta=0.2).fit(X, y)
plt.plot(range(1, len(lr.cost_) + 1), np.log10(lr.cost_))
#print(lr.cost_)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Logistic Regression - Learning rate 0.01')

plt.tight_layout()
plt.show()

df2 = pd.read_csv("/home/dominik/projects_save/scoring_testa2_ex_small.csv", header=None)
X2 = df2.iloc[:, [0,1,2,3,4]].values
y_ref = df2.iloc[:, 5].values

y_ref = np.where(y_ref == 'O', 0, 1)
y_pred = lr.predict(X2)
# counter = 0
# for i in y_pred:
#     if i == 1:
#         counter +=1
print (y_pred)
print(counter)