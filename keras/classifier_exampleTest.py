import numpy as np
np.random.seed(1)
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt

X = np.linspace(-1,1,200)
Y = X*0.3 +2 +np.random.normal(0,0.05,(200,))

X_train = X[:160]
X_test = X[160:]
Y_train = Y[:160]
Y_test = Y[160:]

model = Sequential(
    [
        Dense(1,input_dim=1),
    ]
)
model.compile(loss='mse', optimizer='sgd')

print("train---------")

print('Training -----------')
for step in range(301):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 == 0:
        print('train cost: ', cost)

# test
print('\nTesting ------------')
cost = model.evaluate(X_test, Y_test, batch_size=40)
print('test cost:', cost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)

# plotting the prediction
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()


