We are working only with 34 nodes. Span Wagner Results were verified by explicitly writing down the coefficients. Look at main_fo_verification.py and model.py for more details.

--Update on 10 March :- 8 pm
We have fit the model 1000 epochs, 0 reg and 2500 data points. R2 test = 1, R2 train = 1
And we have run 7 different regularization cases for L0.75 norm.

# Define hyperparameters
learning_rate = 0.0001
batch_size = 64
num_epochs = 1000
regularization_term = 5e-3
Lp = 0.75

Epoch 1000/1000, Train Loss: 0.0097
Epoch 1000/1000, Train R2 Score: 0.3992
Test Loss MSE: 0.0089
R2 Score: 0.4985
Layer: input_layer.weight, Weights: tensor([[-8.2266e-07,  7.0520e-05, -3.5082e-01, -2.3635e-05, -4.1708e-05,
          2.5367e-05,  1.3509e-04, -1.2146e-05,  2.5989e-05,  3.5146e-06,
          7.3583e-05,  5.2992e-05, -4.8770e-06,  3.4320e-05,  3.0651e-05,
         -1.4638e-06, -5.9765e-05, -1.0855e-05,  3.5511e-05,  2.5443e-05,
          1.5072e-05,  5.0289e-05,  3.8584e-05, -4.8967e-06, -2.3144e-05,
         -1.5553e-05, -2.1695e-05,  3.2582e-05, -2.0551e-05,  5.7219e-06,
         -3.6365e-06,  2.6451e-05, -1.5321e-04, -9.8235e-06]])

Number of terms = 1


# Define hyperparameters
learning_rate = 0.0001
batch_size = 64
num_epochs = 1000
regularization_term = 1e-3
Lp = 0.75

Epoch 1000/1000, Train Loss: 0.0043
Epoch 1000/1000, Train R2 Score: 0.9456
Test Loss MSE: 0.0042
R2 Score: 0.9537
Layer: input_layer.weight, Weights: tensor([[ 3.9198e-01, -3.5715e-05, -7.4480e-01,  2.7233e-05,  6.7822e-05,
         -3.4416e-01,  8.8069e-05, -4.5918e-01, -2.6714e-05, -1.3784e-05,
          1.7338e-01, -1.8585e-05,  5.1313e-05,  3.4054e-05,  8.4613e-06,
         -1.9307e-05,  1.5865e-06,  2.6927e-05, -5.0359e-06, -3.8735e-05,
          4.4823e-06,  6.8563e-05,  1.0688e-04,  3.3600e-06, -1.2447e-05,
         -3.0894e-05, -1.2279e-05, -2.8922e-05, -2.5749e-05,  6.4085e-06,
         -1.2711e-04, -4.1631e-05, -4.6005e-05,  2.2787e-06]])

No of terms = 5


# Define hyperparameters
learning_rate = 0.0001
batch_size = 64
num_epochs = 1000
regularization_term = 5e-4
Lp = 0.75

Epoch 1000/1000, Train Loss: 0.0022
Epoch 1000/1000, Train R2 Score: 0.9844
Test Loss MSE: 0.0022
R2 Score: 0.9883
Layer: input_layer.weight, Weights: tensor([[ 3.6824e-01,  1.1173e-05, -5.9642e-01, -7.6440e-06, -2.8048e-06,
         -3.7339e-01,  5.1951e-05,  1.6647e-06, -5.2784e-05,  2.4505e-05,
          7.7844e-02,  4.7796e-05, -2.1145e-05,  2.7079e-05,  1.2402e-05,
          5.9555e-06, -1.0767e+00, -4.8910e-06,  2.3254e-05, -1.9075e-05,
         -3.1166e-05,  7.2498e-06, -4.6440e-05,  3.7348e-05, -1.2065e-05,
         -1.2727e-05,  3.4007e-05,  1.6308e-05, -1.6974e-05, -9.5906e-06,
          3.5748e-06, -1.9404e-05, -2.6818e-05, -2.9514e-06]])
No of terms = 5

# Define hyperparameters
learning_rate = 0.0001
batch_size = 64
num_epochs = 1000
regularization_term = 1e-4
Lp = 0.75

Epoch 1000/1000, Train Loss: 0.0004
Epoch 1000/1000, Train R2 Score: 0.9989
Test Loss MSE: 0.0004
R2 Score: 0.9992
Layer: input_layer.weight, Weights: tensor([[ 4.3759e-01, -6.3877e-07, -4.7316e-01, -7.5987e-01,  2.7680e-05,
         -8.5809e-02, -5.9059e-06, -3.2122e-01, -3.3443e-05,  7.4327e-05,
          9.3823e-02, -2.0938e-05,  3.5930e-05,  1.1899e-04, -3.6121e-05,
         -1.6657e-05, -1.1890e-01, -4.6832e-05,  2.9850e-05,  2.6373e-05,
         -1.4723e-05,  3.9526e-06,  3.8134e-05, -4.1991e-05, -2.7828e-05,
          3.5565e-05,  4.0759e-05,  3.5252e-05,  1.0661e-04,  1.7661e-05,
         -2.8195e-05,  1.1494e-05,  9.4577e-06,  6.7599e-05]])
No of terms = 7

# Define hyperparameters
learning_rate = 0.0001
batch_size = 64
num_epochs = 1000
regularization_term = 5e-5
Lp = 0.75

Epoch 1000/1000, Train Loss: 0.0002
Epoch 1000/1000, Train R2 Score: 0.9994
Test Loss MSE: 0.0002
R2 Score: 0.9996
Layer: input_layer.weight, Weights: tensor([[ 4.4507e-01,  4.0598e-05, -4.8730e-01, -5.4214e-01,  1.0177e-05,
         -2.2823e-01,  9.8859e-06, -4.3337e-01,  6.2082e-06, -2.2497e-05,
          1.0450e-01,  6.2633e-06, -4.2212e-05,  6.8693e-06, -5.1890e-05,
          2.4010e-05, -2.4292e-01, -4.1204e-05,  1.1600e-05,  1.0575e-05,
         -6.3086e-06, -5.0801e-05,  1.2649e-06, -1.5133e-04,  8.9616e-06,
         -1.1406e-05, -1.4238e-05, -2.0685e-05,  1.0802e-05,  3.7938e-05,
         -2.3036e-05, -6.3579e-05, -1.3089e-05, -1.3971e-05]])

No of terms = 7

# Define hyperparameters
learning_rate = 0.0001
batch_size = 64
num_epochs = 1000
regularization_term = 1e-5
Lp = 0.75


Epoch 1000/1000, Train Loss: 0.0001
Epoch 1000/1000, Train R2 Score: 0.9995
Test Loss MSE: 0.0001
R2 Score: 0.9995
Layer: input_layer.weight, Weights: tensor([[ 4.3626e-01, -1.6490e-05, -4.8536e-01, -2.8575e-01, -5.5565e-02,
         -1.9522e-01, -1.3109e-01, -4.7367e-01,  1.9666e-07, -1.6764e-01,
          3.9793e-01, -2.9677e-05, -2.9474e-05, -5.7058e-05,  4.3498e-05,
          1.5041e-05, -5.2350e-01,  2.4881e-06, -4.6996e-05,  5.4567e-06,
         -6.1110e-05,  3.0041e-05, -7.8893e-02,  1.6299e-05, -9.2570e-06,
          1.4280e-05, -1.4347e-05, -8.9267e-06,  1.1574e-04,  4.0333e-05,
         -3.7451e-05, -2.3431e-04, -7.4575e-06, -2.9416e-05]])

No. of terms = 11

# Define hyperparameters
learning_rate = 0.0001
batch_size = 64
num_epochs = 1000
regularization_term = 5e-6
Lp = 0.75

Epoch 1000/1000, Train Loss: 0.0001
Epoch 1000/1000, Train R2 Score: 0.9995
Test Loss MSE: 0.0001
R2 Score: 0.9995
Layer: input_layer.weight, Weights: tensor([[ 4.3394e-01, -8.8938e-05, -4.8085e-01, -2.5275e-01, -7.0893e-02,
         -1.7314e-01, -1.4014e-01, -5.0751e-01, -2.2130e-05, -2.4836e-01,
          4.4361e-01, -5.9418e-05,  1.1211e-01,  1.6102e-02,  5.8068e-02,
         -1.5393e-05, -5.2621e-01,  3.6666e-05, -1.2269e-01, -1.8740e-05,
         -7.6300e-05,  6.8232e-06, -1.3378e-01, -8.0308e-05,  2.3671e-05,
          3.1207e-05, -7.7982e-06,  5.0023e-05, -2.6552e-05,  1.2844e-04,
          4.0922e-05, -7.3018e-02, -1.0819e-05,  3.0873e-05]])

No of terms = 16


# Define hyperparameters
learning_rate = 0.0001
batch_size = 64
num_epochs = 1000
regularization_term = 1e-6
Lp = 0.75

Epoch 1000/1000, Train Loss: 0.0000
Epoch 1000/1000, Train R2 Score: 0.9996
Test Loss MSE: 0.0000
R2 Score: 0.9996
Layer: input_layer.weight, Weights: tensor([[ 4.1752e-01,  4.0076e-02, -4.8561e-01, -2.4124e-01, -6.6265e-02,
         -1.7681e-01, -1.4652e-01, -5.5477e-01,  9.1052e-05, -3.8004e-01,
          4.7776e-01,  8.6705e-05,  1.9555e-01,  6.0357e-02,  1.1635e-01,
         -5.5334e-02, -4.8609e-01, -3.7713e-02, -3.0518e-01,  1.2970e-02,
         -1.6639e-05,  2.1674e-01, -1.8484e-01,  1.4646e-04,  1.5954e-05,
         -4.6536e-05,  1.0122e-05, -4.6270e-05, -2.2053e-05, -8.7456e-06,
          6.1247e-01, -1.7003e-01,  7.1731e-05,  1.9680e-05]])

No of terms = 22

