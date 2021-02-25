Logistic Regression:
- y_pred = sigma(w^T x+b), sigma(z) = 1/(1+e^-z)
- loss_func = -1/m sum(y_true log y_pred + (1-y_true) log (1-y_pred))
- goal is to minimize loss_func (convex)

  - Gradient descent optimizer:
    - w = w - alpha * d loss_func / dw 
    - in code, use "dw" = d loss_func / dw, so w = w - alpha * "dw"
    
    Why "dz" = a - y, first "dz" = dL / dz = dL/da * da/dz
    1. dL/da = d/da [-(ylog(a)+(1-y)log(1-a))] = (a-y)/[a(1-a)]
    2. da/dz = d/dz [sigma(z)] = sigma(z) * [1-sigma(z)] = a(1-a)
    3. "dz" = a-y
    - BTW, z = w^T x + b

    ![plot](gradientDescientLogistic.png)
    
  - TODO: when there is regularization?

  - Newton's Method optimizer:
    - not update at w, but at an updated w TODO
        - essentially, root-finding, given x0
        - x_{n+1} = x_n - J'(x_n)/J''(x_n)
    - auto-varying alpha, but equals to 1/J'' TODO?? Converges faster [article](https://towardsdatascience.com/optimization-loss-function-under-the-hood-part-ii-d20a239cde11)
        
        
Activation function:
- sigmoid(z) = 1/(1+e^-z): between (0,1), only useful for output
- tanh(z) = (e^z-e^-z)/(e^z+e^-z): shifted sigmoid, always better than sigmoid for internal, because distribute around 0
- ReLU(z) = max(0,z): good because gradient is 1 for large z
- leaky ReLU(z) = max(0.01z, z): small negative when z < 0, to make gradient non-zero (practically not necessary)  

Propagation (how): 
- Forward: 
  z[1] = w[1]X + b[1], 
  A[1] = g[1] (z[1]), 
  z[2] = w[2]A + b[2],
  A[2] = g[2] (z[2]) = sigma(z[2])
- Backward:
  dz[2] = A[2] - Y,
  dw[2] = 1/m dz[2] A^T[1]
  db[2] = 1/m np.sum(dz[2], axis=1, keepdims=True)
  dz[1] = w^T[2] dz[2] g'[1] (Z[1]) (n[1],m)
  dw[1] = 1/m dz[1] X^T
  db[1] = 1/m np.sum(dz[1], axis=1, keepdims=True)
  
Random Initialization
- If set all 0, all weights will be symmetric (always same)
  
Get dimensions right
- Input X.shape (2,1), 1st layer 3 neurons: z[1].shape=(3,1), 
  - Since z[1] = w[1]X+b[1], we have w[1].shape=(3,2), b[1].shape=(3,1)
- 2nd layer 5 neurons: z[2].shape=(5,1)
  - ...