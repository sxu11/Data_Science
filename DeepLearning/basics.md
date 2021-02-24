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

  - Newton's Method optimizer:
    - https://towardsdatascience.com/optimization-loss-function-under-the-hood-part-ii-d20a239cde11