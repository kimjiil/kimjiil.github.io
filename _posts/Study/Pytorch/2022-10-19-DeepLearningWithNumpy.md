---
title: "DeepLearning Numpy로 구현해보기"
tags:
  - Pytorch
  - Numpy
categories:
  - Pytorch Study
date: 2022-10-19
toc: true
toc_sticky: true
toc_icon: "sticky-note"
use_math: true
last_modified_at: 2022-10-21T10:13:00
---

- 전체적인 구조는 코드 문법등은 Pytorch와 유사하게 하려고함.
- numpy에서 그치지않고 GPU에 data를 올려 연산하는 과정까지 구현
  - GPU knernel을 사용하는 언어로 Numba, cupy, cuda-python 3가지중 하나를 선택


#### Numba, cupy, Numpy Matrix Multiplication Calc Time

- Numba(GPU, CPU), Numpy의 속도가 비슷하고 cupy가 10배 빨라서 cupy 사용

<details>
<summary> 결과 코드 펼치기 </summary>
<div markdown="1">

```python
from numba import cuda, float32
import cupy
import numpy as np
import math
import time
```


```python
TPB = 16
@cuda.jit
def fast_matmul(A, B, C):
    """
    Perform matrix multiplication of C = A * B
    Each thread computes one element of the result matrix C
    """

    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(int(A.shape[1] / TPB)):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp
```

```python
N = 1024
a = np.random.randn(N, N).astype(np.float32)
b = a.T.copy()
c = np.zeros((N, N), dtype=np.float32)

threadsperblock = (16, 16)
blockspergrid_x = math.ceil(a.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(a.shape[1] / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)

a_d = cuda.to_device(a)
b_d = cuda.to_device(b)
c_d = cuda.to_device(c)

gfcalc = lambda t0, t1: round(t1-t0, 5)
```

```python
for i in range(0, 10):
  t0 = time.time()
  fast_matmul[blockspergrid, threadsperblock](a, b, c)
  cuda.synchronize()
  t1 = time.time()
  print("Numba CPU calc Time", gfcalc(t0, t1))
```

    Numba CPU calc Time 0.54381
    Numba CPU calc Time 0.062
    Numba CPU calc Time 0.061
    Numba CPU calc Time 0.04901
    Numba CPU calc Time 0.04642
    Numba CPU calc Time 0.04843
    Numba CPU calc Time 0.046
    Numba CPU calc Time 0.04922
    Numba CPU calc Time 0.04578
    Numba CPU calc Time 0.048


```python
for i in range(0, 10):
  t0 = time.time()
  fast_matmul[blockspergrid, threadsperblock](a_d, b_d, c_d)
  cuda.synchronize()
  t1 = time.time()
  print("Numba GPU calc Time", gfcalc(t0, t1))
```

    Numba GPU calc Time 0.06
    Numba GPU calc Time 0.056
    Numba GPU calc Time 0.05
    Numba GPU calc Time 0.041
    Numba GPU calc Time 0.04104
    Numba GPU calc Time 0.04101
    Numba GPU calc Time 0.03995
    Numba GPU calc Time 0.041
    Numba GPU calc Time 0.041
    Numba GPU calc Time 0.041


```python
for i in range(0, 10):
  t0 = time.time()
  c_h = a.dot(b)
  t1 = time.time()
  print("Numpy calc Time", gfcalc(t0, t1))
```

    Numpy calc Time 0.05517
    Numpy calc Time 0.008
    Numpy calc Time 0.01207
    Numpy calc Time 0.01093
    Numpy calc Time 0.01001
    Numpy calc Time 0.011
    Numpy calc Time 0.01099
    Numpy calc Time 0.01003
    Numpy calc Time 0.01097
    Numpy calc Time 0.01107
    

```python
with cupy.cuda.Device(0) as dev:
  a_c = cupy.asarray(a)
  b_c = cupy.asarray(b)
  c_c = cupy.asarray(c)
  for i in range(0, 10):
    t0 = time.time()
    a_c.dot(b_c, out=c_c)
    dev.synchronize()
    t1 = time.time()
    print("cupy GPU calc Time", gfcalc(t0, t1))
```

    cupy GPU calc Time 1.02841
    cupy GPU calc Time 0.002
    cupy GPU calc Time 0.00209
    cupy GPU calc Time 0.00091
    cupy GPU calc Time 0.002
    cupy GPU calc Time 0.001
    cupy GPU calc Time 0.002
    cupy GPU calc Time 0.001
    cupy GPU calc Time 0.002
    cupy GPU calc Time 0.001

</div>
</details>


##### pytoch base code
```python
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

my_train_dataset = Dataset(root='./')
my_valid_dataset = Dataset(root='./')
my_train_datalaoder = DataLoader(Dataset=my_train_dataset, transform=transform, shuffle=True, batch_size=10)
my_valid_dataloader = DataLoader(Dataset=my_valid_dataset, transform=transform, shuffle=False, batch_size=10)


model = Model()

if pretrained:
    model.load_statd_dict(pretrained)

optimizer = AdamW(model.parameters(), lr=0.001)
criterion = MSE()

model.to(device)
criterion.to(device)

for i in range(epoch):
    model.train()
    for _batch in my_train_datalaoder:
        optimizer.zero_grad()
        output = model(_batch)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

    model.eval()
    for _batch in my_valid_dataloader:
        output = model(_batch)
```

#### Objectives

##### Layer 추가하기

- [ ] Conv2d
- [ ] Linear
- [ ] BatchNormalization
- [ ] ReLu
- [ ] MaxPool
- [ ] AvgPool
- [ ] Flatten
- [ ] Sigmoid

##### Opimizer 추가하기

- [ ] Adam
- [ ] AdamW
- [ ] Stochastic Gradient Descent


##### Loss 추가하기

- [ ] Mean Square Error
- [ ] Cross Entropy 






#### Reference 

[https://github.com/SkalskiP/ILearnDeepLearning.py](https://github.com/SkalskiP/ILearnDeepLearning.py
)










