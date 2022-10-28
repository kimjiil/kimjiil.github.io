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
last_modified_at: 2022-10-28T18:11:06
---

<hr/>

#### Github 링크

[https://github.com/kimjiil/DeepLearningWithNumpy](https://github.com/kimjiil/DeepLearningWithNumpy){:target="_blank"}

<hr/>

#### Objectives

##### Pytorch code

- 전체적인 구조는 코드 문법등은 Pytorch와 유사하게 하려고함.

<details>
<summary> <span style="color: #4682B4">pytorch code 펼치기/접기</span> </summary>
<div markdown="1">

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

my_train_dataset = Dataset(root='./')
my_valid_dataset = Dataset(root='./')
my_train_datalaoder = DataLoader(Dataset=my_train_dataset, transform=transform, shuffle=True, batch_size=10)
my_valid_dataloader = DataLoader(Dataset=my_valid_dataset, transform=transform, shuffle=False, batch_size=10)

class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
    
        self.layer1 = nn.Sequential(
            nn.Linear(in_feature=10, out_feature=20),
            nn.ReLU(),
            nn.BatchNorm2d()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, strides=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d()
        )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

model = myModel()

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
</div>
</details>


##### Numba, cupy, Numpy Matrix Multiplication Calc Time

- numpy에서 그치지않고 GPU에 data를 올려 연산하는 과정까지 구현
  - GPU knernel을 사용하는 언어로 Numba, cupy, cuda-python 3가지중 하나를 선택

- Numba(GPU, CPU), Numpy의 속도가 비슷하고 cupy가 빨라서 cupy 사용
- [Why are cuda gpu matrix multiplies slower than numpy? How is numpy so fast?](https://stackoverflow.com/questions/68754407/why-are-cuda-gpu-matrix-multiplies-slower-than-numpy-how-is-numpy-so-fast){:target="_blank"}


<details>
<summary><span style="color: #4682B4">nvprof으로 측정한 Numba code tack time 결과 펼치기</span></summary>
<div markdown="1">

- Numpy, Numba에서 같은 크기의 matrix multiplication calc time  

```text
==5688== NVPROF is profiling process 5688, command: python numba_test.py
<Managed Device 0>
Numpy CPU - 1.1220035552978516s

------------------------------------------------------------------------------------------------------
==5688== Profiling application: python numba_test.py
==5688== Profiling result:
   Start  Duration            Grid Size      Block Size     Regs*    SSMem*    DSMem*      Size  Throughput  SrcMemType  DstMemType           Device   Context    Stream  Name
325.82ms  21.908ms                    -               -         -         -         -  97.656MB  4.3532GB/s    Pageable      Device  NVIDIA GeForce          1         7  [CUDA memcpy HtoD]
347.94ms  397.56ms                    -               -         -         -         -  97.656MB  245.64MB/s    Pageable      Device  NVIDIA GeForce          1         7  [CUDA memcpy HtoD]
746.28ms  4.18387s          (500 500 1)       (16 16 1)        41  2.0000KB        0B         -           -           -           -  NVIDIA GeForce          1         7  cudapy::__main__::fast_matmul$241(Array<double, int=2, C, mutable, aligned>, Array<double, int=2, C, mutable, aligned>, Array<double, int=2, C, mutable, aligned>) [151]
4.93015s  4.11571s          (500 500 1)       (16 16 1)        41  2.0000KB        0B         -           -           -           -  NVIDIA GeForce          1         7  cudapy::__main__::fast_matmul$241(Array<double, int=2, C, mutable, aligned>, Array<double, int=2, C, mutable, aligned>, Array<double, int=2, C, mutable, aligned>) [154]
9.04586s  126.57ms                    -               -         -         -         -  488.28MB  3.7673GB/s      Device    Pageable  NVIDIA GeForce          1         7  [CUDA memcpy DtoH]

Regs: Number of registers used per CUDA thread. This number includes registers used internally by the CUDA driver and/or tools and can be more than what the compiler shows.
SSMem: Static shared memory allocated per CUDA block.
DSMem: Dynamic shared memory allocated per CUDA block.
SrcMemType: The type of source memory accessed by memory operation/copy
DstMemType: The type of destination memory accessed by memory operation/copy
```
</div>
</details>


<details>
<summary><span style="color: #4682B4"> Numpy, Numba, cupy tack time 결과 펼치기 </span></summary>
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


##### myModule

- pytorch는 모델을 Module로 생성하고 sub module과 현재 module이 갖고있는 parameter를 저장한다. 
- class myModule을 만들어 _modules에 sub module을 저장하고 _paramters에 parameter들을 저장한다.
- Model()에서 function call이 일어나면 하위 module들의 forward 함수가 호출되도록 함
- pytorch와 마찬가지로 .to()를 호출할 경우 모든 parameter 및 module이 설정한 device에서 돌도록 함
  - 단순히 numpy 와 cupy를 스위칭하는 형식으로 구현함.

  
##### mySequential

- mySequential class는 다음과 같이 Layer들을 argument로 받고 class를 function call할 경우 자동으로
Layer들을 순서대로 forward해준다.

```python
Layer_seq = mySequential(
                Linear(in_features=20, out_features=10, bias=True),
                ReLU(),
            )
```

- myModule class를 상속받고 argument로 들어오는 Layer들을 추가하는 코드만 `__init__` 부분에 추가 했음.

```python
class mySequential(myModule):
    def __init__(self, *args):
        super(mySequential, self).__init__()

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)
```

- 외부에서 mySequential의 forward함수를 class하면 class가 갖고 있는 `_modules`에 있는 
모듈들을 불러와 순서대로 forward 함수를 call한다.

```python
    def forward(self, x):
        module = self.__dict__['_modules']
        for module_key in module:
            x = module[module_key].forward(x)
        return x
```

- 마찬가지로 외부에서 backward 함수를 call할 경우 forward와는 반대로 모듈의 역순으로 backward 함수를 call한다.

```python
    def backward(self, *args, **kwargs):
        back = args[0]
        module = self.__dict__['_modules']
        for module_key in reversed(module):
            back = module[module_key].backward(back)

        self.backward_fn(back)
```

- 여기서 `self.backward_fn`은 다음 순서의 모듈의 backward 함수의 주소를 갖고 있고 function call할 경우 
그 다음 순서의 모듈 backward 함수가 call된다.

##### myParameter

- 이 클래스는 Layer의 weight나 bias를 저장할때 사용한다. 현재 별다른 기능은 구현되있지않고 단순히 myTensor를 상속받아 사용한다.

```python
class myParameter(myTensor):
    def __init__(self, data):
        super(myParameter, self).__init__(data)

```

##### myTensor

- pytorch에서 backward와 각종 연산을 처리하기 위해서 Tensor라는 변수를 사용한다. 여기서도 backward 및 각종 연산을 처리하기 위해
myTensor라는 class를 만들어 처리하기로 함.
- cupy, Numpy 둘다 ndarray를 사용하여 계산이 가능하기 때문에 myTensor에 ndarray를 저장하여 이를 사용해서 계산하기로 함
- ~~.grad와 .grad_fn을 attribute로 갖고 backward시 사용함. (연산자 단위로 backward를 하도록 구현하는건 시간이 너무 오래걸려서 보류)~~
- myTensor와 myModule에 `self.backward_fn`에 이전 모듈의 backward 함수를 저장함.
- Numpy와 cupy의 객체를 따로 저장하는 변수를 만들고 하위 함수를 call하는 wrapper 함수를 만듬
  - 예를 들어 cupyTensor로 만들어진 변수들을 더하기 할 경우 wrapper 함수 내부에서 numpy.add 함수를 다시 call 하는 형식

```python
a = myTensor(np.array([1, 2]))
b = myTensor(np.array([3, 4]))

c = a + b # np.ndarray([4,6])
```

- 위 코드에서 `a + b`를 실행할 경우 myTensor class의 `__add__` 함수가 call되고 `__add__` 함수는 더하기 연산이 이항 연산자이므로
`self.binary_operator_funtion_call` 함수를 `operator.add`와 `other=b`를 argument로 call한다.
- `self.binary_operator_funtion_call` 함수는 `myTensor.data`에 저장된 `ndarray`를 argument로 받은 연산자로 계산하고
이 값을 가지고 새로운 myTensor를 생성하여 연산의 결과로 반환 해준다.

```python
class myTensor(myModule):
    ...
    def binary_operator_function_call(self, operator, other):
        if isinstance(other, myTensor):
            temp_data = other.data
        else:
            temp_data = other

        _new_data = operator(self.data, temp_data)
        _new = myTensor(_new_data)
        _new.grad_fn = f"{operator}"
        _new.op = self.op
        _new.backward_fn = self.backward_prev
        _new.backward_prev = self.backward_prev
        return _new
    ...
    def __add__(self, other):
        return self.binary_operator_function_call(operator.add, other=other)
    ...
```
  
- myTensor를 cpu가 아닌 gpu에서 사용할 경우 다음과 같이 `.to()`,`.to(device="cuda:0")` 함수를 call한다. 함수 인자로 아무것도 주지 않는 경우 자동으로 gpu id 0번에 할당됨.

```python
a_cuda = myTensor(np.array([1,2])).to()
```

- `.to()` 함수가 호출될 경우 class 내부에서 cupy의 setDeivce 함수를 호출하고 numpy array에서 cupy array로 변경해준다.
- myTensor끼리의 연산에서 사용될 library도 numpy에서 cupy로 변경해준다.
- 반대로 `.to("cpu")`가 호출될 경우 cupy에서 numpy로 변경하고 data를 numpy array로 변경하고 gpu에 할당되었던 array의 메모리를 해제한다.

```python
    import cupy as cp
    import numpy as np
    
    class myTensor(myModule):
    ...
    def to(self, *args, **kwargs):
        ...
        if "cuda" in _args:
            cp.cuda.runtime.setDevice(int(_args.split(":")[-1]))
            self.op.set_op(cp)
            self.data = cp.asarray(self.data)
        elif "cpu" in _args:
            self.op.set_op(np)
            self.data = cp.asnumpy(self.data)
            cp._default_memory_pool.free_all_blocks()
    ...
```

- myTensor는 module을 통과하면서 통과한 이전 module의 backward 함수를 `.backward_prev` 속성에 저장한다. 
- module내에서 연산자를 통해 새로 생성된 myTensor의 경우 연산에 참여한 myTensor의 `.backward_prev`를 가져간다.
이 과정에서 중간에서 생성된 myTensor들이 어떤 module에서 생성되었는지 기록해서 backward시에 사용함.

```python
class myTensor(myModule):
    ...
    def binary_operator_function_call(self, operator, other):
        if isinstance(other, myTensor):
            temp_data = other.data
        else:
            temp_data = other
    
        _new_data = operator(self.data, temp_data)
        _new = myTensor(_new_data)
        _new.grad_fn = f"{operator}"
        _new.op = self.op
        _new.backward_fn = self.backward_prev
        _new.backward_prev = self.backward_prev
        return _new
```


##### Layer 
<hr/>
:white_check_mark: Conv2d
<details>
<summary> <span style="color: #4682B4"> 구현 상세 펼치기/접기 </span> </summary>
<div markdown="1">

```python
Test Code 입니다.
Test Code 입니다.
Test Code 입니다.
```

</div>
</details>
<hr/>
:white_check_mark: Linear
<details>
<summary> <span style="color: #4682B4"> 구현 상세 펼치기/접기 </span> </summary>
<div markdown="1">

```python
Test Code 입니다.
Test Code 입니다.
Test Code 입니다.
```

</div>
</details>
<hr/>
:white_check_mark: BatchNormalization
<details>
<summary> <span style="color: #4682B4"> 구현 상세 펼치기/접기 </span> </summary>
<div markdown="1">

```python
Test Code 입니다.
Test Code 입니다.
Test Code 입니다.
```

</div>
</details>
<hr/>
:heavy_check_mark: ReLu
<details>
<summary> <span style="color: #4682B4"> 구현 상세 펼치기/접기 </span> </summary>
<div markdown="1">

```python
Test Code 입니다.
Test Code 입니다.
Test Code 입니다.
```

</div>
</details>
<hr/>

:white_check_mark: MaxPool
<details>
<summary> <span style="color: #4682B4"> 구현 상세 펼치기/접기 </span> </summary>
<div markdown="1">

```python
Test Code 입니다.
Test Code 입니다.
Test Code 입니다.
```

</div>
</details>
<hr/>
:white_check_mark: AvgPool
<details>
<summary> <span style="color: #4682B4"> 구현 상세 펼치기/접기 </span> </summary>
<div markdown="1">

```python
Test Code 입니다.
Test Code 입니다.
Test Code 입니다.
```

</div>
</details>
<hr/>
:white_check_mark: Flatten
<details>
<summary> <span style="color: #4682B4"> 구현 상세 펼치기/접기 </span> </summary>
<div markdown="1">

```python
Test Code 입니다.
Test Code 입니다.
Test Code 입니다.
```

</div>
</details>
<hr/>
:white_check_mark: Sigmoid
<details>
<summary> <span style="color: #4682B4"> 구현 상세 펼치기/접기 </span> </summary>
<div markdown="1">

```python
Test Code 입니다.
Test Code 입니다.
Test Code 입니다.
```

</div>
</details>

<hr/>

##### Opimizer
<hr/>
:white_check_mark: Adam
<details>
<summary> <span style="color: #4682B4"> 구현 상세 펼치기/접기 </span> </summary>
<div markdown="1">

```python
Test Code 입니다.
Test Code 입니다.
Test Code 입니다.
```

</div>
</details>
<hr/>
:white_check_mark: AdamW
<details>
<summary> <span style="color: #4682B4"> 구현 상세 펼치기/접기 </span> </summary>
<div markdown="1">

```python
Test Code 입니다.
Test Code 입니다.
Test Code 입니다.
```

</div>
</details>
<hr/>
:white_check_mark: Stochastic Gradient Descent
<details>
<summary> <span style="color: #4682B4"> 구현 상세 펼치기/접기 </span> </summary>
<div markdown="1">

```python
Test Code 입니다.
Test Code 입니다.
Test Code 입니다.
```

</div>
</details>
<hr/>

##### Loss function
<hr/>
:white_check_mark: Mean Square Error
<details>
<summary> <span style="color: #4682B4"> 구현 상세 펼치기/접기 </span> </summary>
<div markdown="1">

```python
Test Code 입니다.
Test Code 입니다.
Test Code 입니다.
```

</div>
</details>
<hr/>
:white_check_mark: Cross Entropy 
<details>
<summary> <span style="color: #4682B4"> 구현 상세 펼치기/접기 </span> </summary>
<div markdown="1">

```python
Test Code 입니다.
Test Code 입니다.
Test Code 입니다.
```

</div>
</details>
<hr/>

#### 구현할때 사용한 Tip

##### Python 

- 어떤 class를 만들때 `__new__` 을 먼저 call해서 메모리를 할당한 이후 `__init__`을 call한다.
- instance를 생성한 class를 function call하게 되면 class의 `__call__` 함수를 호출한다. `__call__`함수를 정의할 경우 class의 instance
를 callable하게 바꿔준다.
- 어떤 class의 instance에서 `Temp[:,0]`을 하면 class의 `__getitem__` 함수를 call한다.
- 어떤 class의 instance에서 `Temp[3] = 3`와 같은 리스트 할당을 하게 되면 class의 `__setitem__` 함수를 call한다.
- 어떤 class의 instance에서 `Temp1 + Temp2`와 같은 operator call이 일어나면 해당 operator에 대한 함수를 call한다.
여기서는 +이므로 `__add__`를 call한다.

[[operator call에 대한 참조(python 3.10)]](https://docs.python.org/ko/3.10/library/operator.html){:target="_blank"}

- class의 attribute를 가져오거나 얻을때 `__getattr__`와 `__setattr__` 함수를 call한다. 
예를 들어 `Temp.att1 = 1`이 실행되면 python의 `__builtin__`에 정의된 `__setattr__`를 call한다.
기본적으로 모든 Object들은 따로 재정의 하지않으면 python 내부에 정의된 함수들이 실행된다.

#### Reference 

[https://github.com/SkalskiP/ILearnDeepLearning.py](https://github.com/SkalskiP/ILearnDeepLearning.py){:target="_blank"}













