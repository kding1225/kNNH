## [kNN hashing with Factorized Neighborhood Representation](https://ieeexplore.ieee.org/document/7410488)

This code implements the paper: 

```
@INPROCEEDINGS{kNNH,
      author={Ding, Kun and Huo, Chunlei and Fan, Bin and Pan, Chunhong},
      booktitle={ICCV}, 
      title={kNN Hashing with Factorized Neighborhood Representation}, 
      year={2015},
      pages={1098-1106}
  }
```



The code is tested on 64-bit CentOS Linux 7.1.1503 (Core) system with MATLAB 2014b and 64-bit Windows 10 system with MATLAB 2014a. It includes:

1. [demo.m](demo.m): demo code for linear and nonlinear kNN hashing.
2. [kNNH.m](codes/kNNH/kNNH.m): implements the training of linear kNN hashing.
3. [k2NNH.m](code/kNNH/k2NNH.m): implements the training of nonlinear kNN hashing.

To try the code, please run [demo.m](demo.m). If all goes well, it will print the mean average precision

(MAP) of three methods: kNNH, k2NNH and supervised discrete hashing (SDH, CVPR 2015).



If you have any questions, please contact me by Email: kun.ding AT ia.ac.cn.
