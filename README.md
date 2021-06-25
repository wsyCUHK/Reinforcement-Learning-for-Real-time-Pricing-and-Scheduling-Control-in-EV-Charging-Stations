# 22nd Nov 2020 Update
Dear All,

2020-11-17: I notice that people are waiting for the code. As I said in some emails, the paper is finally done when I was @ Tencent. In May 2020, I left from Tencent and join SZU. As you know, for a commercial company, I leave all the material inside Tencent. I really sorry that it is not possible to provide the original MATLAB code any more. I will try to re-implement it in Pytorch recently. Thanks.

2020-11-22: A pytorch implementation is uploaded. Again, I'm sorry that I cannot  provide the original MATLAB implementation with SARSA. As our group is devoting some research efforts with the A3C framework, I implement the EV Charging environment and feature function state with a A3C framework for implementation efficiency. As a result, although the result slighly differs from the orginal one, the fast convergence is still significant. Personally speaking, our main contribution is the problem formulation and transformation, which can be found in the env() function. So, let me stop here and thank you again for your attention. 

## Cite this work (The TII version is prefered)

1. S. Wang, S. Bi, and Y. J. Zhang, “[Reinforcement Learning for Real-time Pricing and
Scheduling Control in EV Charging Stations](https://ieeexplore.ieee.org/document/8888199),” in IEEE Transactions on Industrial Informatics, vol. 17, no. 2, pp. 849-859, Feb. 2021, doi: 10.1109/TII.2019.2950809.

2. S. Wang, S. Bi and Y. A. Zhang, "A Reinforcement Learning Approach for EV Charging Station Dynamic Pricing and Scheduling Control," 2018 IEEE Power & Energy Society General Meeting (PESGM), 2018, pp. 1-5, doi: 10.1109/PESGM.2018.8586075.
![The EV Charging Station.](https://user-images.githubusercontent.com/37823466/68126507-eec60e80-ff4e-11e9-9b1f-cba8514ae5c3.png)

### Abstract
This paper proposes a Reinforcement-Learning (RL) approach for optimizing charging scheduling and pricing strategies that maximize the system objective of a public electric vehicle (EV) charging station. The proposed algorithm is ”on-line” in the sense that the charging and pricing decisions depend only on the observation of past events, and is ”model-free” in the sense that the algorithm does not rely on any assumed stochastic models of uncertain events. To cope with the challenge arising from the time-varying continuous state and action spaces in the RL problem, we first show that it suffices to optimize the total charging rates to fulfill the charging requests before departure times. Then, we propose a feature-based linear function approximator for the state-value function to further enhance the generalization ability. Through numerical simulations with real-world data, we show that the proposed RL algorithm achieves on average 138.5% higher profit than representative benchmark algorithms.

## About authors

- [Shuoyao WANG](https://scholar.google.com/citations?user=RYG-gYYAAAAJ&hl=en), sywang AT szu.edu.cn

- [Suzhi BI](https://scholar.google.com/citations?user=uibqC-0AAAAJ), bsz AT szu.edu.cn

- [Ying Jun (Angela) Zhang](https://scholar.google.com/citations?user=iOb3wocAAAAJ), yjzhang AT ie.cuhk.edu.hk

## Required packages

- Tensorflow

- numpy

- scipy


## About Authors
Shuoyao Wang, sywang AT szu.edu.cn:Shuoyao Wang received the B.Eng. degree (withnfirst class Hons.) and the Ph.D degree in information engineering from The Chinese University of HongnKong, Hong Kong, in 2013 and 2018, respectively. From 2018 to 2020, he was an senior researcher with the Department of Risk Management, Tencent,
Shenzhen, China. Since 2020, he has been with the College of Electronic and Information Engineering, Shenzhen University, Shenzhen, China, where he is currently an Assistant Professor. His research interests include optimization theory, operational research, and machine learning in Multimedia Processing, Smart Grid, and Communications. For any inquiry for this work or cooperation， please feel free to contact us through sywang AT szu.edu.cn.

Suzhi Bi, bsz@szu.edu.cn

Ying-jun Angela Zhang, yjzhang@ie.cuhk.edu.hk

## Thank You for Reading!!!
