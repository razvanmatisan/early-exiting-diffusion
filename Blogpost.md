# DeeDiff: Dynamic Uncertainty-Aware Early Exiting For Accelerating Diffusion Model Generation
### Introduction

This project delves into, replicates, and extends upon the discoveries presented in the paper ["DeeDiff: Dynamic Uncertainty-Aware Early Exiting For Accelerating Diffusion Model Generation"](https://arxiv.org/abs/2309.17074). This work proposes an early exiting framework to improve the generation speed of diffusion models, based on a timestep-aware uncertainty estimation module attached to each intemediate layer, as well as based on an uncertainty-aware layer-wise loss.

This blog post aims to achieve 3 objectives:
  1. Provide readers with a thorough understanding of the proposed framework.
  2. Verify the authors' original claims and results through a reproduction of their experiments.
  3. Propose extensions on the original model and verify whether these extensions can further improve the generation speed of diffusion models. 

### Related work

#### Diffusion models

Diffusion models have achieved great success in image generation, showing a superior performance than GAN-based models. 

Originally, the most common backbone network used in difussion models is UNet, but, more recently, transformer-based architectures such as U-ViT have shown better performance.


#### Efficiency in Diffusion

In spite of their notable performance, generating a high-fidelity sample from a diffusion model is computationally expensive, requiring not only a large number of steps (e.g. 1000 steps) but also considerable computational resources for each step. 

This has motivated a great deal of research aimed at accelerating the inference speed of these models. DDIMs [1] generalize on simple DDPMs, with the same training objective and procedure, but with a deterministic generative process that leads to a sped up by a factor of 10 to even 50. Concurrently, several other works [2, 3, 4] aim to reduce the number of sampling steps by distilling knowledge from a teacher model to a student model with fewer steps.

However, all these methods continue to employ the entire computational capacity of the network for each sampling step, which in the end still incurs low generation speed. 

#### Early-exit strategy

The premise that not all inputs require the same levels of computational resources motivated the exploration of early exiting strategies [5], which can be defined as general neural network accelerating techniques, allowing the models to selectively exit from hidden layers instead of a full computation when confident enough. Traditional early-exit strategies are not applicable in the context of diffusion models due to the time-series property, and thus other modifications are needed. 

<!-- This is where DeeDiff comes into picture, proposing an uncertainty-aware layer-wise loss in order to preserve more information with less layers.  -->

## DeeDiff model architecture

At its base, DeeDiff uses a U-ViT [6] architecture, a transformer-based diffusion model. During inference, each transformer layer output is inputted into a classifier to calculate the confidence or entropy, which acts as a substitute for the difficulty of samples and guides the early-exit decision-making process.

#### Timestep-aware UEM

The authors argue that estimating the noise in each step of diffusion models can be regarded as a regression task, and thus consider each generation step separately. Namely, for the implementation of the uncertainty classifiers, they propose a timestep-aware uncertainty estimation module (UEM) in the form of a fully-connected layer:

![image](https://hackmd.io/_uploads/HkTEfBP70.png)

where w_t, b_t, f, timesteps are the weight matrix, weight bias, activation function and timestep embeddings.

The authors also mention that the output should be unpatched to generate the uncertainty maps, however full implementation details of this module are missing from the paper. 

The pseudo uncertainty ground truth is constructed as follows:

![image](https://hackmd.io/_uploads/HkxEjMSw70.png)

where g_i is the output layer, ϵ is the ground truth noise value, F is a function to smooth the output (i.e. tanh).

This brings forth the loss function of this module, designed as the MSE loss of the estimated and pseudo ground truth uncertainty:

![image](https://hackmd.io/_uploads/SyJ-QBvQR.png)

During inference, early exit is then achieved my comparing the estimated uncertainty of the output prediction from each layer with a predefined threshold. 

#### Uncertainty-aware layer-wise loss

To mitigate the information loss chracteristic to not utilizing the full model, the authors also propose an uncertainty-aware layer-wise loss. They draw inspiration from previous work, with one important modification, a weighting term, as to account for the information loss accumulating over multi-step inference:

![image](https://hackmd.io/_uploads/rySw4HPQC.png)

where u_i is the uncertainty value estimated in each layer.

#### Training strategy

DeeDiff utilizes a joint training strategy in order to balance the effect between uncertainty estimation loss and uncertainty-aware layer-wise loss, arriving at the final loss formula:

![image](https://hackmd.io/_uploads/Hyr34BPmR.png)

where L_simple is the MSE loss between the true and predicted noise, while the other two losses are defined above in (10) and (12).


## Ablations

We decided to conduct several ablations in order to gain a more comprehensive understanding of the method, as well as identify its limitations, potential trends and areas of improvements.

- In our implementation, we have attached a classifier before the first transformer block, such that if the model is confident enough to exit just with the input at the respective time step it can do so, thus skipping the entire sampling step. 
- Our implementation of the classifiers themselves differ from the paper in multiple regards, namely:
    - Instead of a linear layer, we have used a more general approach, namely an attention probe. TODO: add reasoning. 
    - In order to reduce the number of parameters, and ultimately the inference speed, we have used a single shared classifier per time step and have instead injected the time information in the form of an input embedding. 



## Results

## Further research

## Conclusion

## Contributions

## Bibliography

[1] Song, Jiaming, Chenlin Meng, and Stefano Ermon. "Denoising diffusion implicit models." arXiv preprint arXiv:2010.02502 (2020).

[2] Luhman, Eric, and Troy Luhman. "Knowledge distillation in iterative generative models for improved sampling speed." arXiv preprint arXiv:2101.02388 (2021).

[3] Salimans, Tim, and Jonathan Ho. "Progressive distillation for fast sampling of diffusion models." arXiv preprint arXiv:2202.00512 (2022).

[4] Meng, Chenlin, et al. "On distillation of guided diffusion models." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

[5] Teerapittayanon, Surat, Bradley McDanel, and Hsiang-Tsung Kung. "Branchynet: Fast inference via early exiting from deep neural networks." 2016 23rd international conference on pattern recognition (ICPR). IEEE, 2016.

[6] Bao, Fan, et al. "All are worth words: a vit backbone for score-based diffusion models." NeurIPS 2022 Workshop on Score-Based Methods. 2022.

[7] Jayasumana, Sadeep, et al. "Rethinking fid: Towards a better evaluation metric for image generation." arXiv preprint arXiv:2401.09603 (2023).