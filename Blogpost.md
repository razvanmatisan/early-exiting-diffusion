### !!! Note: open thie file in VS Code or using https://hackmd.io/?nav=overview to properly see the equations.

# DeeDiff: Dynamic Uncertainty-Aware Early Exiting For Accelerating Diffusion Model Generation
**D. Gallo, R. Matisan, A. Monroy, J. Partyka, A. Vasilcoiu**
## Introduction

This project delves into and extends upon the discoveries presented in the paper ["DeeDiff: Dynamic Uncertainty-Aware Early Exiting For Accelerating Diffusion Model Generation"](https://arxiv.org/abs/2309.17074). This work proposes an early exiting framework to improve the generation speed of diffusion models, based on a timestep-aware uncertainty estimation module attached to each intemediate layer, as well as based on an uncertainty-aware layer-wise loss.

This blog post aims to achieve 3 objectives:
  1. Provide readers with a thorough understanding of the proposed framework.
  2. Verify the authors' original claims and results through a reproduction of some of their experiments.
  3. Propose extensions on the original model and verify whether these extensions can further improve the generation speed of diffusion models. 

## Related work

#### Diffusion models

Diffusion models [0, 1] have achieved great success in image generation, showing a superior performance than GAN-based models. They are based on progressively adding random noise to a dataset and learning how to reverse that process to recover the origina data. 

The forward process, noise is progresivelly added to a sample $\mathbf{x}_0$ in $T$ timesteps, generating increasingly noisy samples $\mathbf{x}_1, \mathbf{x}_2, ...,  \mathbf{x}_T$ using a Gaussian Markovian difusion kernel:

$$\begin{align} 
q\left(\mathbf x_t \mid \mathbf x_{t-1} \right) = \mathcal{N}\left( \mathbf x_t ; \sqrt{1-\beta_t} \mathbf x_{t-1}, \beta_t \mathbf{I} \right) & \qquad \qquad \text{(Equation 1)} 
\end{align}$$

The mean and the variance of the distribution are parametrized with a diffusion rate $\beta_t$, wich starts from zero and increases with respect to the timestep.

In the reverse process, we need to model the distribution of $\mathbf X_{t-1} | \mathbf X_t$, which implies computing the marginal distribution of $\mathbf X_0$. Therefore, we approximate it with another Gaussian distribution whose parameters are learnt by a neural network:

$$p_\theta \left( \mathbf x_{t-1} \mid \mathbf x_t \right) := \mathcal{N} \left( \mathbf x_{t-1} ; \mathbf \mu_{\mathbf \theta} \left( \mathbf x_t, t \right), \mathbf \Sigma_{\mathbf \theta} \left( \mathbf x_t, t \right) \right) \qquad \qquad \text{(Equation 2)}$$

In the DDPM sampling method, $\mathbf \mu_{\mathbf \theta} \left( \mathbf x_t, t \right)$ is estimated using a neural network and $\mathbf \Sigma_{\mathbf \theta} \left( \mathbf x_t, t \right)$ is kept fixed to $\tilde \beta_t \mathbf I$, where $\tilde \beta_t$ is analitically computed as a function of $\{ \beta_i\}_{i=1}^t$.

$$\begin{align} 
\mathbf \mu_\theta \left( \mathbf x_t, t \right) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \mathbf \epsilon_\theta \left( x_t, t \right) \right) & \qquad \qquad \text{(Equation 3),}
\end{align}$$

where $\alpha_t = 1 - \beta_t$ and $\bar{\alpha_t} = \prod_{i=1}^t \alpha_i$.

During training, the loss is computed as the expected ation of the difference between the real $\mathbf \epsilon_t$ noise and its approximation and its approximation $\epsilon_{\mathbf \theta} \left( \mathbf x_t, t \right)$:
 
$$L_{\text{simple}} = \mathbb{E}_{\mathbf x_0, \epsilon, t } \left[ \left|| \mathbf \epsilon - \mathbf \epsilon_\theta \left(\mathbf x_t, t \right) \right||_2^2 \right], \text{ where } \mathbf x_t = \sqrt{\bar{\alpha_t}}\mathbf x_0 + \sqrt{1 - \bar{\alpha_t}}\mathbf \epsilon, \;, \mathbf \epsilon \sim \mathcal N(\mathbf 0, \mathbf I) \qquad \qquad \text{(Equation 4)}$$

Originally, the most common backbone network used to estimate this term was UNet, but, more recently, transformer-based architectures such as U-ViT have shown better performance.


#### Efficiency in Diffusion

In spite of their notable performance, generating a high-fidelity sample from a diffusion model is computationally expensive, requiring not only a large number of steps (e.g. 1000 steps) but also considerable computational resources for each step. 

This has motivated a great deal of research aimed at accelerating the inference speed of these models. DDIMs [1] generalize on simple DDPMs, with the same training objective and procedure, but with a deterministic generative process that leads to a sped up by a factor of 10 to even 50. Concurrently, several other works [2, 3, 4] aim to reduce the number of sampling steps by distilling knowledge from a teacher model to a student model with fewer steps.

However, all these methods continue to employ the entire computational capacity of the network for each sampling step, which in the end still incurs low generation speed. 

#### Early-exit strategy

The premise that not all inputs require the same amount of computational resources motivated the exploration of early exiting strategies [5], which can be defined as general neural network accelerating techniques, allowing the models to selectively exit from hidden layers instead of a full computation when confident enough. Traditional early-exit strategies are not applicable off-the-shelf in the context of diffusion models due to the time-series property and the lack of a natural confidence estimation measure, and thus other modifications are needed. 

<!-- This is where DeeDiff comes into picture, proposing an uncertainty-aware layer-wise loss in order to preserve more information with less layers.  -->

## DeeDiff model architecture

At its base, DeeDiff uses a U-ViT [6] architecture, a transformer-based diffusion model. During inference, each transformer layer output is inputted into a classifier to calculate the confidence or entropy, which acts as a substitute for the difficulty of samples and guides the early-exit decision-making process.

#### Timestep-aware UEM

The authors argue that estimating the noise in each step of diffusion models can be regarded as a regression task, and thus consider each generation step separately. Namely, for the implementation of the uncertainty classifiers, they propose a timestep-aware uncertainty estimation module (UEM) in the form of a fully-connected layer:

$$\begin{align} 
u_{i, t}=f\left(\mathbf{w}_{\mathbf{t}}^{T}\left[L_{i, t}\right.\right., timesteps \left.]+b_{t}\right), & \qquad \qquad \text{(Equation 5)} 
\end{align}$$
where $w_t$, $b_t$, f, timesteps are the weight matrix, weight bias, activation function and timestep embeddings.

The pseudo uncertainty ground truth is constructed as follows:

$$\begin{align} 
\hat{u}_{i, t}=F\left(\left|g_{i}\left(L_{i, t}\right)-\epsilon_{t}\right|\right), & \qquad \qquad \text{(Equation 6)} 
\end{align}$$

where $g_i$ is the output layer, ϵ is the ground truth noise value and $F$ is a function to smooth the output (i.e. tanh).

This brings forth the loss function of this module, designed as the MSE loss of the estimated and pseudo ground truth uncertainty:

$$\begin{align} 
\mathcal{L}_{u}^{t}=\sum_{i}^{N}\left\|u_{i, t}-\hat{u}_{i, t}\right\|^{2} & \qquad \qquad \text{(Equation 7)}
\end{align}$$

During inference, early exit is then achieved by comparing the estimated uncertainty of the output prediction from each layer with a predefined threshold. 

#### Uncertainty-aware layer-wise loss

To mitigate the information loss that occurs when not utilizing the full model, the authors also propose an uncertainty-aware layer-wise loss. They draw inspiration from previous work, with one important modification, a weighting term, as to account for the information loss accumulating over multi-step inference:s

$$\begin{align} 
\mathcal{L}_{U A L}^{t}=\sum_{i}^{N-1}\left(1-u_{i, t}\right) \times\left\|g_{i}\left(L_{i, t}\right)-\epsilon\right\|^{2}, & \qquad \qquad \text{(Equation 8)}
\end{align}$$

where $u_{i, t}$ is the uncertainty value estimated in each layer.

#### Training strategy

DeeDiff utilizes a joint training strategy in order to balance the effect between uncertainty estimation loss and uncertainty-aware layer-wise loss, arriving at the final loss formula:

$$\begin{align} 
L_{\text {all }}=\mathcal{L}_{\text {simple }}^{t}(\theta)+\lambda \mathcal{L}_{u}^{t}+\beta \mathcal{L}_{U A L}^{t}. & \qquad \qquad \text{(Equation 9)}
\end{align}$$


## Novel contributions

We can summarize our novel contributions as follows:
- As per Equation 5 above, we know that in the original paper, the classifier is implemented as a linear layer. However, this will produce an uncertainty  
- In order to reduce the number of parameters and improve the training speed, we have used a single shared classifier for all timesteps and have instead injected the time information in the input embedding. 
- We have added a 4th component to the model's loss consisting of the UAL loss from Equation 8 without the uncertainty weighting $(1 - u_{i,t})$. The motivation behind this is to prevent the model from learning to generate bad-quality outputs (which imply an uncertainty $u_{i,t}$ close to $1$) to minimize $\mathcal L_{UAL}^t$.
- As a minor addition, in each time step, we apply the uncertainty classifier to the inputs, instead of the outputs of each transformer block (including the first one). This ensures that the model is not only able to early-exit, but to also skip steps entirely.


## Experimental setup

We conducted all experiments on the CIFAR-10 dataset and with a U-ViT Small model comprising of 13 layers, where there are skip connections between the outputs of layers 1-6 and the inputs of layers 7-13. 

In terms of training strategies, we have used the following approaches:
- jointly training the U-ViT and the classifiers from scratch with the original 3 losses
- training U-ViT separately and then, with a frozen backbone, train the DeeDiff model with the original 3 losses
- training U-ViT separately and then, with a frozen backbone, train the DeeDiff model with the original 3 losses plus our additional loss component

#### Evaluation
To qualitately assess the generated images, we chose an improvement over FID, the the most popular metric and the one used in the original DeeDiff paper too, namely the CMMD metric. As shown by the authors in [7], FID has several limitations such as contradicting human raters, not reflecting gradual improvement, not capturing distortion levels and producing inconsistent results across varying sample size. CMMD is a metric based on richer embeddings and the maximum mean discrepancy distance with the Gaussian RBF kernel. This metric addresses the limitations of FID and proves to be a unbiased, distribution-free and sample efficient estimator, thus providing a more robust and reliable assessment of image quality. 

<!-- We have experimented not only with initializing the backbone U-ViT model with [pre-trained weights](https://github.com/baofff/U-ViT), but also with training everything from scratch. -->


## Results

### Early-exit

In terms of early-exiting trends, we observed that both approaches, namely training with the initial 3 loss components and with our new component, managed to train classifiers that lead to stopping the computation in early layers in some timesteps, thus partially validating the proposed technique. 

However, for the model with 3 losses, we observed that it early-exits before layer 4 during the first approximately 250 steps, while it performs a full computation for the other timesteps, with no other in-between exit layers. We observe roughly the same trend for the model trained with the additional loss, but with some other early-exit layers too for the first few generation steps, which can be visualized in Figure 1.

<table align="center">
  <tr align="center">
      <td><img src="https://hackmd.io/_uploads/SkneUiD7A.svg" width=800></td>
  </tr>
  <tr align="left">
    <td colspan=2><b>Figure 1.</b> Early-exit layers for different thresholds with respect to timesteps. </td>
  </tr>
</table>

Figure 2 shows the UEM classifier outputs computed for every layer and timestep during the training of the model with four losses. We observe how the uncertainty values increase as the denoising process advances. It is also interesting to see the differences between layers. The outputs start to increase after the middle layer (7), where the outputs of the initial layers are added to the inputs via the skip connections.

<table align="center">
  <tr align="center">
      <td><img src="https://hackmd.io/_uploads/H1oxUiP7A.svg" width=800></td>
  </tr>
  <tr align="left">
    <td colspan=2><b>Figure 2.</b> Classifier outputs over layers and timesteps. </td>
  </tr>
</table>

### Image quality

In terms of image quality, we observe that the pre-trained backbones resulted in better generated images as opposed to the models trained from scratch. As it can be seen in Figure 3, we also observe that early-exit thresholds between 0 and 0.05 don't alter the image quality significantly, while threholds larger than 0.075 result in considerable alterations.

<table align="center">
  <tr align="center">
      <td><img src="https://hackmd.io/_uploads/Sy3e8sPXC.svg" width=800></td>
  </tr>
  <tr align="left">
    <td colspan=2><b>Figure 3.</b> Images generated from models with different early-exit thresholds. </td>
  </tr>
</table>

Below, some results of comparing 10 generated images with original CIFAR-10 dataset images:

<style>
  table {
    margin: auto;
  }
</style>

<table align="center">
  <tr>
    <th>Model</th>
    <th>CMMD value</th>
  </tr>
  <tr>
    <td>Early-exit (threshold=0.125)</td>
    <td>0.574</td>
  </tr>
  <tr>
    <td>Early-exit (threshold=0.005)</td>
    <td>0.407</td>
  </tr>
  <tr>
    <td>No early-exit</td>
    <td>0.479</td>
  </tr>
</table>


## Further research

## Conclusion

<!-- ## Contributions
- Daniel: vizz guy
- Razvan:
- Alejandro:
- Janusz: 
- Ana: -->

## Bibliography
[0] Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models." Advances in neural information processing systems 33 (2020): 6840-6851.

[1] Song, Jiaming, Chenlin Meng, and Stefano Ermon. "Denoising diffusion implicit models." arXiv preprint arXiv:2010.02502 (2020).

[2] Luhman, Eric, and Troy Luhman. "Knowledge distillation in iterative generative models for improved sampling speed." arXiv preprint arXiv:2101.02388 (2021).

[3] Salimans, Tim, and Jonathan Ho. "Progressive distillation for fast sampling of diffusion models." arXiv preprint arXiv:2202.00512 (2022).

[4] Meng, Chenlin, et al. "On distillation of guided diffusion models." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

[5] Teerapittayanon, Surat, Bradley McDanel, and Hsiang-Tsung Kung. "Branchynet: Fast inference via early exiting from deep neural networks." 2016 23rd international conference on pattern recognition (ICPR). IEEE, 2016.

[6] Bao, Fan, et al. "All are worth words: a vit backbone for score-based diffusion models." NeurIPS 2022 Workshop on Score-Based Methods. 2022.

[7] Jayasumana, Sadeep, et al. "Rethinking fid: Towards a better evaluation metric for image generation." arXiv preprint arXiv:2401.09603 (2023).