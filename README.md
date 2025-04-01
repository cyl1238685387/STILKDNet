# STILKDNet
Spatio-Temporal Interlayer-Knowledge  Distillation Network for Remote Sensing Image  Change Detection

## Abstract
In the task of remote sensing image change detec
tion and segmentation, accurately identifying and segmenting
 changed regions is crucial for surface dynamic monitoring.
 Current methods, when dealing with complex scenarios, have
 limited detection and segmentation accuracy due to difficulties
 in capturing spatio-temporal feature differences. To address this,
 this paper proposes an innovative method named STIKDNet.
 First, a perturbation set is obtained by perturbing Image B.
 Then, the original dual-temporal image set and the perturbation
 set are input into the backbone network for training. Meanwhile,
 knowledge distillation is introduced, with the original set as
 the teacher model and the perturbation set as the student
 model. The original supervised loss, the student’s distillation
 loss, and the supervised loss are combined, and a specially
 designed loss function is used to measure the output differ
ences between the teacher and student models. During change
 detection and segmentation, minimizing this function and back
propagating the three losses optimize the network parameters,
 enabling the student model to fit the teacher model’s output
 distribution, calibrate its predictions to approach the teacher
 model’s judgment logic, and improve performance, providing
 accurate results for surface monitoring. Experiments show that
 STIKDNet outperfroms current state-of-the-art methods on three
 benchmark dataset of LEVIR-CD, SYSU-CD and WHU-CD. 
<br><br>


## Environment require
conda create --name CD python=3.11
<br><br>
## train
bash run.sh
<br><br>
## Method
TABLE I: Choice of data perturbation method

![image](https://github.com/user-attachments/assets/0bce94b7-dcfc-4a15-bb7e-2991ef8a98da)
<br><br>
![image](https://github.com/user-attachments/assets/4d38bbee-41a0-4375-943c-8aad544fb79d)
Fig. 1: The pipeline of the proposed STIKDNet
<br><br>
TABLE II: Experimental Results of Different Transformations on Image B on different Dataset
![image](https://github.com/user-attachments/assets/1bfc1969-210f-4874-adf8-4333391ad12e)
<br><br>
## Comparison With SOTA Methods
TABLE III: Experimental Results for Different Datasets and Loss Combinations

![image](https://github.com/user-attachments/assets/cb380c1f-dc85-4f3d-9175-95557fcced67)
<br><br>
TABLE IV: Quantitative Comparisons on CD Results
![image](https://github.com/user-attachments/assets/079ea2bb-fae5-48a5-83a8-41344d8c8580)

![image](https://github.com/user-attachments/assets/3d9e338e-0228-4df7-82cc-daae3c791e85)
Fig. 2: Visual comparisons of the proposed method and SOTA methods on the LEVIR-CD dataset (a) A images. (b) B images. (c) FC-EF [26]. (d) ChangeFormer [30]. (e) TFI-GR [32]. (f) A2Net [33]. (g) SEIFnet [17] (h) Proposed STIKDNet. (i) Ground truth. S1–S3 represent three samples.
<br><br>

![image](https://github.com/user-attachments/assets/4a58a6d7-eef8-41ea-9b9a-6ec513ed87ae)
Fig. 3: Visual comparisons of the proposed method and SOTA methods on the SYSU-CD dataset (a) A images. (b) B images. (c) FC-EF [26]. (d) ChangeFormer [30]. (e) TFI-GR [32]. (f) A2Net [33]. (g) SEIFnet [17] (h) Proposed STIKDNet. (i) Ground truth. S1–S3 represent three samples.
<br><br>

![image](https://github.com/user-attachments/assets/e472fcde-a83a-451e-91c1-18f153fcacff)
Fig. 4: Visual comparisons of the proposed method and SOTA methods on the WHU-CD dataset (a) A images. (b) B images. (c) FC-EF [26]. (d) ChangeFormer [30]. (e) TFI-GR [32]. (f) A2Net [33]. (g) SEIFnet [17] (h) Proposed STIKDNet. (i) Ground truth. S1–S3 represent three samples.
<br><br>




