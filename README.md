# STILKDNet


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
![Uploading image.png…]()


## Environment require
conda create --name CD python=3.11

## run
bash run.sh

