# poor_teacher

### Requirements
* python >= 3.7.13
* pytorch >= 1.12.0
* torchmetrics
* transformers >= 4.24.0
* numpy
* pandas
* tqdm
* gc

### 1. 데이터
* 링크 : https://gluebenchmark.com/tasks
* 적용 데이터 셋 : **GLUE**(General Language Understanding Evaluation)

  * single-sentence tasks
    * SST-2 (Stanford Sentiment Treebank)
  * similarity and paraphrase tasks
    * MRPC (Microsoft Research Paraphrase Corpus)
    * QQP (Quora Question Pairs)
  * inference tasks
    * MNLI (Multi-Genre Natural Language Inference)
    * QNLI (Question Natural Language Inference)
    * **RTE** (Recognizing Textual Entailment, 업로드 데이터)

![gls 1](https://user-images.githubusercontent.com/100681144/231469975-65868513-12eb-4438-9257-240442c18af5.PNG)

### 2. Knowledge Distillation
* 링크 : https://arxiv.org/pdf/1503.02531.pdf
* Hinton et al.(2015)에 의해 처음으로 제안되었음
    * 큰 모델(Teacher Model)의 지식을 작은 모델(Student Model)에게 증류시키는 방법
    * 일반적인 딥러닝 학습을 위한 Loss Function인 Cross Entropy Loss와 Softmax 확률 분포를 변형시킨 방법을 통한 KL-Divergence Loss를 사용함
        * T(temperature) 하이퍼파라미터를 이용하여 Hard Target을 Soft Target으로 변형시켜 증류
        * 아래의 그림은 기본적인 지식 증류 모델의 구조

![image](https://user-images.githubusercontent.com/100681144/232202734-b324de71-c16e-472b-b8e6-5419a5812a26.png)

### 3. Poor-teacher, Triplet Loss
* Poor-teacher
  * BERT-base의 Layer의 수를 1개로 줄이고, 1 Epoch만 학습을 시킨 모델
  * Teacher, Student, Poor-Teacher를 Triplet Loss로 학습

### 4. 적용 Model
* BERT-base(Teacher Model)
  * Layer = 12
  * Hidden size = 768
  * Attention heads = 12
  * Total Parameters : 110M

* BERT-layer-6(Student Model)
  * Layer = 6
  * Hidden size = 768
  * Attention heads = 12
  * Total Parameters : 55M
  
* BERT-layer-1(Poor-Teacher Model)
  * Layer = 1
  * Hidden size = 768
  * Attention heads = 12
  * Total Parameters : 9.2M
  
### 5. 결과

![poor 결과](https://user-images.githubusercontent.com/100681144/232205651-f55d2d90-1e58-4c0d-b702-ae3c3c9a0298.PNG)

