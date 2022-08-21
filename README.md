# Ranking-transferability-of-Adversarial-Examples
This is the code for the paper named “Transferability ranking for
adversarial examples”. We include here the content of the code, its requirements
and the instructions on how to use it.

Content
1. The attached code includes the code that runs experiment E1.1, E1.2, E1.3 and
E1.4.
2. The code include automatic download of  
  a. CIFAR10 dataset
  b. CIFAR10 models required for the experiments
    i. Resnet-18 models
    ii. Wideresnet models
  c. Imagenet models required for the experiment
    i. Resnet-50
Note that the code does not include the Imagnet dataset.
3. The final product of running the attached notebook is a plot of the specified dataset,
scenario (general / resized / compressed / different architecture) and attack
algorithm.

Requirements
1. Appropriate hardware and software
a. CIFAR10 can be ran on google colab (on the gpu framework)
b. Imagenet requires stronger environments, we used four 3090 cards.
c. Pytorch version 1.10 or higher
2. To run the Imagenet experiments, the validation set of imagenet is required
(ILSVRC2012_img_val.tar).

Instructions
1. Extract all files into a single directory
2. Open the notebook named ‘ranking_main’
3. Run all blocks

Configuration
1. The default config is experiment E1.1 (general) on the CIFAR dataset
2. On the second block in the notebook there are the configuration variables.
3. The ‘architecture’ scenario stands for the experiment where the victim (𝑓) is of the
wideresnet architecture.
4. Note that the Imagenet experiment may take several hours.
