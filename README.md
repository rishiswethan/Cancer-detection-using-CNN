# Breast-cancer-detection-using-CNN

Breast cancer constitutes a leading cause of cancer-related deaths worldwide. Accurate diagnosis of cancer from eosin-stained images remains a complex task, as medical professionals often encounter discrepancies in reaching a final verdict. Computer-Aided Diagnosis (CAD) systems offer a means to reduce cost and enhance the efficiency of this intricate process. Traditional classification approaches rely on problem-specific feature extraction methods based on domain knowledge. To address the numerous challenges posed by feature-based techniques, deep learning methods have emerged as significant alternatives.

We propose a method for the classification of hematoxylin and eosin-stained breast biopsy images using Convolutional Neural Networks (CNNs). Our method classifies images into four categories: normal tissue, benign lesion, in situ carcinoma, and invasive carcinoma, as well as a binary classification of carcinoma and non-carcinoma. The network architecture is meticulously designed to extract information at various scales, encompassing both individual nuclei and the overall tissue organization. This design enables the seamless integration of our proposed system with whole-slide histology images. Our method achieves an accuracy of 77.8% for the four-class classification and demonstrates a sensitivity of 95.6% for cancer cases.

To use this project:

1. You'll need python3 to run the program

2. I've included the preprocessed image data. You can download it from [here](https://drive.google.com/open?id=17LR9ssbENit-3vsEAM63FptNasB5AHrr). Now place the 5 files that you just downloaded with the folder with the `.py` file

3. Use `pip install package-name` to install the below packages

4. You need to have the following python packages installed
	* keras
	* tensorflow (Both CPU or GPU version should do)
	* PIL
	* numpy

5. You can modify the default hyparameters by modifying the variables between the `#` in the first few lines line

To run the program, navigate to the folder in command line and use the following command,
```
python BreastCancer.py
```
I've also included a pretrained model. To test your own image or one of the samples using it, paste the image in the folder with the `.py` file and rename it as `my_image.jpg`, then during execution choose to test your own image by following the on screen commands
