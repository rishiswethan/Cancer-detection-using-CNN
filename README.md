# Breast-cancer-detection-using-CNN

Breast cancer is one of the main causes of cancer death worldwide. The diagnosis of cancer from eosin stained images is non-trivial and specialists often disagree on the final diagnosis. Computer-aided Diagnosis systems contribute to reduce the cost and increase the efficiency of this process. Conventional classification approaches rely on feature extraction methods designed for a specific problem based on field-knowledge. To overcome the many difficulties of the feature-based approaches, deep learning methods are becoming important alternatives. A method for the classification of hematoxylin and eosin stained breast biopsy images using Convolutional Neural Networks (CNNs) is proposed. Images are classified in four classes, normal tissue, benign lesion, in situ carcinoma and invasive carcinoma, and in two classes, carcinoma and non-carcinoma. The architecture of the network is designed to retrieve information at different scales, including both nuclei and overall tissue organization. This design allows the extension of the proposed system to whole-slide histology images. Accuracies of 77.8% for four classes is achieved. The sensitivity of our method for cancer cases is 95.6%.

To use this project:

1. You'll need python3 to run the program

2. I've included the preprocessed image data. You can download it from [here](https://drive.google.com/open?id=17LR9ssbENit-3vsEAM63FptNasB5AHrr). Now place the 5 files that you just downloaded with the folder with the .py file

3. Use "pip install package-name" to install the below packages

4. You need to have the following python packages installed
	* keras
	* tensorflow (Both CPU or GPU version should do)
	* PIL
	* numpy

5. You can modify the default hyparameters by modifying the variables between the '#' in the first few lines line

To run the program, navigate to the folder in command line and use the following command,
python BreastCancer.py

I've also included a pretrained model. To test your own image or one of the samples using it, paste the image in the folder with the .py file and rename it as 'my_image.jpg', then during execution choose to test your own image by following the on screen commands
