# PyTNet
A deep neural net Libraby Built on top of Pytorch

## **File Structure**

1. **Modles** - Contains the dnn models
   * NewResnetModel.py - Dawnbench mark 2019 winners model
   * QuizModel.py - Dense Net 
   * ResNetModel.py - Resnet 
   * S7Model.py - custom model
   * MaskDepthModel - model that estimates both mask and depth
   * DepthModel - Model that estimates depth
   * MaskModel - Model that estimates mask
   
2. **Dataset** - contains data related modules
   * **extract.py** - Unzips the data set for monocular depth estimation and segmentation
   * **MaskDepth.py** - It brings the depth estimation andsegmentation to dataset format and applies the given transformations.
   * **tinyimagenet** - It downloads the tiny imagenet data, mix train-test, split into the given ratio and returns train and test set of type dataset.
   
3. **Evaluation Metrics**
   * **Accuracy.py** - Implements the dice score for evaluation of mask and depth.
   * **loss.py** - Implementation of different loss functions.
   
4. **Results**
   * **showMnD** - displays the predicted and target images of mask and depth.
   
5. **Training**
   * **train_test_MnD.py** - training for depth estimation and segmentation.
   * **train_test.py** - training for object recognisation.
   
2. **Albumentation transforms** - Used for Image Agumentations. It is from Albumentations library.

3. **GradCam** - Implements gradCam of the given images and specified layer of the model.

4. **LrFinder** - It finds the Lr of given range.

5. **LR_Range_test** - It finds the best Lr for One Cyce Policy

6. **evaluate** - It evaluates the final test accuracy, classwise accuracy, plots the given curves, gives misclassified inages and plots misclassified images, 

7. **show_images** - Plots the given images of tensor for. Mainly used to visualise the train data.

8. **train_test** - Used to train the model.

9. **train_test_loader** - takes the train test data of type dataset, converts into data loader form, set the seed, check for the cuda availability.
