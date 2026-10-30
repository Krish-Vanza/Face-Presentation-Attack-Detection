# Face Presentation Attack Detection

## A guide to the project, results, and limitations


## 1. The Problem

A face authentication camera should accept a genuine, live person and reject an image presented to fool it. The attack could be a printed photograph, a video replay on a phone or tablet, a cut-out photo, a mask, or an artificial image. This kind of attack is called a **presentation attack**. A system that checks whether a face is live or spoofed is called a **presentation attack detector**, usually shortened to **PAD**.

Assignment 39 asks for a framework that can distinguish **Live** from **Spoof**, test it on different data sources, and examine whether small adversarial changes can fool it. It names OULU-NPU as the training dataset, Replay-Mobile as the validation dataset, and CASIA-FASD plus CelebA-Spoof as test datasets. It also asks for confusion matrices, performance measures, and a comparison between normal images and adversarial images.

This project implements a small image-based PAD model and completes a clean and adversarial evaluation. It is an educational baseline, not a deployable face authentication product. The most important result is that a model can look very successful on a validation split yet fail on images from other datasets.

### PAD is not face recognition

Face recognition asks **who is this person?** PAD asks **does this image appear to show a live person, or a presentation attack?** This project only predicts one of two labels: Live or Spoof. It does not identify a person, verify an identity, or make an access-control decision by itself.

## 2. What is the vocabulary used

- **Live:** a genuine face image labelled as coming from a live person.
- **Spoof:** an image labelled as an attack or artificial presentation.
- **Training set:** labelled examples used to update the model's parameters.
- **Validation set:** separate examples used to monitor training and select a checkpoint. They should not be described as an untouched final test if they were used to choose a model.
- **External test:** data from a different dataset that was not used for training. It checks whether the learned patterns transfer to a new source.
- **Adversarial example:** an input changed using the model's gradients to make its prediction less reliable. A small numeric change can still alter the model's decision.
- **Checkpoint:** a saved copy of model weights from one training epoch. This experiment selected the checkpoint with the best OULU validation balanced accuracy.

## 3. Datasets 

The experiment used a subject-separated OULU subset for scored validation and clearly treats Replay-Mobile as unavailable for numeric scoring.

| Dataset | Available labelled data | Role in this experiment |
|---|---:|---|
| OULU-NPU local subset | 1,701 images: 343 Live and 1,358 Spoof, from 20 subject IDs | Training and subject-held-out validation |
| Replay-Mobile local files | 39 MP4 videos; no usable Live/Spoof labels found | Not scored |
| CASIA-FASD test split | 65,786 images: 10,128 Live and 55,658 Spoof | Full local test split, clean and FGSM evaluation |
| CelebA-Spoof intra-test labels | 67,170 label rows: 19,923 Live and 47,247 Spoof | A balanced sample of 1,000 per class, 2,000 total |

### OULU labels and the split

The local OULU folders use `true` for Live and `false` for Spoof. The notebook maps these to class 0 (Live) and class 1 (Spoof). It reads the subject ID from the third underscore-separated field of each filename. The fixed-seed split keeps all images of one subject together, so a subject cannot appear in both training and validation. This reduces the risk that the model is rewarded for memorizing the same person's appearance across both splits.

The split used 1,361 training images from 16 subjects: 276 Live and 1,085 Spoof. The validation set used 340 images from four subjects: 67 Live and 273 Spoof. The local subset did not include official OULU protocol split files, so these numbers describe the available local copy, not the complete official OULU-NPU benchmark protocol.

### Why CelebA-Spoof was sampled

The labelled CelebA-Spoof test list contained 67,170 rows. Rather than load a much larger dataset, the notebook chose 1,000 Live and 1,000 Spoof images using a fixed random seed. This makes that evaluation balanced and practical to run. It does not mean the original dataset is balanced, and its results should not be treated as estimates over every CelebA-Spoof test image.

## 4. The solution and the training approach

The solution is a compact **convolutional neural network (CNN)**. A CNN is suited to images because small filters can learn patterns such as edges, textures, and local changes in appearance, then combine them into more useful visual features. The design is intentionally modest for a beginner project and a consumer graphics card.

### Image preparation

Images are read as RGB, resized to 128 by 128 pixels, and scaled into the zero-to-one range. Mini-batches are used so the entire image collection does not need to fit in memory at once. During training, random horizontal flips and small brightness and contrast changes provide mild augmentation. Validation and test images are not augmented.

The training set contains many more Spoof than Live examples. The sampler uses square-root inverse-frequency weights to show Live images more often than plain random sampling would. This partially offsets the class imbalance without forcing each training batch to be exactly half Live and half Spoof.

### CNN layout

1. The input is a three-channel RGB image with dimensions 128 by 128.
2. The first convolution block uses two 3 by 3 convolutions with 24 channels, with batch normalization, ReLU activation, and max pooling.
3. The next two blocks use the same pattern with 48 and then 96 channels.
4. Adaptive average pooling reduces the feature map to 4 by 4 while retaining some spatial texture information.
5. A 64-unit dense layer, ReLU, and dropout with probability 0.4 lead to two output scores: Live and Spoof.

A convolutional layer learns small filters. **Batch normalization** keeps intermediate values in a more stable range during training. **ReLU** adds a simple non-linear decision. **Max pooling** reduces the spatial size while keeping strong responses. **Dropout** randomly hides some units during training to reduce reliance on any one feature.

### Training settings

- Image size: 128 by 128 RGB
- Batch size: 16
- Training duration: 20 epochs
- Optimizer: AdamW
- Learning rate: 0.0005
- Weight decay: 0.0001
- Loss: cross-entropy for the two classes
- Random seed: 42
- Augmentation: horizontal flip and small brightness/contrast changes
- Imbalance handling: square-root weighted sampler
- Checkpoint selection: highest OULU validation balanced accuracy

The notebook is written to use CUDA when available and fall back to CPU. The saved evaluation table does not record which device ran this particular training session, so the report does not claim a specific device.

### What the training curve says

Training loss fell from approximately 0.4837 to 0.0736 over the run, which means the model fit the training examples more closely. Validation balanced accuracy moved up and down across epochs, and the selected checkpoint was epoch 19 with about 98.7% balanced accuracy. The falling training loss is not proof of generalization. A model can become better at the training data while remaining weak on new cameras, backgrounds, attack types, or image-processing pipelines.

## 5. How the model was evaluated

Each external test was evaluated first on clean images and then on adversarially perturbed images. For the adversarial test, the notebook uses the **Fast Gradient Sign Method (FGSM)** with epsilon 8/255 on the zero-to-one pixel scale. FGSM uses the gradient of the loss with respect to the input to choose a small change that is likely to increase the model's error. Pixel values are clipped to remain in range after the change.

This test demonstrates sensitivity to an attack. It is **not a defense**: no adversarial examples were used to train the model, and no adversarial training or other robustness method was added.

## 6. Metrics: what to read and why

### Accuracy

Accuracy is the proportion of all images classified correctly:

**Accuracy = correct predictions / all predictions.**

It is easy to understand, but it can hide a weak minority class. If most examples are Spoof, a model can get a high accuracy by predicting Spoof too often while failing to recognize genuine Live images.

### Recall and balanced accuracy

**Live recall** is the share of actual Live images predicted as Live. Low Live recall means genuine people are often rejected as Spoof.

**Spoof recall** is the share of actual Spoof images predicted as Spoof. Low Spoof recall means attacks are often accepted as Live.

**Balanced accuracy** is the average of Live recall and Spoof recall. It gives both classes equal importance even when one class has many more examples. This is why CASIA-FASD's accuracy and balanced accuracy tell different stories.

### Confusion matrix

Every confusion matrix in this project has actual labels in rows and predicted labels in columns. With the class order Live, Spoof:

| | Predicted Live | Predicted Spoof |
|---|---:|---:|
| Actual Live | correctly accepted Live images | Live images incorrectly rejected |
| Actual Spoof | spoof images incorrectly accepted as Live | correctly rejected Spoof images |

The upper-right cell represents genuine users rejected as Spoof. The lower-left cell represents missed attacks, which is particularly important for an anti-spoof system.

### Precision and F1

The notebook also calculates precision and F1 for each class. Precision asks: among the images predicted as this class, what fraction truly belongs to it? F1 combines precision and recall into one score using their harmonic mean. These measures can give more detail, but the report emphasizes accuracy, balanced accuracy, and per-class recall because they are easier to explain and directly show the two main types of error.

## 7. Results and interpretation

The following values are from the saved final evaluation metrics and agree with the displayed confusion matrices.

| Dataset and condition | Images | Accuracy | Balanced accuracy | Live recall | Spoof recall |
|---|---:|---:|---:|---:|---:|
| OULU validation, clean | 340 | 98.8% | 98.7% | 98.5% | 98.9% |
| CASIA-FASD, clean | 65,786 | 72.5% | 49.7% | 16.7% | 82.7% |
| CASIA-FASD, FGSM | 65,786 | 28.0% | 16.6% | 0.2% | 33.1% |
| CelebA-Spoof sample, clean | 2,000 | 54.5% | 54.5% | 50.7% | 58.2% |
| CelebA-Spoof sample, FGSM | 2,000 | 5.7% | 5.7% | 1.0% | 10.4% |

### OULU validation: excellent score, but not a final generalization result

The OULU holdout reached 98.8% accuracy and 98.7% balanced accuracy. Its confusion matrix is [[66, 1], [3, 270]]. That means 66 of 67 Live images and 270 of 273 Spoof images were classified correctly. This is strong performance on this validation split, but it is not an independent final test: the same 340-image set was used to choose the best epoch. It is also small, especially the 67 Live examples, so a few changed predictions would shift the score noticeably.

### CASIA-FASD clean: ordinary accuracy is misleading

CASIA-FASD has 55,658 Spoof images out of 65,786, or about 84.6%. A trivial system that always predicts Spoof would therefore achieve about 84.6% accuracy. The CNN achieved only 72.5%, below that simple majority-class baseline. Its balanced accuracy was 49.7%, near chance, because Live recall was just 16.7% while Spoof recall was 82.7%.

The clean CASIA confusion matrix is [[1,687, 8,441], [9,638, 46,020]]. The model recognized many Spoof images but missed most Live images. Its overall accuracy should not be described as good performance: the class imbalance and low Live recall make that conclusion false.

### CelebA-Spoof clean: close to chance

The balanced CelebA-Spoof sample produced 54.5% accuracy and balanced accuracy. Live recall was 50.7%, and Spoof recall was 58.2%. Because the sample has equal numbers of both classes, its accuracy and balanced accuracy match. This is only modestly better than a 50% balanced-class guess and indicates weak transfer from OULU to CelebA-Spoof.

### FGSM: strong loss of performance

Under FGSM, CASIA-FASD balanced accuracy fell from 49.7% to 16.6%, and CelebA-Spoof fell from 54.5% to 5.7%. On CelebA-Spoof, only 10 of 1,000 Live images and 104 of 1,000 Spoof images remained correctly classified. The model is highly sensitive to the tested perturbation. Since no robustness training was performed, these results show a weakness rather than a defense success.

## 8. What worked and what did not

### What worked

- The notebook runs a complete image-classification pipeline: labelled data loading, subject-separated validation, training, external testing, confusion matrices, and FGSM evaluation.
- Grouping OULU images by subject for the split reduces a common form of subject leakage.
- Weighted sampling addresses the major Live/Spoof imbalance in training and helped move away from the earlier baseline that predicted almost everything as Spoof.
- The small CNN, 128-pixel images, and batch size 16 keep the experiment practical for limited hardware.
- The clean and adversarial results make the generalization and robustness limitations visible instead of hiding them behind the high OULU validation score.

### What did not work

- The first baseline collapsed toward predicting Spoof only. The training set has roughly four times as many Spoof images as Live images, so plain training can favor the majority class.
- Better balancing and a stronger small CNN improved the experimental setup, but they did not solve cross-dataset generalization. The clean external balanced accuracy remained near chance.
- The OULU validation result is optimistic because the holdout also selected the checkpoint. A separate development set and untouched final test would give a less biased estimate.
- The local Replay-Mobile videos could not be evaluated because labels were unavailable. The assignment's specified validation source is therefore missing from the numeric results.
- FGSM performance collapsed. The model has no adversarial defense and should not be described as robust.
- The local OULU data copy lacks official protocol splits. The experiment is not a full reproduction of the official OULU-NPU evaluation.

### Why results differ across datasets

The CNN was trained only on one local OULU subset. External datasets can differ in camera sensors, resolution, lighting, compression, face alignment, background, attack media, and how images were extracted from videos. A CNN may learn features that identify its training source instead of stable cues of liveness. A subject-separated split helps measure unseen people, but it does not by itself solve changes in devices and datasets.

## 9. Important consistency note about the notebook

The final notebook contains a prose cell titled **“What it means”** that lists older figures: OULU 98.2%, CASIA balanced accuracy 51.8%, CelebA-Spoof 58.2%, and FGSM accuracy 10.4% and 3.7%. These values do not match the final saved metrics table. The authoritative values used in this guide and in the report are the saved evaluation results:

- OULU validation: 98.8% accuracy and 98.7% balanced accuracy.
- CASIA-FASD clean: 72.5% accuracy and 49.7% balanced accuracy.
- CelebA-Spoof clean: 54.5% accuracy and 54.5% balanced accuracy.
- CASIA-FASD FGSM: 28.0% accuracy and 16.6% balanced accuracy.
- CelebA-Spoof FGSM: 5.7% accuracy and 5.7% balanced accuracy.

For a consistent submission, that old prose cell should be corrected before presenting or submitting the notebook. The CSV values, report tables, and confusion matrices agree with one another.

## 10. How to state the conclusion honestly

A fair conclusion is: **The CNN trained successfully and achieved a high score on the available OULU subject holdout, but its performance did not transfer to the external datasets and it was not robust to FGSM.** The experiment demonstrates the implementation of a PAD pipeline and provides evidence about its limits. It does not show that a reliable or attack-resistant authentication framework has been achieved.

## 11. Useful next steps

1. Obtain labelled Replay-Mobile files and run the intended validation protocol.
2. Use separate training, checkpoint-selection, and final-test data. Do not select a checkpoint on the same set reported as the final result.
3. Use official OULU protocol splits if the assignment requires benchmark-comparable results.
4. Add more labelled subjects and attack types, and check class-wise results after every change.
5. Test an appropriately small pretrained model only if there is time and hardware capacity.
6. Explore adversarial training or another robustness method, then evaluate on held-out adversarial examples. Robustness must be measured after a defense is trained; merely generating FGSM examples is not a defense.
7. For a more complete PAD study, report attack-oriented measures such as APCER, BPCER, and ACER if the required protocols and labels are available.

## 12. Short glossary

| Term | Plain meaning |
|---|---|
| CNN | A neural network that learns filters and patterns from images |
| PAD | Presentation attack detection: classify an input as Live or Spoof |
| Epoch | One pass through the training examples |
| AdamW | An optimizer that updates model weights during training |
| Cross-entropy | A loss function for comparing predicted class scores with known labels |
| Data augmentation | Small changes to training images that encourage the model to handle variation |
| Class imbalance | One label has many more examples than another |
| FGSM | A gradient-based method for creating adversarial image changes |
| Generalization | How well a model works on examples different from its training data |
| Dataset shift | Differences in image source or conditions between training and evaluation data |

## References

1. CS414 Biometrics Projects, Assignment 39: Face Authentication under Presentation and Adversarial Attacks, course handout.
2. Z. Boulkenafet, J. Komulainen, L. Li, X. Feng, and A. Hadid, “OULU-NPU: A Mobile Face Presentation Attack Database with Real-World Variations,” IEEE International Conference on Automatic Face & Gesture Recognition, 2017.
3. Z. Zhang, J. Yan, S. Liu, Z. Lei, D. Yi, and S. Z. Li, “A Face Antispoofing Database with Diverse Attacks,” International Conference on Biometrics, 2012.
4. Y. Zhang et al., “CelebA-Spoof: Large-Scale Face Anti-Spoofing Dataset with Rich Annotations,” European Conference on Computer Vision, 2020.
5. I. J. Goodfellow, J. Shlens, and C. Szegedy, “Explaining and Harnessing Adversarial Examples,” International Conference on Learning Representations, 2015.
