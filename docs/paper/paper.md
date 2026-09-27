# IEEESTEC paper — content source

<!--
Content source for docs/paper/IEEESTEC_paper.doc. The .doc is generated from this file by
docs/paper/build_paper.py and only the text changes; every paragraph keeps the template's style.

Markup understood by the build script:
  # / ## / ###  -> Heading 1 / Heading 2 / Heading 3 (the template numbers them itself)
  - item        -> bullet list
  [FIG n: file | width | caption]  -> image paragraph + "figure caption" paragraph
  [TABLE n: caption] followed by a pipe table -> "table head" + table in template table styles
  *text*        -> italic run, **text** -> bold run
  [n]           -> citation, kept verbatim
Every number is taken from results/results.csv (latest run per cell, raw_gray) or from
docs/report/main.tex, which traces each number to its source note.
-->

@title Controlled Comparison of Image Representations for Traffic Sign Classification Under Input Degradation

@author Mihajlo Madić
@author Faculty of Electronic Engineering
@author University of Niš
@author Niš, Serbia
@author mihajlo.madic@elfak.rs, ORCID: N/A

@abstract Traffic sign classification on the German Traffic Sign Recognition Benchmark (GTSRB) is widely regarded as solved, with convolutional networks exceeding 99% accuracy. For a deployed system, however, the relevant question is which image representation stays reliable under the degradations that dominate its operating conditions. This paper compares four representations, namely principal component analysis (PCA), histograms of oriented gradients (HOG), a bag of visual words over dense SIFT descriptors (BoVW) and a convolutional neural network (CNN), the last used both as a feature extractor and end-to-end. The linear classifier, the preprocessing and the track-disjoint data split are held fixed, so that differences in the results can be attributed to the representation alone. Each of the five configurations is evaluated on clean test data and under Gaussian noise, motion blur and gamma correction at five severity levels. On clean data the CNN leads (macro-F1 0.968) and PCA is last (0.731). Under strong noise the ranking inverts (Spearman correlation of −0.60 with the clean ranking), PCA and BoVW outperform both CNN variants on identical pixels, and HOG retains only 14.8% of its clean score. Blur and gamma produce no such inversion. Clean accuracy is therefore not a sufficient criterion for choosing a representation, and the choice should follow the degradation that dominates the application.

@keywords traffic sign recognition, image representation, robustness, image degradation, convolutional neural network, bag of visual words

# Introduction

Traffic sign recognition is a mature problem. On the German Traffic Sign Recognition Benchmark (GTSRB) [1], a committee of convolutional networks exceeded human performance more than a decade ago, with an accuracy above 99% [2], and a further point of clean accuracy says little about which method should be used in practice. A deployed recogniser does not see clean crops. A camera on a moving vehicle produces motion blur, a small sensor at dusk produces noise, and low sun or a tunnel exit push the exposure to its extremes. The question a system designer faces is therefore which representation of the sign image remains reliable under the degradation that dominates a given application, and a clean leaderboard does not answer it.

A complete traffic sign system is usually decomposed into detection and classification [3], [4]. For a precise scope, this work refines that decomposition into five modules, shown in Fig. 1. A detector localises candidate regions in the full frame, a tracker propagates them through subsequent frames with optical flow [5] to save computation, a recognition module assigns a class to each crop, a temporal aggregation module fuses the per-frame decisions for one physical sign, and a semantic module turns the recognised signs into constraints for the vehicle. This paper addresses only the recognition module, on crops whose boundaries are given by the dataset annotation.

[FIG 1: system_modules.png | column | The five modules of a traffic sign recognition system. This work addresses only the recognition module (M3), on crops given by the dataset annotation.]

The four representations compared here differ along several structural axes at once. They differ in whether the spatial layout of the sign is kept or discarded, in whether they describe the image globally or through local patches, and in whether they are designed by hand or learned [6]. A representation that discards layout should tolerate a shifted crop but lose information that distinguishes similar signs, and a representation built on image gradients should suffer when noise corrupts those gradients. Because the axes interact with different degradations in different ways, the ranking of the methods is expected to depend on the type of degradation. That interaction, rather than the best clean score, is the subject of this paper.

The contributions are as follows.

- A controlled comparison of PCA, HOG, BoVW and a CNN in which the classifier, the preprocessing, the data split and the evaluation protocol are held fixed, so that the representation is the only factor that varies.
- A grid of 80 evaluations (five configurations, sixteen conditions) showing that the ranking of the representations inverts under noise and does not invert under blur or gamma.
- A set of predictions, written down and dated before the first model was trained, and an explanation of each prediction that failed.
- A measurement of how much an image-level split inflates validation scores on GTSRB, the most common protocol error on this dataset.

# Related Work

GTSRB [1] contains 43 classes of German traffic signs cropped from video, and its companion GTSDB [4] provides the full frames for detection. A survey of vision-based sign detection and analysis is given in [3]. The four representations studied here come from different periods of computer vision. PCA as a representation for recognition goes back to Eigenfaces [7], HOG was introduced for pedestrian detection [8], BoVW quantises local SIFT descriptors [9] into a histogram of visual words [10], and CNNs learn the representation jointly with the classifier [2], [6]. An evaluation of several classical pipelines on GTSRB and on a Belgian dataset showed that hand-designed features combined with good classifiers were already close to the best reported accuracy [11].

The effect of image quality on deep networks has been studied on natural images, where noise and blur were found to degrade CNN classifiers sharply [12], and a benchmark of common corruptions has made robustness a standard evaluation axis [13]. These studies measure how much a given network degrades. The present work instead compares representations of different kinds under one fixed protocol, so that a difference in degradation can be attributed to the representation itself.

# Experimental Protocol

## Data and a Track-Disjoint Split

GTSRB provides 39,209 training and 12,630 test crops, whose sizes range from 15×15 to 250×250 pixels. The classes are imbalanced by a factor of 10.7 in the training set, so macro-averaged F1 (macro-F1), which weights all 43 classes equally, is used as the primary metric and accuracy as the secondary one.

The training images are grouped into tracks of 30 frames of the same physical sign, recorded as the vehicle approaches it. A random split by image places near-identical frames of one sign on both sides of the split. The training set is therefore divided into training and validation by track, 80% to 20% within each class, which yields 1,307 tracks, 31,379 training images and 7,830 validation images with no track on both sides. The track identifier in the file names restarts inside every class directory, so the track key has to be the pair of class and track, as the raw identifier alone collapses the 1,307 tracks into 75 groups.

The cost of ignoring the tracks was measured by training the same PCA model twice, once on the track-disjoint split and once on random image-level splits (three seeds). The image-level split inflates validation accuracy by 5.08 percentage points (pp) and validation macro-F1 by 9.58 pp, and in that split every validation image keeps a sibling frame in the training set. Macro-F1 inflates about twice as much as accuracy because leakage helps the rare classes most. A class with only seven tracks offers little variation to generalise across, so its held-out frames become close to a lookup. The metric chosen because of the imbalance is thus the one most distorted by leakage.

## Representations

The four representations are summarised in Table I. Each takes a grayscale crop resized to 48×48 pixels.

[TABLE I: Structural properties of the compared representations | w=26,36,16,22]
| Representation | Spatial layout | Locality | Learned |
| PCA | holistic, alignment-sensitive | global | no |
| HOG | rigid grid, layout kept | local | no |
| BoVW | orderless, layout discarded | local | vocabulary only |
| CNN | hierarchical, pooled | local | fully |

*PCA* projects the 2,304-dimensional image onto its 256 leading principal components, following Eigenfaces [7]. The projection is not whitened, so the features are the coordinates of an orthogonal projection. *HOG* [8] uses cells of 6×6 pixels, nine orientation bins and L2-Hys normalisation over blocks of 2×2 cells, giving 1,764 dimensions. *BoVW* computes SIFT descriptors [9] on a dense grid without a detector, since a detector returns a variable and often zero number of keypoints on crops this small, quantises them against a vocabulary of 1,000 words learned with k-means, and encodes each image as a power- and L2-normalised histogram of word counts [10]. *The CNN* has three convolutional blocks of 32, 64 and 128 channels with batch normalisation and max pooling, followed by a 128-dimensional fully connected layer and a softmax output, 0.88 M parameters in total. Global average pooling was deliberately not used, because it would make the CNN orderless and place two of the four representations on the same point of the layout axis.

Fig. 2 shows what each representation keeps of two signs. The PCA reconstruction preserves the global shape and brightness and smooths the fine detail, HOG keeps oriented edges on a rigid grid of cells, BoVW assigns a visual word to every keypoint and then discards the positions, and the last convolutional block of the CNN describes the whole sign on a 6×6 grid of learned channels.

[FIG 2: representations.png | column | What each representation keeps of two signs. PCA is the image rebuilt from its 256 coordinates, HOG the orientation histograms of its cells, BoVW the visual word at each keypoint (colours identify words and carry no order), and CNN the mean activation of the last convolutional block.]

Because the classifier is held fixed, the CNN is evaluated twice. As a feature extractor, its 128-dimensional penultimate layer is passed to the same linear classifier as the other methods. End-to-end, it keeps its own softmax output. This gives five configurations, which are referred to as PCA, HOG, BoVW, CNN-feat and CNN-e2e.

## Controlled Design

Every configuration except CNN-e2e uses the same classifier, a linear support vector machine (SVM). A kernel SVM would scale quadratically with the 31,379 training samples, and a weaker classifier also keeps the differences between the methods closer to differences between the representations. The regularisation parameter C is selected for each method on validation macro-F1. A fixed classifier means the same estimator and the same selection rule, and does not mean the same value of C, since PCA features are not internally normalised and span a much wider range than the others. Class weighting, which changes the objective itself, is fixed to balanced for every method. The CNN is trained with Adam and early stopping on validation macro-F1, and the weights of the best epoch are kept.

All hyperparameters, including the number of PCA components, the HOG cell size and the BoVW descriptor geometry and vocabulary size, are selected on the validation set only. All methods are trained on the training split alone, never on training and validation together, because the CNN needs the validation set for early stopping. The test set is used exactly once, in the final evaluation grid. The input is fixed to raw grayscale for every method, because letting each method choose its own preprocessing would mix the effect of the preprocessing into every comparison.

## Controlled Degradations

The test set is evaluated clean and under three degradations at five levels each, shown in Fig. 3. Additive Gaussian noise has a standard deviation σ of 0, 5, 10, 20 and 40 gray levels. Motion blur uses a line kernel of length k of 0, 3, 5, 9 and 15 pixels at a random angle, sampled at sub-pixel resolution so that the blur strength does not depend on the angle. Gamma correction uses γ of 0.4, 0.7, 1.0, 1.5 and 2.5, where γ = 1.0 is the identity and lies in the middle of the range. The three were chosen because they attack different properties of the image. Noise adds high-frequency energy, blur removes it, and gamma leaves the geometry untouched and changes only the intensities. Together with the clean condition this gives 16 conditions and 80 evaluations.

[FIG 3: degradation_contact_sheet.png | column | The three degradations at their five levels, applied to one 48×48 model input. The identity level of each row is outlined, and for gamma it lies in the middle of the row.]

The degradations are applied to the preprocessed 48×48 model input. What is injected is then, by construction, the degradation that preprocessing did not remove, and a given level means the same condition for every sign size. The random stream of each degraded image is derived from the image itself, so all five methods are evaluated on identical degraded pixels. No degraded image is ever used for training.

## Predictions Recorded in Advance

Before the first model was trained, the expected most and least robust representation under each degradation was written down and dated, with a justification derived from the properties in Table I. The central prediction was a crossing. PCA was expected to be the most robust under noise, since most of the noise energy falls outside its 256-dimensional subspace, and the least robust under gamma, since a global intensity remap moves every projection coefficient. HOG was expected to show the opposite pattern, because differentiation amplifies noise while block normalisation cancels contrast changes. These predictions are compared with the outcomes in Section V.

# Results

## Clean Test Data

Table II lists the results on the clean test set. The two CNN configurations lead with a macro-F1 near 0.97, HOG and BoVW follow, and PCA is last at 0.731. The ordering follows the degree to which the representation is learned. The CNN-feat configuration reaches the highest accuracy but a lower macro-F1 than CNN-e2e, because its features were optimised for a softmax output and the linear SVM on top of them loses most on the rare, difficult classes. The time column is the median inference time for a single image on a CPU, including feature extraction, which is the relevant figure for a camera that delivers one frame at a time.

[TABLE II: Results on the clean test set | w=24,20,20,15,21]
| Method | Accuracy | Macro-F1 | Dim. | Time (ms) |
| CNN-e2e | 0.979 | 0.968 | 128 | 1.40 |
| CNN-feat | 0.980 | 0.965 | 128 | 2.04 |
| HOG | 0.924 | 0.906 | 1764 | 0.98 |
| BoVW | 0.914 | 0.860 | 1000 | 2.79 |
| PCA | 0.793 | 0.731 | 256 | 0.33 |

## Robustness to Degradation

Fig. 4 shows, for each degradation, the macro-F1 of every method as a percentage of its own clean macro-F1. This retention measures how fast a method degrades, independently of where it starts, and the absolute scores are given alongside it in Table III.

*The ranking inverts under noise.* On clean data the order is CNN-e2e, CNN-feat, HOG, BoVW, PCA. At σ = 40 it becomes BoVW, PCA, CNN-feat, CNN-e2e, HOG, and the Spearman rank correlation with the clean order is −0.60. The two methods that are last and fourth on clean data become the two best, and both outperform the CNN on identical pixels, with a macro-F1 of 0.561 for PCA and 0.564 for BoVW against 0.425 and 0.428 for the two CNN configurations. The 0.3 pp gap between PCA and BoVW is within the resolution of a single-seed study and is not interpreted. The crossover happens early. At σ = 20 BoVW already leads both CNNs (0.712 against 0.693 and 0.691), and at σ = 40 PCA also has the highest accuracy of all methods (0.610). In retention PCA is clearly the most robust, keeping 76.7% of its clean score, while HOG keeps only 14.8%. HOG loses a fifth of its score already at σ = 5, a level at which the noise is hardly visible (Fig. 3). Noise is spread evenly over all orientations, and in the low-contrast cells of the background, which contain no real gradient, it dominates the orientation histograms completely.

[TABLE III: Macro-F1 at the strongest degradation levels, absolute and as retention (%) | w=20,13,13,15,20,19]
| Level | PCA | HOG | BoVW | CNN-feat | CNN-e2e |
| σ = 40 | 0.561 | 0.134 | 0.564 | 0.428 | 0.425 |
| retained | 76.7 | 14.8 | 65.6 | 44.3 | 43.8 |
| k = 15 | 0.330 | 0.216 | 0.178 | 0.355 | 0.384 |
| retained | 45.1 | 23.9 | 20.7 | 36.8 | 39.6 |
| γ = 2.5 | 0.513 | 0.808 | 0.819 | 0.832 | 0.834 |
| retained | 70.1 | 89.2 | 95.3 | 86.2 | 86.1 |

[FIG 4: robustness_curves.png | page | Macro-F1 retained relative to each method's own clean score, under (a) Gaussian noise, (b) motion blur and (c) gamma correction. The gamma axis is logarithmic, so that the levels lie symmetrically around the identity at γ = 1.0.]

*The inversion is specific to the degradation.* Under blur and under gamma the ranking follows the clean ranking, with Spearman correlations of +0.70 and +0.90. The CNN remains the best in absolute terms under blur, and PCA, which is the most robust in retention, is still behind both CNNs at k = 15. The claim of the paper is therefore narrower than a general instability of the ranking. The ranking depends on the degradation, and among the three degradations studied it is noise that inverts it.

*Gamma has three different winners.* On retention BoVW is the most robust under γ = 2.5 (95.3%), on absolute macro-F1 the CNN remains best (0.834), and on absolute accuracy BoVW is best (0.887, against 0.841 for HOG and 0.811 for CNN-e2e). Gamma costs the CNN mostly on rare classes, which macro-F1 weights equally and accuracy does not. Every best and worst method in Table III is the same whether retention is computed on macro-F1 or on accuracy, but a statement about robustness under gamma has to name the metric it refers to.

## Accuracy by Sign Size

Fig. 5 partitions the clean test predictions by the height of the sign in pixels. The smallest bucket is the most common case, since 47.6% of the training images are below 32 pixels. Both CNN configurations stay above 0.96 in every bucket, and CNN-feat is even more accurate below 32 pixels (0.975) than above 72 pixels (0.960). Among the hand-designed methods HOG loses the most from the largest to the smallest signs (8.1 pp), while BoVW, which loses only 1.0 pp, is the most size-stable method in the study. PCA is the least accurate in every bucket and peaks at medium sizes. A plausible cause, not tested here, is that PCA is the only representation without internal normalisation that could absorb the difference in sharpness between downscaled large crops and upscaled small ones.

[FIG 5: accuracy_by_size.png | column | Clean-test accuracy by the height of the sign, with the number of test images in each bucket. The vertical axis starts at 0.70.]

## The CNN Makes Different Errors

The methods also differ in which signs they confuse. For each method the ten most frequent confusion pairs on clean data were extracted and compared. The three hand-designed methods share two to four of their ten pairs with each other, and the two CNN configurations share six with each other, but only zero to two with any hand-designed method. CNN-feat shares none of its ten pairs with PCA and none with HOG. The CNN is therefore not a better version of the same classifier, and its errors are largely complementary to those of the hand-designed methods.

# Discussion

## Predictions and Outcomes

Table IV compares the predictions with the outcomes. Eight of the ten predictions held, including the central crossing, since PCA is the most robust under noise and the least robust under gamma, and HOG is the least robust under noise. The two failures are the more informative rows, and both have an identifiable cause.

## The Kind of Normalisation Matters

The gamma prediction assumed that SIFT descriptor normalisation gives BoVW protection similar to, and slightly weaker than, the block normalisation of HOG. It turned out to be the strongest protection in the study. The two normalisations are of a different kind. HOG normalises over a block of 2×2 cells with a linear L2 norm, which cancels a linear change of contrast exactly (a measured mean descriptor change of 0.0000) but a gamma curve only partly (0.0339). SIFT normalises each 128-dimensional descriptor, clips every component at 0.2 and normalises again [9]. The clipping is a non-linearity that suppresses exactly the large gradient components a gamma curve inflates, so a monotone intensity transform is close to its best case.

[TABLE IV: Predictions recorded before training and their outcomes | w=23,35,30,12]
| Degradation | Prediction | Outcome | Held |
| Noise | PCA most robust | 76.7%, first | yes |
| Noise | HOG least robust | 14.8%, last | yes |
| Noise | BoVW between PCA and HOG | 65.6%, second | yes |
| Blur | PCA most robust | 45.1%, first | yes |
| Blur | BoVW least robust | 20.7%, last | yes |
| Blur | HOG ahead of BoVW | 23.9% vs 20.7% | yes |
| Gamma | PCA least robust | 70.1%, last | yes |
| Gamma | HOG most robust | BoVW first (95.3%) | no |
| Small signs | CNN most robust | 0.975 below 32 px | yes |
| Small signs | BoVW least robust | BoVW most stable | no |

The same property explains a second unexpected result. Heavy blur was expected to make the BoVW descriptors collapse onto a few visual words, but the number of distinct words per image falls by only 18% from k = 0 to k = 15, and the similarity between histograms of different classes stays almost flat (0.232 to 0.244). Blur reduces the magnitude of the gradients while their direction still varies, and the normalised descriptor keeps the direction. BoVW is nevertheless the least robust method under blur, as predicted, for the second reason given in the prediction. When blur merges the edges of a sign, BoVW has no spatial layout to fall back on.

## Tuning Changed the Method

The prediction for small signs stated that dense SIFT would have too little local support on an upsampled crop. That reasoning was correct for the configuration in the original plan, with 12-pixel descriptors on a 6-pixel grid, which reached a validation macro-F1 of only 0.35. Tuning the descriptor geometry on the validation set moved BoVW to 0.88, an improvement of 53.5 pp and larger than any difference between the methods in this study. The decisive factor was the descriptor scale. At a fixed number of keypoints, reducing the scale from 6 to 4 pixels was worth 12.3 pp, while a 2.25 times denser grid at a fixed scale was worth only 2.2 pp. The selected 2-pixel descriptor needs almost no local support, so BoVW became the most size-stable method in the study. A weakness attributed to a method is often a property of its configuration, and a comparison that uses default parameters risks measuring the configuration instead.

## The Cost of Discarding Layout

Because the dense keypoints lie on a regular grid, the position of every descriptor is known and BoVW simply discards it. This allows a controlled experiment in which the descriptors, the vocabulary and the classifier stay the same and only the recording of position changes. Adding a spatial pyramid [14] over the same descriptors, with a vocabulary of 500 words, raises validation macro-F1 from 0.814 to 0.897 with a 2×2 grid and to 0.918 with a 4×4 grid, an increase of 10.4 pp that brings BoVW within 0.1 pp of HOG. The gap between BoVW and HOG is therefore almost entirely the layout information. A permutation test makes the orderlessness exact, since randomly permuting the positions of the keypoints leaves the plain BoVW histogram identical to the last bit. The same property makes a testable prediction for future work. BoVW should be the most robust method to a shifted or rescaled bounding box, which cannot be simulated honestly on GTSRB, because its images are already cropped with a median margin of 16.7% and a 40% enlargement would run off the image for 72% of them. Full road scenes, as in GTSDB [4], are the appropriate instrument.

# Conclusion

Five configurations of four image representations were compared on GTSRB under a protocol in which only the representation varies, on clean data and under 15 levels of controlled degradation. The results support a practical view of how a representation should be chosen.

The CNN is the right default baseline. It has the best clean score (macro-F1 0.968), the best absolute score under blur and on macro-F1 under gamma, and the highest accuracy on signs smaller than 32 pixels (0.975). Its learning rate was not tuned, so these figures are a lower bound. When the operating conditions are unknown, the CNN is the representation to start from.

Clean accuracy does not, however, predict robustness, and there are conditions in which another representation is the better choice.

- Where sensor noise dominates, as in low light, PCA and BoVW are the better choice. Both outperform the CNN in absolute macro-F1 from σ = 20 onwards, and at σ = 40 PCA keeps 76.7% of its clean score against the CNN's 44%.
- Where exposure is extreme and accuracy is the relevant measure, BoVW is the most accurate method at γ = 2.5 and keeps 95.3% of its clean score, owing to its non-linear descriptor normalisation.
- Where single-image latency or memory are tight, PCA (0.33 ms) and HOG (0.98 ms) are cheaper than the CNN (1.40 ms), although both give up a large part of the clean score.
- HOG should be avoided wherever noise is expected, since it loses a fifth of its score at noise that is barely visible and keeps only 14.8% at σ = 40.

The results also suggest a mixture of methods. The errors of the CNN overlap very little with those of the hand-designed methods, and the methods are robust to different degradations. This motivates two designs. The first is a condition-aware selection, in which the noise level of the input is estimated and noisy crops are routed to a noise-robust branch such as PCA or BoVW, while all other crops go to the CNN. The second is a late-fusion ensemble of the CNN with BoVW, whose errors and robustness profiles are the most complementary. Neither design was evaluated in this work, and both are stated as hypotheses motivated by the measurements, to be tested next.

The general conclusion is that the representation of a traffic sign should be chosen according to the degradation that dominates the deployment, since the ranking on clean data can reverse under exactly the conditions a real system faces.

@references
[1] J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel, “Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition,” Neural Networks, vol. 32, pp. 323–332, 2012.
[2] D. Cireşan, U. Meier, J. Masci, and J. Schmidhuber, “Multi-column deep neural network for traffic sign classification,” Neural Networks, vol. 32, pp. 333–338, 2012.
[3] A. Møgelmose, M. M. Trivedi, and T. B. Moeslund, “Vision-based traffic sign detection and analysis for intelligent driver assistance systems: Perspectives and survey,” IEEE Trans. Intell. Transp. Syst., vol. 13, no. 4, pp. 1484–1497, 2012.
[4] S. Houben, J. Stallkamp, J. Salmen, M. Schlipsing, and C. Igel, “Detection of traffic signs in real-world images: The German Traffic Sign Detection Benchmark,” in Proc. Int. Joint Conf. Neural Networks (IJCNN), 2013.
[5] B. D. Lucas and T. Kanade, “An iterative image registration technique with an application to stereo vision,” in Proc. Int. Joint Conf. Artificial Intelligence (IJCAI), 1981, pp. 674–679.
[6] Y. Bengio, A. Courville, and P. Vincent, “Representation learning: A review and new perspectives,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 35, no. 8, pp. 1798–1828, 2013.
[7] M. Turk and A. Pentland, “Face recognition using eigenfaces,” in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 1991, pp. 586–591.
[8] N. Dalal and B. Triggs, “Histograms of oriented gradients for human detection,” in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2005, pp. 886–893.
[9] D. G. Lowe, “Distinctive image features from scale-invariant keypoints,” Int. J. Comput. Vis., vol. 60, no. 2, pp. 91–110, 2004.
[10] G. Csurka, C. Dance, L. Fan, J. Willamowski, and C. Bray, “Visual categorization with bags of keypoints,” in Proc. ECCV Workshop on Statistical Learning in Computer Vision, 2004, pp. 1–22.
[11] M. Mathias, R. Timofte, R. Benenson, and L. Van Gool, “Traffic sign recognition – How far are we from the solution?,” in Proc. Int. Joint Conf. Neural Networks (IJCNN), 2013.
[12] S. Dodge and L. Karam, “Understanding how image quality affects deep neural networks,” in Proc. 8th Int. Conf. Quality of Multimedia Experience (QoMEX), 2016.
[13] D. Hendrycks and T. Dietterich, “Benchmarking neural network robustness to common corruptions and perturbations,” in Proc. Int. Conf. Learning Representations (ICLR), 2019.
[14] S. Lazebnik, C. Schmid, and J. Ponce, “Beyond bags of features: Spatial pyramid matching for recognizing natural scene categories,” in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2006, pp. 2169–2178.
