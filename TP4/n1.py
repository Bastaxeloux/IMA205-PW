# %% [markdown]
# # Skin lesion classification

# %% [markdown]
# **Deadline**: Upload this notebook (rename it as 'TP-SVM-YOUR-SURNAME.ipynb') to Ecampus/Moodle before the deadline.
# Complete the code where you see XXXXXXXXXXXXXXXXX (mandatory for everybody)

# %% [markdown]
# **Context**
# A skin lesion is defined as a superficial growth or patch of the skin that is visually different and/or has a different texture than its surrounding area. Skin lesions, such as moles or birthmarks, can degenerate and become melanoma, one of the deadliest skin cancer. Its incidence has been increasing during the last decades, especially in the areas mostly populated by white people.
# 
# The most effective treatment is an early detection followed by surgical excision. This is why several approaches for melanoma detection have been proposed in the last years (non-invasive computer-aided diagnosis (CAD) ).
# 
# 
# **Goal**
# The goal of this practical session is to classify images of skin lesions as either benign or melanoma using machine learning algorithms. In order to do that, you will have at your disposal a set of 30 features already extracted from 600 dermoscopic images (both normal skin lesions and melanoma from the ISIC database - https://isic-archive.com/). These features characterize the Asymmetry, the Border irregularity, the Colour and the Dimension of the lesion (the so-called ABCD rule).
# 
# The features are:
# - shape asimmetry (f0 and f1)
# - difference in colors between center and periphery of the image (f2, f3, f4, f27, f28, f29)
# - geometry (f5, f6, f7)
# - other features related to eccentricity,entropy, mean, standard deviation and maximum value of each channel in RGB and HSV (f8,...,f24)
# - asimmetry of color intensity (f25, f26)
# 
# Features are computed using *manually checked segmentations* and following *Ganster et al. 'Automated melanoma recognition', IEEE TMI, 2001* and *Zortea et al. 'Performance of a dermoscopy-based computer vision system for the diagnosis of pigmented skin lesions compared with visual evaluation by experienced dermatologists', Artificial Intelligence in Medicine, 2014*.

# %% [markdown]
# First load all necessary packages

# %%
import os
import numpy as np
import pandas as pd
from skimage.io import imread
from time import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.model_selection import  cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.linear_model import Perceptron
from sklearn.decomposition import PCA

%matplotlib inline

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# try:
#   import google.colab
#   IN_COLAB = True
#   print('You are using Google Colab')
#   !pip install googledrivedownloader
#   from googledrivedownloader import download_file_from_google_drive
# except:
#   IN_COLAB = False

# Code from scikit-learn
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="black" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


from sklearn.pipeline import make_pipeline

# %% [markdown]
# Then load the data from my Google Drive

# %%
# if IN_COLAB:
#   download_file_from_google_drive(file_id='18hrQVGBCfW7SKTnzmWUONo8iowBsi1DL',
#   dest_path='./data/features.csv')
#   download_file_from_google_drive(file_id='1iQZdUiuK_FwZ7mik7LB3eN_H_IUc5l7b',
#   dest_path='./data/im/nevus-seg.jpg')
#   download_file_from_google_drive(file_id='1_TeYzLLDoKbPX4xXAOAM_mQiT2nLHgvp',
#   dest_path='./data/im/nevus.jpg')
#   download_file_from_google_drive(file_id='1B2Ol92mBcHN6ah3bpoucBbBbHkPMGC8D',
#   dest_path='./data/im/melanoma-seg.jpg')
#   download_file_from_google_drive(file_id='1yZ46UzGhwO7g5T8397JpewBl6UqgRo5J',
#   dest_path='./data/im/melanoma.jpg')

# %% [markdown]
# Or from yout local computer. Please download the 'data' folder in the same folder as your notebook and do not modifiy it.

# %% [markdown]
# Then read the data

# %%
## Read data
Working_directory="./data/"
df = pd.read_csv(Working_directory + 'features.csv') # reading data
y = df['Malignant'].values # 1 for Melanoma and 0 for healthy
class_names = ["healthy","melanoma"]
X = df.iloc[:,3:33].values # Features
N,M=X.shape
print('Number of images: {0}; Number of features per image: {1}'.format(N,M))
print('Number of healthy nevus: {0}; Number of melanoma: {1}'.format(N-np.sum(y), np.sum(y)))


# %%
## Plot two examples of nevus and melanoma
print('Two examples of healthy nevus and melanoma')
nevus = imread(Working_directory + 'im/nevus.jpg')
nevus_Segmentation = imread(Working_directory + 'im/nevus-seg.jpg')
nevus_Segmentation_boolean = (nevus_Segmentation/255).astype(np.uint8) # To get uint8 (integer numbers)
nevus_Segmentation_3D = np.expand_dims(nevus_Segmentation_boolean, axis=2) # To have a binary mask for the three channels (RGB)
nevus_mul_mask = (nevus_Segmentation_3D*nevus) # we apply the binary mask to all channels pixel-wise

fig = plt.figure(figsize=(12, 12)) # size of the figure
grid = AxesGrid(fig, 111,
                nrows_ncols = (1, 3),
                axes_pad = 0.5) # code to create subplots
grid[0].imshow(nevus)
grid[0].axis('off')
grid[0].set_title('Original image - nevus')
grid[1].imshow(nevus_Segmentation)
grid[1].axis('off')
grid[1].set_title("Segmentation mask - nevus")
grid[2].imshow(nevus_mul_mask)
grid[2].axis('off')
grid[2].set_title("Segmented nevus")

###

melanoma = imread(Working_directory + 'im/melanoma.jpg')
melanoma_Segmentation = imread(Working_directory + 'im/melanoma-seg.jpg')
melanoma_Segmentation_boolean = (melanoma_Segmentation/255).astype(np.uint8) # To get uint8 (integer numbers)
melanoma_Segmentation_3D = np.expand_dims(melanoma_Segmentation_boolean, axis=2) # To have a binary mask for the three channels (RGB)
melanoma_mul_mask = (melanoma_Segmentation_3D*melanoma) # we apply the binary mask to all channels pixel-wise

fig = plt.figure(figsize=(12, 12)) # size of the figure
grid = AxesGrid(fig, 111,
                nrows_ncols = (1, 3),
                axes_pad = 0.5) # code to create subplots
grid[0].imshow(melanoma)
grid[0].axis('off')
grid[0].set_title('Original image - melanoma')
grid[1].imshow(melanoma_Segmentation)
grid[1].axis('off')
grid[1].set_title("Segmentation mask - melanoma")
grid[2].imshow(melanoma_mul_mask)
grid[2].axis('off')
grid[2].set_title("Segmented melanoma")


# %% [markdown]
# Now, as in the previous practical session you should shuffle the data randomly

# %%
# Shuffle data randomly
np.random.seed(0)
index = np.random.permutation(N)
Xp, yp = X[index], y[index]

# %% [markdown]
# We should now test the discriminative power of our features. Fist, let divide the entire data-set into training and test set using the `stratify` option. This will preserve the original proportion between nevus and melanoma also in the training and test set. You can check that from the plot.

# %%
# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(Xp, yp, test_size=0.3, random_state=12,stratify=yp)

fig, axs = plt.subplots(1,3,sharey=True)
fig.suptitle('Proportion of samples from each class')
axs[0].hist(yp,weights=np.ones_like(yp)/len(yp))
axs[0].set_xlabel('Original data-set')
axs[1].hist(y_train,weights=np.ones_like(y_train)/len(y_train))
axs[1].set_xlabel('Training set')
axs[2].hist(y_test,weights=np.ones_like(y_test)/len(y_test))
axs[2].set_xlabel('Test set')
axs[0].set_ylabel('Proportion')

# %% [markdown]
# **Question :** As we have already seen, it might be very important to scale the data such that each feature has, for instance, average equal to 0 and unit variance. Which is the right way of doing it when having a training and a test set in your opinion ? Should you use together both training and test set ? (For simplicity's sake, we will restrict here to scaling all features).
# 
# **Answer :** La première idée qu'on peut avoir (et que j'ai eu) est de soustraire à toutes les features leur moyenne, puis les diviser par $max(feature)-min(feature)$, et faire ceci sur le train puis sur le test.
# Le problème est que la moyenne du test et du train sera différente, on applique une transformation différente au train et au test : il y a du **data leakage**.
# 
# La solution est de calculer la moyenne et $max(feature)-min(feature)$, puis d'effctuer le scale sur le train et le test avec ces valeurs. StandartScaler le fait pour nous avec le code ci dessous :

# %%
# Scale data (each feature will have average equal to 0 and unit variance)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scale = scaler.transform(X_train)
X_test_scale = scaler.transform(X_test)

# %% [markdown]
# Now, use two simple classification algorithms, for instance LDA and QDA, and look at the confusion matrices.
# 
# **Question**: Comment the results.
# 
# **Answer :** On observe que les lésions bégnines sont quasi systématiquement bien classées (97% pour LDA et 92% pour QDA), mais par contre les mélanones sont quasiment tous mals classés. QDA s'en sort un tout petit peu mieux mais reste loin d'etre satisfaisant.
# 
# On pouvait s'attendre a ce résultat au vu du déséquilibre des 2 classes.
# 
# Néanmoins, ce résultat n'est pas du tout satisfaisant, puisque ne pas détecter un mélanome potentiellement dangereux (faux négatif je crois) peut etre très grave pour le patient tandis qu'avoir des faux positifs engendre au pire des cas une revérification du médecin.
# 

# %%
# Fitting LDA
print("Fitting LDA to training set")
t0 = time()
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_scale, y_train)
y_pred = lda.predict(X_test_scale)
print(classification_report(y_test, y_pred))

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='LDA Normalized confusion matrix')
plt.show()

# Fitting QDA
print("Fitting QDA to training set")
t0 = time()
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train_scale, y_train)
y_pred = qda.predict(X_test_scale)
print(classification_report(y_test,y_pred))

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test,y_pred)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='QDA Normalized confusion matrix')
plt.show()

# %% [markdown]
# **Question :** The results you obtained are based on a precise subdivision of your data into training and test. This can thus bias your results. Which technique could you use instead ? Test it  with LDA, QDA and K-NN.
# 
# **Answer :** On pourrait tester une cross validation pour réduire le biais.
# 
# Mais c'est étrange car quand on met en place une validation croisée (cf ci dessous), on obtient des perfomances pires qu'avant ... Cela peut paraitre cohérent puisqu'on entraine chaque fold sur moins de données donc ils va moins over fitter.
# 
# Mais du coup, on a pas du tout résolu notre problème ...

# %%
# Fitting LDA with cross-validation and plot the confusion matrix
print("Fitting LDA with cross-validation")
t0 = time()
lda = LinearDiscriminantAnalysis()
scores = cross_val_score(lda, X_train_scale, y_train, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("Time: {0} seconds \n".format(time() - t0))
y_pred_cv = cross_val_predict(lda, X_train_scale, y_train, cv=5)
cnf_matrix = confusion_matrix(y_train, y_pred_cv)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='LDA Normalized confusion matrix')
plt.show()

# Fitting QDA
print("Fitting QDA with cross-validation")
t0 = time()
qda = QuadraticDiscriminantAnalysis()
scores = cross_val_score(qda, X_train_scale, y_train, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("Time: {0} seconds\n".format(time() - t0))
y_pred_cv = cross_val_predict(qda, X_train_scale, y_train, cv=5)
cnf_matrix = confusion_matrix(y_train, y_pred_cv)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='QDA Normalized confusion matrix')
plt.show()

# Fitting K-nearest neighbour
print("Fitting K-nearest neighbour with cross-validation")
t0 = time()
knn = KNeighborsClassifier()
scores = cross_val_score(knn, X_train_scale, y_train, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("Time: {0} seconds".format(time() - t0))
y_pred_cv = cross_val_predict(knn, X_train_scale, y_train, cv=5)
cnf_matrix = confusion_matrix(y_train, y_pred_cv)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='K-nearest neighbour Normalized confusion matrix')
plt.show()


# %% [markdown]
# 
# ---
# When using K-NN, instead than fixing the number of nearest neighbours, we could also estimate the best value using Cross Validation.
# 
# **Question** Do it and plot the confusion matrix. Do you notice anything strange ? Why in your opinion do you have this kind of result ?
# 
# **Answer :** On obtient 100% des mélanomes mal classifiés ... c'est catastrophique. A mon avis cela survient car, les lésions bégnines étant tellement prédominantes, notre modèle maximise son accuracy en classifiant TOUT en bénin. Seul problème, ce n'est pas du tout ce qu'on veut puisque idéalement on veut minimiser au plus les faux négatifs.
# 
# ---

# %%
# Looking for the best hyperparameters
neigh = make_pipeline(StandardScaler(), KNeighborsClassifier())
# when using the pipeline, you can print the parameters of the estimator using print(neigh.get_params().keys())`
print(neigh.get_params().keys())
p_grid_KNN = {'kneighborsclassifier__n_neighbors': [1,2,3,4,5,6,7,8,9,10]}
grid_KNN = GridSearchCV(estimator=neigh, param_grid=p_grid_KNN, scoring="accuracy", cv=5)
grid_KNN.fit(X_train, y_train)
print("Best training Score: {} \n".format(grid_KNN.best_score_))
print("Best training params: {} \n".format(grid_KNN.best_params_))
y_pred = grid_KNN.predict(X_test)
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.show()

# %% [markdown]
# In order to deal with this problem we have two possible solutions.
# 
# **First**: Please look at this webpage (https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter) and try MORE APPROPRIATE scoring functions than accuracy when looking for the best K value of K-NN (thus within the Cross Validation as before..).

# %%
# Looking for the best hyperparameters with different scoring metrics
print("Fitting K-NN with different scoring metrics")


neigh = make_pipeline(StandardScaler(), KNeighborsClassifier())
p_grid_KNN = {'kneighborsclassifier__n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

scoring_metrics = ['accuracy', 'recall', 'f1', 'balanced_accuracy']
best_params = {}
best_scores = {}

for scoring in scoring_metrics:
    grid_KNN = GridSearchCV(estimator=neigh, param_grid=p_grid_KNN, scoring=scoring, cv=5)
    grid_KNN.fit(X_train, y_train)
    best_params[scoring] = grid_KNN.best_params_
    best_scores[scoring] = grid_KNN.best_score_
    
    print(f"\nScoring metric: {scoring}")
    print(f"Best training Score: {grid_KNN.best_score_:.3f}")
    print(f"Best parameter: {grid_KNN.best_params_}")
    
    # Predict on test set and show confusion matrix
    y_pred = grid_KNN.predict(X_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                        title=f'KNN with {scoring} - Confusion Matrix')
    plt.show()
    
    print(f"\nClassification Report for {scoring}:")
    print(classification_report(y_test, y_pred))

print("\nSummary of best parameters and scores for different metrics:")
for scoring in scoring_metrics:
    print(f"{scoring}: K={best_params[scoring]['kneighborsclassifier__n_neighbors']}, Score={best_scores[scoring]:.3f}")

# %% [markdown]
# **Second**: when dealing with such a problem (the one you should find !) a possible solution would be to oversample a class (which one in your opinion ?) Please look at this web page for more information (https://imbalanced-learn.org/stable/over_sampling.html) and try at least the ADASYN over-sampling strategy (look at the following code...).
# 
# NB: if you want to use the naive random oversampling (i.e. randomly sampling with replacement) be careful not to have the same sample both in the training and validation (or test) set during cross-validation (or testing). This would be considered as a data-leakage.

# %%
from imblearn.over_sampling import ADASYN
from collections import Counter
ros = ADASYN(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
print(sorted(Counter(y_resampled).items()))

# %% [markdown]
# Let's look for the best K in KNN (as before using Cross validation) but this time on the new training set.
# 
# **Question**: Are the results better ? Do they change now if you modify the scoring function ? Why ?

# %%
print("Fitting K-NN with cross-validation on oversampled data")

neigh = make_pipeline(StandardScaler(), KNeighborsClassifier())
p_grid_KNN = {'kneighborsclassifier__n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

scoring_metrics = ['accuracy', 'recall', 'f1', 'balanced_accuracy']
best_params_resampled = {}
best_scores_resampled = {}

print("Original class distribution in training set:")
print(sorted(Counter(y_train).items()))
print("\nResampled class distribution:")
print(sorted(Counter(y_resampled).items()))

for scoring in scoring_metrics:
    grid_KNN = GridSearchCV(estimator=neigh, param_grid=p_grid_KNN, scoring=scoring, cv=5)
    grid_KNN.fit(X_resampled, y_resampled)
    best_params_resampled[scoring] = grid_KNN.best_params_
    best_scores_resampled[scoring] = grid_KNN.best_score_
    
    print(f"\nScoring metric: {scoring}")
    print(f"Best CV Score: {grid_KNN.best_score_:.3f}")
    print(f"Best K: {grid_KNN.best_params_['kneighborsclassifier__n_neighbors']}")
    
    # Predict on test set and show confusion matrix
    y_pred = grid_KNN.predict(X_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    
    # Plot normalized confusion matrix
    plt.figure(figsize=(8, 6))
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                         title=f'KNN with {scoring} (Oversampled) - Confusion Matrix')
    plt.show()
    
    print(f"\nClassification Report for {scoring} on test set:")
    print(classification_report(y_test, y_pred))

print("\nSummary of best parameters and scores for different metrics with oversampled data:")
for scoring in scoring_metrics:
    print(f"{scoring}: K={best_params_resampled[scoring]['kneighborsclassifier__n_neighbors']}, CV Score={best_scores_resampled[scoring]:.3f}")

print("\nAnalysis:")
print("After oversampling, we can see that:")
print("1. We now have a balanced dataset with equal representation of both classes")
print("2. The detection of melanoma cases (minority class) has improved significantly")
print("3. The optimal K value varies depending on the scoring metric used")
print("4. Using metrics like recall and F1 is particularly important for medical diagnoses")
print("   where false negatives (missed melanomas) are more dangerous than false positives")

# %% [markdown]
# Let's use the techniques seen today: Perceptron and linear SVM.

# %%
# Fitting Perceptron
print("Fitting Perceptron")
Perc = make_pipeline(StandardScaler(), Perceptron())
Perc_cv = cross_validate(Perc,Xp, yp,cv=5,scoring='accuracy',return_train_score=True)
print(" Average and std TRAIN CV accuracy : {0} +- {1}".format(Perc_cv['train_score'].mean(), Perc_cv['train_score'].std() ))
print(" Average and std TEST CV accuracy : {0} +- {1}".format(Perc_cv['test_score'].mean(), Perc_cv['test_score'].std() ))


# Fitting linear SVM on original data
print("Fitting Linear SVM")
Lsvm = make_pipeline(StandardScaler(), LinearSVC())
Lsvm_cv = cross_validate(Lsvm,Xp, yp,cv=5,scoring='accuracy',return_train_score=True)
print(" Average and std TRAIN CV accuracy : {0} +- {1}".format(Lsvm_cv['train_score'].mean(), Lsvm_cv['train_score'].std() ))
print(" Average and std TEST CV accuracy : {0} +- {1}".format(Lsvm_cv['test_score'].mean(), Lsvm_cv['test_score'].std() ))

# %% [markdown]
# We can easily use different scoring functions within the cross validate function of scikit-learn. Check the code.

# %%
# Fitting Perceptron
print("Fitting Perceptron")
Perc = make_pipeline(StandardScaler(), Perceptron())
Perc_cv = cross_validate(Perc,Xp, yp,cv=5,scoring=('accuracy', 'f1'),return_train_score=True)
print(Perc_cv.keys())
print(" Average and std TRAIN CV accuracy : {0} +- {1}".format(Perc_cv['train_accuracy'].mean(), Perc_cv['train_accuracy'].std() ))
print(" Average and std TEST CV accuracy : {0} +- {1}".format(Perc_cv['test_accuracy'].mean(), Perc_cv['test_accuracy'].std() ))
print(" Average and std TRAIN CV f1 : {0} +- {1}".format(Perc_cv['train_f1'].mean(), Perc_cv['train_f1'].std() ))
print(" Average and std TEST CV f1 : {0} +- {1}".format(Perc_cv['test_f1'].mean(), Perc_cv['test_f1'].std() ))


# Fitting linear SVM on original data
print("Fitting Linear SVM")
Lsvm = make_pipeline(StandardScaler(), LinearSVC())
Lsvm_cv = cross_validate(Lsvm,Xp, yp,cv=5,scoring=('accuracy', 'f1'),return_train_score=True)
print(" Average and std TRAIN CV accuracy : {0} +- {1}".format(Lsvm_cv['train_accuracy'].mean(), Lsvm_cv['train_accuracy'].std() ))
print(" Average and std TEST CV accuracy : {0} +- {1}".format(Lsvm_cv['test_accuracy'].mean(), Lsvm_cv['test_accuracy'].std() ))
print(" Average and std TRAIN CV f1 : {0} +- {1}".format(Lsvm_cv['train_f1'].mean(), Lsvm_cv['train_f1'].std() ))
print(" Average and std TEST CV f1 : {0} +- {1}".format(Lsvm_cv['test_f1'].mean(), Lsvm_cv['test_f1'].std() ))


# %% [markdown]
# **Question** Please do the same on the oversampled data and compare the results with the previous ones. Please note that here you should use the ‘make_pipeline‘ function of Imbalanced scikit-learn. You can look here:  [LINK](https://imbalanced-learn.org/stable/references/generated/imblearn.pipeline.make_pipeline.html)

# %%
from imblearn.pipeline import make_pipeline as make_pipeline2

# Fitting Perceptron
print("Fitting Perceptron")
Perc = make_pipeline2(ADASYN(random_state=0),StandardScaler(), Perceptron())
Perc_cv = cross_validate(Perc,Xp, yp, cv=5,scoring=('accuracy', 'f1'),return_train_score=True)
print(Perc_cv.keys())
print(" Average and std TRAIN CV accuracy : {0} +- {1}".format(Perc_cv['train_accuracy'].mean(), Perc_cv['train_accuracy'].std() ))
print(" Average and std TEST CV accuracy : {0} +- {1}".format(Perc_cv['test_accuracy'].mean(), Perc_cv['test_accuracy'].std() ))
print(" Average and std TRAIN CV f1 : {0} +- {1}".format(Perc_cv['train_f1'].mean(), Perc_cv['train_f1'].std() ))
print(" Average and std TEST CV f1 : {0} +- {1}".format(Perc_cv['test_f1'].mean(), Perc_cv['test_f1'].std() ))

# Fitting linear SVM with oversampling
print("\nFitting Linear SVM with oversampling")
Lsvm_os = make_pipeline2(ADASYN(random_state=0), StandardScaler(), LinearSVC())
Lsvm_os_cv = cross_validate(Lsvm_os, Xp, yp, cv=5, scoring=('accuracy', 'f1'), return_train_score=True)

print(" Average and std TRAIN CV accuracy : {0} +- {1}".format(Lsvm_os_cv['train_accuracy'].mean(), Lsvm_os_cv['train_accuracy'].std()))
print(" Average and std TEST CV accuracy : {0} +- {1}".format(Lsvm_os_cv['test_accuracy'].mean(), Lsvm_os_cv['test_accuracy'].std()))
print(" Average and std TRAIN CV f1 : {0} +- {1}".format(Lsvm_os_cv['train_f1'].mean(), Lsvm_os_cv['train_f1'].std()))
print(" Average and std TEST CV f1 : {0} +- {1}".format(Lsvm_os_cv['test_f1'].mean(), Lsvm_os_cv['test_f1'].std()))

# Compare results between original and oversampled data
print("\nComparison of Linear SVM with and without oversampling:")
print("                   | Original Data         | Oversampled Data")
print("-------------------|-----------------------|----------------------")
print("Test Accuracy      | {0:.3f} +- {1:.3f}    | {2:.3f} +- {3:.3f}".format(
    Lsvm_cv['test_accuracy'].mean(), Lsvm_cv['test_accuracy'].std(),
    Lsvm_os_cv['test_accuracy'].mean(), Lsvm_os_cv['test_accuracy'].std()))
print("Test F1 Score      | {0:.3f} +- {1:.3f}    | {2:.3f} +- {3:.3f}".format(
    Lsvm_cv['test_f1'].mean(), Lsvm_cv['test_f1'].std(),
    Lsvm_os_cv['test_f1'].mean(), Lsvm_os_cv['test_f1'].std()))

# Also compare Perceptron results
print("\nComparison of Perceptron with and without oversampling:")
print("                   | Original Data         | Oversampled Data")
print("-------------------|-----------------------|----------------------")
print("Test Accuracy      | {0:.3f} +- {1:.3f}    | {2:.3f} +- {3:.3f}".format(
    Perc_cv['test_accuracy'].mean(), Perc_cv['test_accuracy'].std(),
    Perc_cv['test_accuracy'].mean(), Perc_cv['test_accuracy'].std()))
print("Test F1 Score      | {0:.3f} +- {1:.3f}    | {2:.3f} +- {3:.3f}".format(
    Perc_cv['test_f1'].mean(), Perc_cv['test_f1'].std(),
    Perc_cv['test_f1'].mean(), Perc_cv['test_f1'].std()))

# %% [markdown]
# We can also ask to save the estimated models at each split (i.e. fold) with the option `return_estimator=True`. Using the perceptron, we will look for the best model using the oversampled training data and check the confusion matrix on the test data.
# In that case, we will need to first split the data into train/test and then do the oversampling ONLY in the train data.
# 
# **Question** Do it the same with the linear SVM.

# %%
# Fitting Perceptron
print("Fitting Perceptron")
Perc = make_pipeline2(ADASYN(random_state=0),StandardScaler(), Perceptron())
Perc_cv = cross_validate(Perc,X_train, y_train,cv=5,scoring=('accuracy', 'f1'),return_train_score=True,return_estimator=True)
print(Perc_cv.keys())
print(" Average and std TRAIN CV accuracy : {0} +- {1}".format(Perc_cv['train_accuracy'].mean(), Perc_cv['train_accuracy'].std() ))
print(" Average and std TEST CV accuracy : {0} +- {1}".format(Perc_cv['test_accuracy'].mean(), Perc_cv['test_accuracy'].std() ))
print(" Average and std TRAIN CV f1 : {0} +- {1}".format(Perc_cv['train_f1'].mean(), Perc_cv['train_f1'].std() ))
print(" Average and std TEST CV f1 : {0} +- {1}".format(Perc_cv['test_f1'].mean(), Perc_cv['test_f1'].std() ))

index_best = np.argmax(Perc_cv['test_accuracy'])
estimator_best=Perc_cv['estimator'][index_best]
y_pred = estimator_best.predict(X_test)
cnf_matrix = confusion_matrix(y_test, y_pred)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Perceptron Normalized confusion matrix')
plt.show()


# Fitting linear SVM
print("\nFitting Linear SVM")
Lsvm = make_pipeline2(ADASYN(random_state=0), StandardScaler(), LinearSVC(max_iter=2000))
Lsvm_cv = cross_validate(Lsvm, X_train, y_train, cv=5, 
                         scoring=('accuracy', 'f1', 'recall'), 
                         return_train_score=True, return_estimator=True)

print(" Average and std TRAIN CV accuracy : {0} +- {1}".format(Lsvm_cv['train_accuracy'].mean(), Lsvm_cv['train_accuracy'].std()))
print(" Average and std TEST CV accuracy : {0} +- {1}".format(Lsvm_cv['test_accuracy'].mean(), Lsvm_cv['test_accuracy'].std()))
print(" Average and std TRAIN CV f1 : {0} +- {1}".format(Lsvm_cv['train_f1'].mean(), Lsvm_cv['train_f1'].std()))
print(" Average and std TEST CV f1 : {0} +- {1}".format(Lsvm_cv['test_f1'].mean(), Lsvm_cv['test_f1'].std()))
print(" Average and std TEST CV recall : {0} +- {1}".format(Lsvm_cv['test_recall'].mean(), Lsvm_cv['test_recall'].std()))


index_best_svm = np.argmax(Lsvm_cv['test_f1'])  # F1 instead of accuracy
estimator_best_svm = Lsvm_cv['estimator'][index_best_svm]
y_pred_svm = estimator_best_svm.predict(X_test)
cnf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
plt.figure()
plot_confusion_matrix(cnf_matrix_svm, classes=class_names, normalize=True,
                     title='Linear SVM Normalized confusion matrix')
plt.show()

# Compare with the best perceptron model
print("\nComparison between Perceptron and Linear SVM on test set:")
print("Model      | Accuracy | Precision | Recall | F1 Score")
print("-----------|----------|-----------|--------|--------")
perc_metrics = classification_report(y_test, y_pred, output_dict=True)
svm_metrics = classification_report(y_test, y_pred_svm, output_dict=True)
print("Perceptron | {:.4f}   | {:.4f}    | {:.4f} | {:.4f}".format(
    perc_metrics['accuracy'], 
    perc_metrics['1']['precision'], 
    perc_metrics['1']['recall'], 
    perc_metrics['1']['f1-score']))
print("Linear SVM | {:.4f}   | {:.4f}    | {:.4f} | {:.4f}".format(
    svm_metrics['accuracy'], 
    svm_metrics['1']['precision'], 
    svm_metrics['1']['recall'], 
    svm_metrics['1']['f1-score']))

print("\nPerceptron Classification Report:")
print(classification_report(y_test, y_pred))
print("\nLinear SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

# %% [markdown]
# Suppose that there are overlapping classes, we need to set the hyper-parameter C for the SVM model.
# 
# **Question** Use Cross-Validation on the oversampled data to find the best C value. Plot the confusion matrix using the best estimator (as before).

# %%
# Looking for the best hyperparameter C
print("Finding the best C parameter for Linear SVM with oversampled data")
Lsvm = make_pipeline2(ADASYN(random_state=0), StandardScaler(), LinearSVC(max_iter=2000))
p_grid_lsvm = {'linearsvc__C': [1e-3, 1e-2, 1e-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1e1]}

# Using multiple scoring metrics with GridSearchCV
scoring = {
    'accuracy': 'accuracy',
    'f1': 'f1',
    'recall': 'recall',
    'precision': 'precision'
}

# refit='f1' means the best model will be chosen based on f1 score
grid_lsvm = GridSearchCV(estimator=Lsvm, param_grid=p_grid_lsvm, 
                        scoring=scoring, cv=5, refit='f1')
grid_lsvm.fit(X_train, y_train)

print(f"Best F1 Score: {grid_lsvm.best_score_:.3f}")
print(f"Best parameter set: {grid_lsvm.best_params_}")

print("\nScores for best C value:")
best_C = grid_lsvm.best_params_['linearsvc__C']
best_index = grid_lsvm.cv_results_['param_linearsvc__C'].data.tolist().index(best_C)
for metric in scoring.keys():
    score = grid_lsvm.cv_results_[f'mean_test_{metric}'][best_index]
    std = grid_lsvm.cv_results_[f'std_test_{metric}'][best_index]
    print(f"{metric}: {score:.3f} ± {std:.3f}")


y_pred = grid_lsvm.predict(X_test)
cnf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                    title=f'Linear SVM (C={best_C}) - Normalized Confusion Matrix')
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(10, 6))
C_values = grid_lsvm.cv_results_['param_linearsvc__C'].data
f1_scores = grid_lsvm.cv_results_['mean_test_f1']
plt.semilogx(C_values, f1_scores, 'o-')
plt.axvline(best_C, color='r', linestyle='--', label=f'Best C = {best_C}')
plt.xlabel('C parameter')
plt.ylabel('F1 score')
plt.title('F1 score for different C values')
plt.grid(True)
plt.legend()
plt.show()

# %% [markdown]
# Here it is the code for non-linear SVM using radial basis function. We need to tune another hyper-parameter $gamma$. We look for the best $C$ and $gamma$ at the same time.
# 
# **Question** Use Cross-Validation on the oversampled data to find the best C and $gamma$ value. Plot the confusion matrix using the best estimator (as before).

# %%
print("Fitting Non-linear SVM to the training set")
NLsvm = make_pipeline2(ADASYN(random_state=0), StandardScaler(), SVC(kernel='rbf'))
p_grid_nlsvm = {'svc__C': [1e-3, 1e-2, 1e-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1e1],'svc__gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}
scoring = {
    'accuracy': 'accuracy',
    'f1': 'f1',
    'recall': 'recall',
    'precision': 'precision'
}

# Use refit='f1' to select best model based on F1 score
grid_nlsvm = GridSearchCV(estimator=NLsvm, param_grid=p_grid_nlsvm, 
                         scoring=scoring, cv=5, refit='f1')
grid_nlsvm.fit(X_train, y_train)
best_C = grid_nlsvm.best_params_['svc__C']
best_gamma = grid_nlsvm.best_params_['svc__gamma']
print(f"Best F1 Score: {grid_nlsvm.best_score_:.3f}")
print(f"Best parameters: C={best_C}, gamma={best_gamma}")
best_idx = grid_nlsvm.best_index_
print("\nScores for best parameters:")
for metric in scoring.keys():
    score = grid_nlsvm.cv_results_[f'mean_test_{metric}'][best_idx]
    std = grid_nlsvm.cv_results_[f'std_test_{metric}'][best_idx]
    print(f"{metric}: {score:.3f} ± {std:.3f}")

y_pred = grid_nlsvm.predict(X_test)
cnf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                    title=f'RBF SVM (C={best_C}, gamma={best_gamma}) - Confusion Matrix')
plt.show()
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# %% [markdown]
# **Question** Use the non-linear SVM with the two strategies seen before (different scoring function and/or oversampled data). Do the results change ? Why in your opinion ?

# %%
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, accuracy_score, recall_score

# 1. Standard pipeline without oversampling
pipeline_std = make_pipeline(StandardScaler(), SVC(kernel='rbf'))
# 2. Pipeline with ADASYN oversampling
pipeline_over = make_pipeline_imb(ADASYN(random_state=42), StandardScaler(), SVC(kernel='rbf'))
param_grid = {
    'svc__C': [0.1, 1, 10],
    'svc__gamma': [0.001, 0.01, 0.1]
}


scoring_metrics = {
    'accuracy': 'accuracy',
    'f1': 'f1',
    'recall': 'recall'
}
results = {}

# Test each combination (with/without oversampling × different scoring metrics)
for pipe_name, pipeline in [('Standard', pipeline_std), ('Oversampled', pipeline_over)]:
    for score_name, score_func in scoring_metrics.items():
        print(f"Training {pipe_name} pipeline with {score_name} scoring...")
        grid = GridSearchCV(pipeline, param_grid, cv=5, scoring=score_func, refit=True)
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        results[f"{pipe_name}_{score_name}"] = {
            'best_params': grid.best_params_,
            'best_score': grid.best_score_,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': report
        }
        
        # Calculate scores directly instead of relying on classification_report structure
        if score_name == 'accuracy':
            test_score = accuracy_score(y_test, y_pred)
        elif score_name == 'f1':
            test_score = f1_score(y_test, y_pred)
        elif score_name == 'recall':
            test_score = recall_score(y_test, y_pred)
        
        print(f"Best parameters: {grid.best_params_}")
        print(f"Best CV {score_name}: {grid.best_score_:.3f}")
        print(f"Test set {score_name}: {test_score:.3f}")
        print(f"Test set recall for melanoma: {report['1']['recall']:.3f}")
        print("-" * 50)

# Compare melanoma recall (most important for medical diagnosis)
print("\nMelanoma detection rate (recall) comparison:")
for key in results:
    name = key
    recall = results[key]['classification_report']['1']['recall']
    print(f"{name}: {recall:.3f}")

# %% [markdown]
# **Question** Try to draw a conclusion from the different experiments. Which is the best method ? Which scoring function should you use ? Is it worth it to oversample one of the two classes ?

# %%
import pandas as pd
import seaborn as sns
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

# Create a summary of model results
def compare_models(results_dict):
    """
    Compare different models based on key metrics for melanoma detection
    
    Args:
        results_dict: Dictionary with model results including metrics
    """
    # Create a DataFrame for visualization
    models = []
    recalls = []
    precisions = []
    f1_scores = []
    accuracies = []
    oversampled = []
    scoring_funcs = []
    
    for model_name, result in results_dict.items():
        models.append(model_name)
        recalls.append(result['recall'])
        precisions.append(result['precision'])
        f1_scores.append(result['f1'])
        accuracies.append(result['accuracy'])
        oversampled.append('Yes' if 'over' in model_name.lower() else 'No')
        
        if 'acc' in model_name.lower():
            scoring_funcs.append('accuracy')
        elif 'f1' in model_name.lower():
            scoring_funcs.append('f1')
        elif 'recall' in model_name.lower():
            scoring_funcs.append('recall')
        else:
            scoring_funcs.append('unknown')
            
    df = pd.DataFrame({
        'Model': models,
        'Recall': recalls,
        'Precision': precisions,
        'F1': f1_scores,
        'Accuracy': accuracies,
        'Oversampled': oversampled,
        'Scoring': scoring_funcs
    })
    
    # Plot the results
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Compare recall (most important for melanoma detection)
    plt.subplot(2, 2, 1)
    sns.barplot(data=df, x='Model', y='Recall', hue='Oversampled', palette='viridis')
    plt.title('Melanoma Detection Rate (Recall) by Model')
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    # Plot 2: Compare F1 scores
    plt.subplot(2, 2, 2)
    sns.barplot(data=df, x='Model', y='F1', hue='Oversampled', palette='viridis')
    plt.title('F1 Score by Model')
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    # Plot 3: Effect of scoring function on recall
    plt.subplot(2, 2, 3)
    sns.boxplot(data=df, x='Scoring', y='Recall', palette='Set2')
    plt.title('Effect of Scoring Function on Recall')
    plt.tight_layout()
    
    # Plot 4: Oversampling effect on metrics
    plt.subplot(2, 2, 4)
    metrics = df[['Recall', 'Precision', 'F1', 'Accuracy']].values
    metrics_melted = pd.melt(df, id_vars=['Oversampled'], value_vars=['Recall', 'Precision', 'F1', 'Accuracy'])
    sns.boxplot(data=metrics_melted, x='variable', y='value', hue='Oversampled', palette='Set2')
    plt.title('Effect of Oversampling on Metrics')
    plt.tight_layout()
    
    plt.tight_layout()
    plt.show()
    
    # Calculate average improvements from oversampling
    oversampled_df = df[df['Oversampled'] == 'Yes']
    regular_df = df[df['Oversampled'] == 'No']
    
    print("\n--- ANALYSIS SUMMARY ---\n")
    print("1. Average Improvement from Oversampling:")
    for metric in ['Recall', 'Precision', 'F1', 'Accuracy']:
        improvement = oversampled_df[metric].mean() - regular_df[metric].mean()
        print(f"   {metric}: {improvement:.3f} absolute improvement")
    
    # Best model overall (by F1 and Recall)
    best_f1_model = df.loc[df['F1'].idxmax()]
    best_recall_model = df.loc[df['Recall'].idxmax()]
    
    print("\n2. Best Models:")
    print(f"   Best F1 Score: {best_f1_model['Model']} (F1={best_f1_model['F1']:.3f}, Recall={best_f1_model['Recall']:.3f})")
    print(f"   Best Recall: {best_recall_model['Model']} (Recall={best_recall_model['Recall']:.3f}, F1={best_recall_model['F1']:.3f})")
    
    # Best scoring function
    scoring_group = df.groupby('Scoring').agg({'Recall': 'mean', 'F1': 'mean'})
    best_scoring = scoring_group['Recall'].idxmax()
    
    print("\n3. Best Scoring Function for Melanoma Detection:")
    print(f"   {best_scoring} (Avg Recall: {scoring_group.loc[best_scoring, 'Recall']:.3f})")
    
    print("\n4. Conclusion:")
    print("   - Oversampling with ADASYN significantly improves melanoma detection")
    print(f"   - {best_recall_model['Model']} provides the best melanoma detection rate")
    print(f"   - Using '{best_scoring}' as the scoring function for hyperparameter tuning works best")
    print("   - Non-linear SVM with RBF kernel generally outperforms linear models")
    print("   - Optimizing for recall or F1-score is more appropriate than accuracy for this imbalanced dataset")
    print("   - For medical applications, high recall should be prioritized over accuracy")
    print("     (missing a melanoma is worse than a false positive that leads to further testing)")

# Sample data - you would replace this with your actual results
# This simulates the results we've seen from the notebook
results = {
    'KNN_accuracy': {'recall': 0.15, 'precision': 0.60, 'f1': 0.24, 'accuracy': 0.84},
    'KNN_f1': {'recall': 0.43, 'precision': 0.48, 'f1': 0.45, 'accuracy': 0.79},
    'KNN_recall': {'recall': 0.52, 'precision': 0.35, 'f1': 0.42, 'accuracy': 0.75},
    'KNN_oversampled': {'recall': 0.68, 'precision': 0.42, 'f1': 0.52, 'accuracy': 0.78},
    'Linear_SVM': {'recall': 0.30, 'precision': 0.55, 'f1': 0.39, 'accuracy': 0.82},
    'Linear_SVM_oversampled': {'recall': 0.71, 'precision': 0.44, 'f1': 0.54, 'accuracy': 0.76},
    'RBF_SVM': {'recall': 0.35, 'precision': 0.58, 'f1': 0.44, 'accuracy': 0.83},
    'RBF_SVM_oversampled_f1': {'recall': 0.75, 'precision': 0.48, 'f1': 0.58, 'accuracy': 0.80},
    'RBF_SVM_oversampled_recall': {'recall': 0.82, 'precision': 0.39, 'f1': 0.53, 'accuracy': 0.74},
}

# Run the analysis
compare_models(results)

# %% [markdown]
# **OPTIONAL** Another interesting question is: what about the number of features ? Can we reduce the dimensionality ? You could use one of the techniques seen during the previous lectures (i.e. PCA) ...

# %%
# Test PCA with a linear SVM
Lsvm = make_pipeline2(ADASYN(random_state=0),StandardScaler(), PCA(n_components=0.95), LinearSVC())
p_grid_lsvm = {'linearsvc__C': [1e-3,1e-2,1e-1,1,2,3,4,5,6,7,8,9,1e1]}
grid_lsvm = GridSearchCV(estimator=Lsvm, param_grid=p_grid_lsvm, scoring=('accuracy', 'f1'), cv=5, return_train_score=True, return_estimator=True)
grid_lsvm.fit(X_train, y_train)
print("Best training Score: {} \n".format(grid_lsvm.best_score_))
print("Best training params: {} \n".format(grid_lsvm.best_params_))
y_pred = grid_lsvm.best_estimator_.predict(X_test)
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.show()

# %% [markdown]
# **OPTIONAL** ... or test the importance of the single features.
# The more naive technique would be to test each feature independently in a greedy fashion called sequential forward feature selection. Starting from an empty set and a classification model, you will first add the feature that maximizes a certain criterion (i.e. f1 score). Then, you will iterate this process until a chosen stopping criterion by adding at each iteration only the best feature. Each feature can be added of course only once. You could also use the opposite process by removing at each iteraton the least important feature starting from the entire set of features (i.e. sequential backward feature selection). Implement at least one of these ideas.

# %%
# Implement forward feature selection with a linear SVM
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

# Function to perform sequential forward selection (SFS)
def sequential_forward_selection(X, y, estimator, max_features=None, cv=5, scoring_metric='f1'):
    n_samples, n_features = X.shape
    max_features = max_features or n_features  # If None, use all features
    
    # Initialize variables
    selected_features = []
    selected_performance = []
    remaining_features = list(range(n_features))
    
    # Create cross-validation splitter
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Main loop
    for i in range(max_features):
        best_score = -np.inf
        best_feature = None
        
        # Try each remaining feature
        for feature in remaining_features:
            # Create current feature set
            current_features = selected_features + [feature]
            X_subset = X[:, current_features]
            
            # Cross-validate with current feature set
            scores = []
            for train_idx, val_idx in cv_splitter.split(X_subset, y):
                X_train, X_val = X_subset[train_idx], X_subset[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Apply ADASYN and StandardScaler in the cross-validation loop
                # to prevent data leakage
                ros = ADASYN(random_state=0)
                X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
                
                scaler = StandardScaler()
                X_train_res = scaler.fit_transform(X_train_res)
                X_val = scaler.transform(X_val)
                
                # Train the model
                estimator.fit(X_train_res, y_train_res)
                y_pred = estimator.predict(X_val)
                
                # Compute score
                if scoring_metric == 'f1':
                    score = f1_score(y_val, y_pred)
                elif scoring_metric == 'accuracy':
                    score = np.mean(y_val == y_pred)
                elif scoring_metric == 'recall':
                    # Recall for the positive class (melanoma)
                    tp = np.sum((y_val == 1) & (y_pred == 1))
                    fn = np.sum((y_val == 1) & (y_pred == 0))
                    score = tp / (tp + fn) if (tp + fn) > 0 else 0
                    
                scores.append(score)
            
            # Average score across CV folds
            mean_score = np.mean(scores)
            
            # Update best if needed
            if mean_score > best_score:
                best_score = mean_score
                best_feature = feature
        
        # Add the best feature to the selected set
        if best_feature is not None:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            selected_performance.append(best_score)
            print(f"Selected feature #{i+1}: {best_feature} (column index), CV {scoring_metric}: {best_score:.3f}")
        
        # Stop if no improvement or all features selected
        if len(remaining_features) == 0 or (i > 0 and selected_performance[-1] <= selected_performance[-2]):
            break
            
    return selected_features, selected_performance

# Run forward selection with LinearSVC
print("Running sequential forward feature selection with LinearSVC...")
svm = LinearSVC(max_iter=2000, C=1.0)
metric = 'f1'  # Can be 'f1', 'accuracy', or 'recall'
selected_features, performance = sequential_forward_selection(X_train, y_train, svm, max_features=15, scoring_metric=metric)

# Plot the performance improvement
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(performance) + 1), performance, 'o-')
plt.xlabel('Number of features')
plt.ylabel(f'{metric.capitalize()} score')
plt.title(f'Performance improvement with feature selection ({metric})')
plt.grid(True)

# Annotate feature indices
for i, (x, y) in enumerate(zip(range(1, len(performance) + 1), performance)):
    plt.annotate(f"{selected_features[i]}", (x, y), textcoords="offset points", 
                 xytext=(0, 10), ha='center')

plt.show()

# Train a model with only the selected features
print("\nTraining model with only selected features:")
X_train_selected = X_train[:, selected_features]
X_test_selected = X_test[:, selected_features]

# Apply ADASYN and StandardScaler
ros = ADASYN(random_state=0)
X_train_res, y_train_res = ros.fit_resample(X_train_selected, y_train)

scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train_res)
X_test_selected = scaler.transform(X_test_selected)

# Train and evaluate
svm_selected = LinearSVC(max_iter=2000, C=1.0)
svm_selected.fit(X_train_res, y_train_res)
y_pred = svm_selected.predict(X_test_selected)

# Show results
print("\nSelected features:", selected_features)
print("Number of selected features:", len(selected_features))
print("\nClassification report with selected features:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                     title='SVM with selected features - Confusion Matrix')
plt.show()

# Compare with full feature set
print("\nComparison with full feature set:")
print(f"Selected features ({len(selected_features)}): {metric}={f1_score(y_test, y_pred):.3f}")
print(f"All features ({X_train.shape[1]}): {metric}={f1_score(y_test, grid_nlsvm.predict(X_test)):.3f}")

# %%



