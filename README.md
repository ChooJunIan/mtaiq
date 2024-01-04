# MTAIQ: Modelling Technical and Aesthetic Image Quality

MTAIQ is a project that focuses on creating a multi-task machine learning model that is able to predict both the Aesthetic and Quality rating of a given image. Thhe project uses shared weights
models with the InceptionResNet-v2 as its backbone model.

`Duplicate_image_investigation`:
Contains files investigating duplicated images between 3 image datasets. These datasets are KonIQ-10k, PARA, and SPAQ image datasets. 

`ImageDataset_CSVs`:
Contains CSVs with respect to the datasets used for transfer learning in this project. The datsets are AVA, PARA, SPAQ, and KonIQ-10k.

`ViPr_Lab_files`:
Folders containing the bulk of the work. Within the folders are the 2 following folders:
1. `Ianv2-files`:
- `ava-mlsp`:
Contains files for initial prototyping using multi-layer spatially pooled features.
- `inceptionResNet-multitask`:
Contains files for training models and predicting image Aesthetic and Quality scores.
- `mtaiq`:
Contains CSVs from the `ava-mlsp` prototyping results.
2. `multimodel_csv-files`:
- `multimodel_datasets`:
Contains the datasets used to train, validate, and test the baseline and multimodel architectures.
- `prediction`:
Contains the prediction results for the testing data from the baseline and multimodel architectures.

`demo`:
Contains the demo application to run our best model for multi-task prediction. 

