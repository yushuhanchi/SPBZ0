# Urban Sound Classification

This project builds a low-cost audio classification system using MFCC features and a CNN model.

## Method
- Audio processed into MFCC features using librosa
- CNN model for classification
- Evaluation using accuracy and confusion matrix

## Results
- Test accuracy: ~89.8%
- High performance on clear sounds (gun_shot, siren)
- Lower performance on complex sounds (street_music, children_playing)

## Files
- report_template.md: main report
- train_model.py: model training
- make_figures.py: generate figures
- images: results and diagrams

## Dataset
UrbanSound8K (not included due to size)
