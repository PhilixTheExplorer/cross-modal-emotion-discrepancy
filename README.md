# Cross-Modal Emotion Discrepancy Detection

Code for the paper:

**"Cross-Modal Emotion Discrepancy Detection: Detecting When Voice Reveals 
Emotions That Text Conceals"**  
Kaung Hset Hein and Jonathan H. Chan  
IAIT 2026, Bangkok, Thailand  
DOI: 10.1145/3816713.3818220

## Overview
A cross-modal emotion discrepancy detection system that models disagreement 
between text sentiment and voice prosody as a signal for emotional 
incongruence. Uses 26 engineered features from DistilRoBERTa, librosa, and 
wav2vec2, evaluated across 8 classifiers on RAVDESS and MELD.

## Requirements
- Python 3.8+
- torch
- transformers
- librosa
- scikit-learn
- xgboost
- lightgbm
- pandas, numpy, matplotlib, seaborn, scipy, tqdm, soundfile

Install with:
pip install -r requirements.txt

## Datasets
- RAVDESS: https://kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
- MELD: https://huggingface.co/datasets/ajyy/MELD_audio

## Usage
Run the notebook cells in order on Kaggle with RAVDESS dataset attached.

## Results
| Model | F1 | AUC |
|---|---|---|
| Gradient Boosting | 0.954 | 0.966 |
| XGBoost | 0.952 | 0.968 |
| Random Forest | 0.952 | 0.966 |
| Logistic Regression | 0.918 | 0.947 |

## Citation
```
@inproceedings{hein2026crossmodal,
  title={Cross-Modal Emotion Discrepancy Detection: Detecting When Voice 
  Reveals Emotions That Text Conceals},
  author={Hein, Kaung Hset and Chan, Jonathan H.},
  booktitle={Proceedings of IAIT 2026},
  year={2026},
  doi={10.1145/3816713.3818220}
}
```

## License
MIT License. See LICENSE file.
The accompanying paper is licensed under CC BY 4.0.