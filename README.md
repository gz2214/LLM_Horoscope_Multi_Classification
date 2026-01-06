# LLM_Horoscope_Multi_Classification

## Introduction
The purpose of this analysis is to create a end-to-end pipeline that will fine-tune LLM to perform multi-classification. Specifically, we use the [facebook/bart-base](https://huggingface.co/facebook/bart-base) to see if it can learn to identify if horoscope text can be attributed to the appropriate zodiac signs. 

## Data
Data comes from the horoscope data set from [kaggle](https://www.kaggle.com/datasets/shahp7575/horoscopes?resource=download). There is approximately 18K text samples, distributed equally among the 12 zodiac signs.

## Methodology 
We use the [Optuna](https://optuna.org/) package in python to automate hyperperameter tuning. We ran 20 trials to find the best model. 70% of the data was used for training, while validation and testing both used 15% of the data respectively. We used [F1 classification score](https://en.wikipedia.org/wiki/F-score) to guide the optimization of the model. 

## Results

The model's best performance was when accuracy rate was around 10%, and F1 classification score was around 0.09. This is only marginally better than random guessing. Looking at the confusion matrix results when evaluating on the test data, we see that model learned to predict two classes are majority of the time. In this case, Aquarius and Aries were the classes predicted the most, no matter the actual zodiac sign. We can reasonably conclude then that the model was unable to uncover differentiating features from the text samples that can be attributed to a specific Zodiac sign. The model then had to learn a degenerate solution (picking two classes) in order to raise its accuracy and F1 classification score, showcasing model collapse. 

