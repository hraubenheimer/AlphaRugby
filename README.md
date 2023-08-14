# AlphaRugby
A machine learning model that predicts the winner and score margin of international rugby matches. Achieved an 83% test accuracy in the 2019 World Cup.
Note: data the model is trained on data that may be out of date.

To predict the outcomes of matches, type use the following command:

python alpharugby.py <start date> <end date>

where 'start date' and 'end date' indicate a time interval of matches to be predicted. Both dates need to be specified in the format 'YYYY-MM-DD'. Additionally, the following command will also work:

python alpharugby.py <start date>

which will make predictions of all matches played after 'start date'.

The model is trained on all matches played before the earliest match to be predicted.
