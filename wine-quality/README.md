## Linear Regression
Using a simple linear regression model (solving for theta with `normalEqn`), we're able to predict better than "guessing". Which is to say, the model does better than guessing the mean() value.

The R Squared isn't great, but given the labels are subjective human grades, at least there's some improvement over random.

Funny enough, eyeballing the data, it looks like alcohol is the best predictor of a high score. Some analysis could be added to confirm that.

## Non-linear 
It's possible a non-linear model could provide some improvement due to interaction between features. Rather than generating a polynomial mapping of the features, I'll cook up a small neural network implementation for the practice and see if that helps.

## Multi-Classification
Bucketing the numerical ratings into a few classes (eg, Low; Avg; High) should yield good accuracy and is probably more realistic. The 100 pt scale that most wines are rated on has always seemed ludicracy to me, but I'm no [Supertaster](https://en.wikipedia.org/wiki/Supertaster). No code submitted for this yet.

Octave example,
```
>> run setup-data.m

>> theta = normalEqn(XTraining, yTraining);

>> rSquared(XTest, yTest, theta)
ans =  0.43233

```
