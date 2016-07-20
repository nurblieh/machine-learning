## Linear Regression
Using a simple linear regression model (solving for theta with `normalEqn`), we're able to predict better than "guessing". Which is to say, the model does better than guessing the mean() value. One unknown for me is that normalEqn() (aka the "closed form") solution for theta offers significantly better results than gradientDescent. I need to noodle on that more.

The R Squared isn't great (~.43 for the sample below), but given the labels are subjective human grades, at least there's some improvement over random.

Funny enough, eyeballing the data, it looks like alcohol is the best predictor of a high score. Some analysis could be added to confirm that.

## Neural Network / Non-linear
Using a neural network, it's possible to get  better results than with the above linear method. This could mean there are feature-feature interactions which the simple linear model can't represent. 

Initial tests used a 3-layer (1 hidden layer) neural network with the # of units being 12-6-1, respectively. Lambda in the [0.5, 1.0] range seemed to work well. Code and configs in `costFunctionNN.m` and `runNN.m`. Example run at the bottom of this page.

## Multi-Classification
Bucketing the numerical ratings into a few classes (eg, Low; Avg; High) should yield good accuracy and is probably more realistic. The 100 pt scale that most wines are rated on has always seemed ludicracy to me, but I'm no [Supertaster](https://en.wikipedia.org/wiki/Supertaster). No code submitted for this yet.

## Octave example
### Setup
```
>> run setup-data.m
```
### Solve with normal equation (aka closed-form) method.
```
>> theta = normalEqn(XTraining, yTraining);
>> rSquared(XTest, yTest, theta)
ans =  0.43233
```
### Alternatively, solve with iterative batch gradient descent.
```
>> [theta, J_History] = gradientDescent(XTraining, yTraining, theta, .0003, 250);

>> figure; ylabel('Cost J'); xlabel('Number of iterations');
>> plot(1:numel(J_History), J_History, '-b', 'LineWidth', 2);
```
![image](https://cloud.githubusercontent.com/assets/311298/16923027/0efcab1a-4ccd-11e6-86a8-dd2310ff29ee.png)

### Neural Network attempt
```
>> run runNN.m
Running neural network for 100 iterations...
Cost (training set): 0.2177
Cost (test set): 0.1616
R-Squared (training set): 0.3342
R-Squared (test set): 0.4786
```
