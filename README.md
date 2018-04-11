# Perceptron
Contains an object-oriented model for a perceptron made from scratch using mathematical theory.

This model has not been optimized and is intended for educational purposes rather than maximum performance.

## Getting Started

These instructions will get a copy of the project up and running on your local machine.


### Prerequisites

Instructions for installing these software are listed in the next section: Installing. These are the software packages needed to run:

* Python **2.7**

These Python packages are also needed:

* numpy
* matplotlib
* pandas
* scikit-learn


### Installing

If your computer does not already have Python **2.7** installed, download it [here](https://www.python.org/downloads/).

By default, Python should come with pip (a package manager). Use it to install the following dependencies by opening the Terminal/command line and entering the commands as follows, each line as a separate command:

```
pip install numpy
pip install matplotlib
pip install pandas
pip install scikit-learn
```

## Usage

For all models, it is assumed that the model receives well-prepared and
cleaned input data X and targets T. Any feature engineering should be
done prior to creating a model.

### Basic

1) Clone/fork this repository
2) Open and run the 'example.py' file

### Advanced

1) Create a new Python script
2) Save it in the cloned/forked folder
2) Use a format similar to the following:
```     
from perceptron import Perceptron

X = ...   # input data
T = ...   # target data

proportion_train = 0.8

Ntrain = int(N * proportion_train)

Xtrain, Ttrain = X[:Ntrain], T[:Ntrain]
Xtest, Ttest = X[Ntrain:], T[Ntrain:]

model = Perceptron()

model.fit(Xtrain, Ttrain)

train_acc = model.score(Xtrain, Ttrain)
test_acc = model.score(Xtest, Ttest)

print 'Final w:', model.w
print 'Final b:', model.b

print 'Train Accuracy:', train_acc
print 'Test Accuracy:', test_acc

plt.plot(model.costs)
plt.xlabel('Iteration')
plt.ylabel('Cost')
        
plt.show()

```

## Built With

* [Python](https://www.python.org/about/) - A programming language used here to create exploratory data graphs
* [Numpy](http://www.numpy.org/) - Python library for mathematical and matrix operations 
* [Matplotlib](https://matplotlib.org/) - Python library for graphing data
* [Pandas](https://pandas.pydata.org/pandas-docs/stable/) - Python library for data manipulation
* [Scikit-learn](http://scikit-learn.org/stable/) - Python library used for its shuffle function


## Authors

* **Eric Yates** - [Github Profile](https://github.com/eric-yates)

## License

This project is licensed under the MIT License - see the [LICENSE.md](/LICENSE.md) file for details.

## Acknowledgments

* **LazyProgrammer**: For his [courses](https://www.udemy.com/user/lazy-programmer/) on machine learning


