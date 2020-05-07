# Cuisine Prediction

### Written by Nicholas Cejda for Text Analytics Spring 2020 - Final Project

This program is designed to accept a user's list of ingredients, say "salsa", "tortilla", and "beef", and predict the style of cuisine those ingredients most belong to, as well as recommending recipes which utilize as many of the listed ingredients as possible. The user is able to select how many recipes they would like to display.

This is achieved by first generating 'Word Vectors' for each word in the Training dataset's (80% of the full dataset) recipe lists, using Spacy's "en_core_web_lg" model, which contains word vectors for many food words. I then took an average of all the word vectors within a recipe to generate 300 numerical features for each recipe. From there, I used the K-Nearest Neighbor's (KNN) approach to train a classifer. I then evaluated the performance of my classifer using the test data (the remaining 20%). It performs with an average accuracy of 71% across all the classes, doing better on the larger classes, and slightly worse on the smaller classes. This represents a 51 point increase in accuracy from a naiive base model (which will simply always select the largest class). Finally, I used this classifer to predict the cuisine style of the user's ingredients and recommend new recipes.

We are using data provided by Yummly.com, located at:
https://www.dropbox.com/s/f0tduqyvgfuin3l/yummly.json


### To run the project
- Clone the repository into a directory of your choice.
- Download the data, using the link above. Place the data into the **cs5293sp20-project3/docs** directory.
- Activate a virtual environment, you can use virtualenv or pipenv.
- If using pipenv, run:
```
    pipenv run pip install requirements.txt
```    
- If using virtualenv, run:
```
    pip install requirements.txt
``` 
- Open a jupyper notebook with the command:
```
    jupyter notebook
```
- When your jupyter server opens and you have the data downloaded into the docs/ folder and you have all dependencies installed, you can now use the notebook and run all code blocks.
   

### How did I turn my text into features and why?

I utilized the Spacy model "en_core_web_lg", which contains 300-dimensional numerical vectors for most English words. I tokenized each recipe in the dataset into individual words with the Spacy's nlp tokenizer, and extracted each 300d word vector. Because each recipe has a variable number of ingredients (words), each recipe has a variable number of word vectors. This is a problem, because I needed a consistent number of features for each recipe for use in downstream Machine Learning tasks. To achieve this, I took the Mean for each element across all the word vectors in that recipe. The result is a single 300-dimensional numerical vector for each recipe. I then added each element of this new vector to a column of a dataframe, as well as the Cuisine style (which represents the labeled Class). The result is a dataframe that is ready to be used for various Machine Learning algorithms for training / classifying.


### What classifiers / clustering methods did you choose and why?

I chose to implement a K Nearest Neighbors (KNN) model for training and prediction. This was primarily due to the simplicity of the method, as well as ease of implementation (Sklearn has a KNN method ready to use almost directly off the shelf). In essence, what KNN does is measure the distance from your point to all the other points, and selects the K nearest neighbors by distance. If the majority of these neighbors belong to a given class, then the prediction will be for that class as well. For this project, I utilized K=8 nearest neighbors, and used the default Minkowski distance as my measurement to the nearest neighbors. This resulted in an average accuracy across all the classes of 71%. The smaller classes had lower scores, while the larger classes had higher scores.

With more time, I would have liked to attempt more advanced ML methods, to hopefully improve my prediction accuracy above 71%. Ideally, we could have implemented a basic 1 or 2 layer Neural Network for more accurate training. As it stands however KNN represents a good start.


### What N did you choose and why?

I elected to give the user the option to choose the number of recommended recipes themselves, but set the default display of n=3. Here was my thought process when designing the recommendation system. The user ("the chef") I had in mind knew what ingredients they had on hand, but needed help deciding what cuisine style is most suited to these ingredients and they wanted some assistance generating fresh recipe ideas with as many of these ingredients as possible. So, the program will first predict the cuisine style, using word vectors and KNN as discussed above. Then our task is to find all the recipes with the most in-common ingredients to the supplied list. In most cases, there will be a tie for the highest number of ingredients. Since the purpose of this recommendation system is to help generate new ideas, it will select amongst the tied recipes randomly.

As an example, the user inputs 'cheese', 'beef', 'red wine'. The prediction for cuisine style might be 'Italian', so the program will look for all the Italian recipes with as many ingredients containing 'cheese', 'beef', and 'red wine' as possible. Since these are common ingredients, it could be that 100 Italian recipes all have an overlap score of 3, the max score. In this case, it will choose the top N number of these 100 recipes randomly. If the user doesn't like their options, they can increase N, or run the program again, and it will find new recipes.

By default I set N = 3, as it gives you a decent number of options without being overwhelming. The user has the option to increase this value should they choose to do so.

### Describe the functions and the code

##### The general workflow of the Notebook:

- First I read in the .json data and write it to a Pandas dataframe ('yumdf').
- Then I take a look and do some very surface-level data exploration, checking how many items are in each class and how many classes we have, and plotting the results.
- I then leverage the Spacy 'en_core_web_lg' model to tokenize my recipes and to append the word vectors to yumdf.
- I utilize the speed and power of numpy to quickly calculate an 'average' word vector for each recipe, and each element of this new vector becomes a column of a new dataframe, 'yumML'
- I add the correct Class (cuisine style) as an extra column to yumML.
- Next, I use sklearn package test_train_split to generate a training dataset and a test dataset. I choose 80% of the data to go into training, and 20% to be reserved for testing (I then check how many of each class are in the training dataset, as having very small classes will skew the training step in favor of the bigger classes, it looks OK, our smallest class has 366 recipes)
- I apply the K-Nearest-Neighbors approach to train our data, using n=8 neighbors and leaving other settings as defaults.
- Print out the confusion matrix (which shows for all classes how many test items were placed in each class correctly (located on the diagonal) or incorrectly (located anywhere else). Also print the precision, recall, f1-score, and support for each of the classes. 
- I then convert the precision / recall / f1-score/ support string output to a tab-delimited format, save the results, and write it to a dataframe, 'modeldf'. I do this because I want to report the f1-score later when the algorithm predicts the user's cuisine class so they have an idea about the confidence level of the predicted class.
- Finally, I am ready to actually predict what cuisine style the user's ingredients belong to! Since I am using the Jupyter Notebook, I just named a variable user_input, but if this were a command-line program, I would have utilized ParseArgs. I wrote a method called **predictCuisine()**, which takes a list of strings as the input and outputs a cuisine prediction.
- I test out the prediction method on a few different recipies, looks like it is doing a good job on lists that have distinct ingredients like 'tortilla' (mexican) and 'tabasco' (us_southern). As expected by the f1-scores, it isn't doing as well on the smaller classes, like Russian, although it can still manage to correctly classify a recipe for the 'Sirkini', a classic Russian cottage-cheese filled pastry.
- Next, we find the **recrecipes()** method, which takes in a list of strings as the user input, as well as an optional parameter to adjust the N of how many recommendations to display. This method starts by calling predictCuisine(), and determining the cuisine style. From there, I subset yumdf to be only the set which matches our predicted style. Then, I build a 2D boolean numpy array which indicates true or false if ingredient i in the user_list is found in recipe j in the yumdf subset. I wind up with an I x J array, where I is the number of items in the the user_list, and J is the number of items in the yumdf subset. I can then count the number of TRUE's in each row i, this represent's recipe j's *score* for the number of in-common ingredients. Using this score, we can find the rows with the highest score, and display those to the user. In the case of a tie for highest score, we will select amongst the top-scores randomly. Finally, it prints the output to the screen.
- The last part of the Notebook tests a few different recipes out (including non-sensical recipes), to see how well the recommendation system is working. Seems like it is performing as expected!


### Describe the tests used

For testing the K-Nearest-Neighbors training model, I set aside 20% of my yumdf data as testing data. This is a necessary step to evaluate how well my KNN method is performing, as all of the test data is, importantly, labeled. So, when the KNN predicts a particular class, it can evaluate whether or not is was correct. From there, we can generate the very informative and widely-used test statistics precision, recall, f1-score, and support. Precision is the number of correct positives / true positives + false positives. Recall is the number of correct positive / true positives + false negatives. F1 is the harmonic mean of the precision and the recall. Support is the number of rows that actually match that class in your test data, this gives you a scale to measure the other numbers by. These tests are critical to see how well your ML traning went and how predictive your model really is. A global accuracy of 71% isn't jaw-dropping, but it is a very reasonable improvement over random guessing or the base model of always selecting the majority class. I tested a few different parameters for the KMM model, and it seems to max out around this point. I would like to have tested more ML models, as well as more parameters, as well as changes to the features (maybe try harmonic mean, or geometric mean of the word vectors for example, or perhaps doing some ingredient 'pre-processing' to remove extremely common ingredients like 'salt' and 'water'. Collectively, these adjustments and further testing would have boosted the perfomance of the model.

While testing the two functions, predictCuisine() and recrecipes(), I tried to account for various errors in user input, as well as provide a variety of ingredients to test the methods with to make sure the program is robust. I am sure there are still remaining bugs! See the "Known Bugs" section below. With the recrecipes() method, I tried to account for situations where No recipes are found matching any ingredients, as well as situations where less than the requested N recipes are found (it just prints out all that it can).


### Known Bugs
- If the user supplies an integer instead of a string, it will crash. This is a pretty easy fix with some try and assert statements and would be addressed with additional time. 

- If you provide words, like 'asdquwry' to predictCuisine() that are not in the Spacy vocabulary, it doesn't account for this and goes ahead and predicts the cuisine anyway. This leads to funny conclusions like ingredients 'abcd', 'erggwer', and '123' are 'Italian'. The model must predict amongst the known classes, so it just goes ahead and predicts the majority class. It doesn't have a 'No class' or 'unknown' option right now.

 - If you set N too high (more than the number of recipes that exist for that cuisine in yumdf), then the program will fail. Within the recrecipes() method, I need to check N to see if it is greater than the total number of recipes. 
