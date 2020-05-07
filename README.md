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
- Next, I use sklearn package test_train_split to generate a training dataset and a test dataset. I chose 80% of the data to go into training, and 20% to be reserved for testing (I check how many of each class are in the training dataset, as having very small classes will skew the training step, it looks OK, our smallest class has 366 recipes)
- I apply the K-Nearest-Neighbors approach to train our data.

... 

Remaining README is a work in progress.



5 pts: Describe functions/code
5 pts: Describe tests
