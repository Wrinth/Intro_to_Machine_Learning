
# coding: utf-8

# # Hello World

# ### 6 lines of code is all it takes to write your first Machine Learning program. This paper will walk you through writing Hello World for Machine Learning.

# Machine Learning is a subfield of Artificial Intelligence(AI). Early AI programs typically excelleed at just one thing. For instance, [Deep Blue](http://www-03.ibm.com/ibm/history/ibm100/us/en/icons/deepblue/) could play chess at a championship level, but that's all it could do. However, we want to write a program that can solve many problems without needing to be rewritten, like [AlphaGo](https://en.wikipedia.org/wiki/AlphaGo).
# 
# Machine Learning is that behind those AI. In another word, It's the study of algorithms that learn from example and experience, instead of relying on hard-coded rules. One simple example is a software that can figure out the difference between an apple and an orange.

# This hello world application will classify a piece of fruit by user input. First, it will take a description of the fruit as input, and predict whether it is an apple or an orange as output based on its weight and texture.

# To star, I am going to import the Decition Tree from the sklearn library.

# In[1]:


from sklearn import tree


# Let's write down our training data in code. We'll use two variable, features and labels. The features variable is an array of arrays that contain the fruit's weight and its texture. However, instead of using string texture, I will use 0 for "bumpy" and 1 for "smmoth". 
# 
# 
# **The more training data you have, the bettera classifier you can create.**

# In[2]:


features = [[140, 0], [130, 0], [150, 1], [170, 1]]
labels = ["apple", "apple", "orange", "orange"]


# Now, use these two variable to train a classifier. The type of classifier we'll use for this application is called Decition Tree. There are many type of classifier, but the input and output are always the same.
# 
# We'll create the classifier using the DecisionTreeClassifier function. The function will return an empty box of rules. To train the classifier, we'll need a learning algorithm.

# In[3]:


clf = tree.DecisionTreeClassifier()


# If a classifier is a box of rules, then the learning algorithm is the procedure that creates them. The algoritms will find the patterns from your training data.
# 
# In sklearn, the learning algorithm is included in the classifier object, and it's called fit(). fit() is a synonym for "Find Patterns in Data."

# In[4]:


clf = clf.fit(features, labels)


# At this point, we have a trained classifier. Let's take it for a spin and use it to classify a new fruit.
# 
# The input to the classifier is the features for a new example. Let's say the fruiy we want to classify is 150 grams and bumpy(0). The output will be either an apple or an orange.

# In[5]:


print(clf.predict([[150, 0]]))


# If everything work for you then congraduration! You made your first Machine Learning program!
# 
# You can create a new classifier for a new problem just by changing the training data.

# **The neat thing is that programming with Machine Learning isn't hard. However, to get it right, you need to understand a few important concepts.**

# ### Resources:
# 
# * [Hello World - Machine Learning Recipes #1](https://www.youtube.com/watch?v=cKxRvEZd3Mw&list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal&index=1)
