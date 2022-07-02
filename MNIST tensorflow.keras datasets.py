#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[3]:


print(x_train.shape) # images
print(y_train.shape) # labels

print(x_test.shape) # images
print(y_test.shape) # labels


# In[4]:


plt.imshow(x_train[100], cmap='gray')
plt.show()


# In[5]:


y_train[100]


# In[6]:


y_train


# In[7]:


x_train = x_train.reshape(60000, 28*28) # Flattening the image
y_train_ohe = to_categorical(y_train)

x_test = x_test.reshape(10000, 28*28)
y_test_ohe = to_categorical(y_test)


# In[8]:


print(x_train.shape) # images
print(y_train_ohe.shape) # labels

print(x_test.shape) # images
print(y_test_ohe.shape) # labels


# In[9]:


# Model Building


# In[10]:


# Model Definition


# In[12]:


model = Sequential()
model.add(Input(shape=(784,)))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation="softmax"))


# In[13]:


# Model Compilation


# In[15]:


model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics = 'accuracy')


# In[16]:


# Model Training


# In[17]:


model.fit(x=x_train,y=y_train_ohe,epochs=30,batch_size=1000,validation_data=(x_test,y_test_ohe))


# In[18]:


model.summary()


# In[19]:


# Output


# In[20]:


train_loss = model.history.history['loss']
val_loss = model.history.history['val_loss']
train_accuracy = model.history.history['accuracy']
validation_accuracy = model.history.history['val_accuracy']


# In[21]:


plt.plot(train_loss, label='Train')
plt.plot(val_loss,label='Validation')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curves")
plt.legend()
plt.grid()
plt.show()


# In[22]:


plt.plot(train_accuracy, label='Train')
plt.plot(validation_accuracy,label='Validation')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curves")
plt.legend()
plt.grid()
plt.show()


# In[23]:


# predictions


# In[24]:


x_test[:3]


# In[25]:


np.argmax(model.predict(x_test[:101]),axis=1)


# In[26]:


plt.imshow(x_test[99].reshape(28,28),cmap='gray')
plt.show()


# In[ ]:




