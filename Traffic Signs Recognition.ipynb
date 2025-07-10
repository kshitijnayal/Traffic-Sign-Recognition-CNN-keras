{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "9ebf0cd0-c830-4052-a044-265b4409e056",
   "metadata": {
    "jp-MarkdownHeadingCollapsed": true
   },
   "source": [
    "# Importing Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "7382bcf2-c03d-4f43-ad96-9ef9ad781fe4",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np \n",
    "import pandas as pd \n",
    "import matplotlib.pyplot as plt\n",
    "import cv2\n",
    "import tensorflow as tf\n",
    "from PIL import Image\n",
    "import os\n",
    "from sklearn.model_selection import train_test_split\n",
    "from keras.utils import to_categorical\n",
    "from keras.models import Sequential, load_model\n",
    "from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout\n",
    "from keras import Input\n",
    "import keras\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dba33454-8281-4cf7-a9a5-3a043675897b",
   "metadata": {
    "jp-MarkdownHeadingCollapsed": true
   },
   "source": [
    "# Pre Processing Images"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "5fd17371-d02c-4467-984f-c3fae83e0135",
   "metadata": {},
   "outputs": [],
   "source": [
    "data = []\n",
    "labels = []\n",
    "classes = 43\n",
    "cur_path = os.getcwd()\n",
    "#Retrieving the images and their labels \n",
    "for i in range(classes):\n",
    "    path = os.path.join(cur_path,'Train',str(i))\n",
    "    images = os.listdir(path)\n",
    "    for a in images:\n",
    "        try:\n",
    "            image = Image.open(path + '\\\\'+ a)\n",
    "            image = image.resize((30,30))\n",
    "            image = np.array(image)\n",
    "            #sim = Image.fromarray(image)\n",
    "            data.append(image)\n",
    "            labels.append(i)\n",
    "        except:\n",
    "            print(\"Error loading image\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6f37cdb8-02fc-4bed-9b10-5de1e1bcba91",
   "metadata": {
    "jp-MarkdownHeadingCollapsed": true
   },
   "source": [
    "# Converting Lists into Numpy arrays"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "0ea8221f-c102-41df-9822-e84ab2fc26e3",
   "metadata": {},
   "outputs": [],
   "source": [
    "data = np.array(data)\n",
    "labels = np.array(labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "9d2b0c61-835d-4c16-bd05-3146503f82ca",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(39209, 30, 30, 3) (39209,)\n"
     ]
    }
   ],
   "source": [
    "print(data.shape, labels.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "82b84182-26d0-4be4-b125-fb2b7eb09a93",
   "metadata": {
    "jp-MarkdownHeadingCollapsed": true
   },
   "source": [
    "# Train-Test Split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "f062f87c-e439-46d0-a614-d4a5c865cc14",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train,x_test,y_train,y_test=train_test_split(data,labels,test_size=0.25,random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "58e8319c-3375-4817-a9e7-16ca2a112292",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(29406, 30, 30, 3)"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "e0fc3661-0468-45c7-a4c9-81acfeb3e238",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(9803, 30, 30, 3)"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "1df17658-e076-464e-81ea-b43cc0eef1a3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(29406,)"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "47c0d62f-bb50-4ea4-8ff9-dc73d1221fc1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(9803,)"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_test.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f63aa661-49e8-4abe-9bcf-1659fcd7948f",
   "metadata": {
    "jp-MarkdownHeadingCollapsed": true
   },
   "source": [
    "# Converting the labels into one hot encoding"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "b57e8da1-90f7-425b-b244-d9f1920f1830",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_train = to_categorical(y_train, 43)\n",
    "y_test = to_categorical(y_test, 43)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f5c805d3-26cf-4b79-a204-f10540e87dbc",
   "metadata": {
    "jp-MarkdownHeadingCollapsed": true
   },
   "source": [
    "# Building The Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "fabb8515-9bf6-47f8-a9b7-c286c4c4a2b7",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = Sequential()\n",
    "model.add(Input(shape=x_train.shape[1:]))  # Proper Input layer\n",
    "model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))\n",
    "model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))\n",
    "model.add(MaxPool2D(pool_size=(2, 2)))\n",
    "model.add(Dropout(rate=0.25))\n",
    "model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))\n",
    "model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))\n",
    "model.add(MaxPool2D(pool_size=(2, 2)))\n",
    "model.add(Dropout(rate=0.25))\n",
    "model.add(Flatten())\n",
    "model.add(Dense(256, activation='relu'))\n",
    "model.add(Dropout(rate=0.5))\n",
    "model.add(Dense(43, activation='softmax'))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f331a823-69de-40ff-9155-4e5b2e18770e",
   "metadata": {
    "jp-MarkdownHeadingCollapsed": true
   },
   "source": [
    "# Compiling The Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "826ea981-d518-4d33-bc41-cf1ec171d621",
   "metadata": {},
   "outputs": [],
   "source": [
    "callback1=keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "56d7e6c5-8644-462a-a3cc-2dec93f15088",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "fe41c959-0301-4572-bde3-65481a1b806e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/20\n",
      "\u001b[1m919/919\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m39s\u001b[0m 43ms/step - accuracy: 0.9461 - loss: 0.2279 - val_accuracy: 0.9849 - val_loss: 0.0586\n",
      "Epoch 2/20\n",
      "\u001b[1m919/919\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m47s\u001b[0m 49ms/step - accuracy: 0.9598 - loss: 0.1720 - val_accuracy: 0.9872 - val_loss: 0.0432\n",
      "Epoch 3/20\n",
      "\u001b[1m919/919\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m78s\u001b[0m 45ms/step - accuracy: 0.9509 - loss: 0.2217 - val_accuracy: 0.9889 - val_loss: 0.0458\n",
      "Epoch 4/20\n",
      "\u001b[1m919/919\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m91s\u001b[0m 54ms/step - accuracy: 0.9536 - loss: 0.2057 - val_accuracy: 0.9763 - val_loss: 0.1052\n",
      "Epoch 5/20\n",
      "\u001b[1m919/919\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m69s\u001b[0m 40ms/step - accuracy: 0.9448 - loss: 0.2731 - val_accuracy: 0.9869 - val_loss: 0.0532\n",
      "Epoch 6/20\n",
      "\u001b[1m919/919\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m45s\u001b[0m 49ms/step - accuracy: 0.9526 - loss: 0.2253 - val_accuracy: 0.9903 - val_loss: 0.0368\n",
      "Epoch 7/20\n",
      "\u001b[1m919/919\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m45s\u001b[0m 48ms/step - accuracy: 0.9526 - loss: 0.2155 - val_accuracy: 0.9811 - val_loss: 0.0912\n",
      "Epoch 8/20\n",
      "\u001b[1m919/919\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m78s\u001b[0m 44ms/step - accuracy: 0.9446 - loss: 0.2807 - val_accuracy: 0.9520 - val_loss: 0.1868\n",
      "Epoch 9/20\n",
      "\u001b[1m919/919\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m15s\u001b[0m 16ms/step - accuracy: 0.9475 - loss: 0.2539 - val_accuracy: 0.9801 - val_loss: 0.0819\n"
     ]
    }
   ],
   "source": [
    "history =model.fit(x=x_train,y=y_train,batch_size=32,epochs=20,verbose='auto',callbacks=[callback1],validation_data=(x_test,y_test),validation_split=0.2)\n",
    "model.save(\"Traffic_sign_recog_model.keras\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b901f522-c865-4738-8613-2ca164b7482e",
   "metadata": {
    "jp-MarkdownHeadingCollapsed": true
   },
   "source": [
    "# Plotting graphs for accuracy "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "2eb29dd5-8a01-4c16-b7db-c163fd200fc3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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ol+/fsbWlxs3bjB27FiqV69OmTJlcHNz4+jRow/VApSYmMiwYcOoXLkynp6eeHp6cv36deNj7N27l8DAQGPx81+5LUBF1aBBg3znZs6cSYMGDfDz88PNzY3vvvvOmCsxMZELFy7c87mHDBnC3Llzjdf//fff+YosSyQtQEIIi9exRjker1+BJbvP8/qi/SwY2oTn5u4g9vINyns68eOQxpT1cNI6pvmxd1FbY7R43ofQvXt3DAYDf//9Nw0bNmTz5s1MmTLFePsbb7zBypUr+d///kdERATOzs707t2brKysB36OQYMGcfnyZb788kuCg4NxdHSkadOmxsdwdna+5/3vd7uNjQ3Kf7r+Chrk7Oqat2Xs119/5dVXX2Xy5Mk0bdoUd3d3Pv/8c7Zv3/5Azwvw7LPPMm7cOLZt28a2bdsICQmhZcuW971faScFkBDCKrzbvQbRJ68Ql3SDdpM3cj0zB183B34a0phAr4d7w7UaOt0Dd0VpydnZmccff5yffvqJkydPUrlyZSIjI423b968mUGDBtGrVy9AHRN0+vTph3qOzZs3M336dLp06QLA2bNnSUpKMt5eu3Ztzp07x/HjxwtsBapduzZr167N0111Jz8/vzzjcFJTU4mLi3ugXM2aNWP48OHGc7GxscZjd3d3QkJCWLt2LW3atCnwMXx8fOjZsydz585l27ZtPPfcc/d9XksgXWBCCKvg6WzPp71rA3A9MwcPJzvmP9+YMD/LHudgLZ566in+/vtvoqKiePrpp/PcFhERwZIlS9i7dy/79u1jwIAB+WaN3U9ERATz58/nyJEjbN++naeeeipP60qrVq145JFHeOKJJ1i9ejVxcXGsWLGCf/75B4Dx48ezY8cOhg8fzv79+zl69CgzZswwFlFt27Zl/vz5bN68mYMHDzJw4EBjF979cu3cuZOVK1dy/PhxJkyYwI4dO/Jc89577zF58mSmTp3KiRMn2L17N19//XWea4YMGcIPP/zAkSNHjLPnLJ0UQEIIq9Gqsh9j2lemir87PwxuRLXyHlpHEibStm1bvL29OXbsGAMGDMhz2xdffIGXlxfNmjWje/fudOzYkfr16z/U40dFRZGcnEy9evV45plneOWVVyhbtmyeaxYvXkzDhg3p378/1atXZ+zYscZZXpUrV2bVqlXs27ePRo0a0bRpU/7880/s7NSO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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(0)\n",
    "plt.plot(history.history['accuracy'], label='training accuracy')\n",
    "plt.plot(history.history['val_accuracy'], label='val accuracy')\n",
    "plt.title('Accuracy')\n",
    "plt.xlabel('epochs')\n",
    "plt.ylabel('accuracy')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "52c1453c-8552-4774-8dda-d34530585939",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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4Nlnn6V///706NEjR+9/3LhxVKtWDT8/P2rVqsU333yT5flXXnmFihUr4u/vT9myZRk6dGjmc2PHjqVGjRoEBAQQERFBr169cvTaOaUAJCIeY8Xek8zacISUNAdDp21gzYHTVpckOWAYBudT0ix5GIaRrRo/+ugjWrduzSOPPEJMTAwxMTFUqFAh8/lnnnmG0aNHs337dho2bMi5c+fo2rUrixYtYsOGDXTu3Jnu3bsTHR19zdd59dVX6d27N5s2baJr16707duX06f//ef5/PnzvPfee3zzzTcsXbqU6Ohonnrqqczn3377bb777jsmT57M33//TXx8PD///HO23nOGWbNmMWzYMP7zn/+wZcsW/u///o+HHnqIv/76C4CffvqJDz/8kPHjx7N7925+/vlnGjRoAMDatWsZOnQoo0aNYufOnfz+++/ceOONOXr9nNI1YBHxCHaHwahftwEQFuhL3IVUBk1Zw0+PtaFmRIjF1Ul2XEi1U/d/8y157W2jOhPkd/2PzLCwMPz8/AgKCrrqbahRo0bRsWPHzD+XKFGCRo0aZf759ddfZ9asWcyePZsnnnjiX19nwIAB3HfffQC8+eabfPLJJ6xevZrbbrvtqsenpqby+eefU61aNQCeeOIJRo0alfn8J598wvPPP0/Pnj0B+PTTT5k7d+513++l3nvvPQYMGMDjjz8OwMiRI1m1ahXvvfce7du3Jzo6mtKlS9OhQwd8fX2pWLEiLVq0ACA6OpoiRYpw++23ExISQqVKlWjSpEmOXj+ndAVIRDzCD2sPsSM2gdAAH34f3o7ISsWIT0qj/6TVxMRdsLo88RDNmjXL8ufExESeeeYZ6tatS9GiRQkODmbHjh3XvQLUsGHDzO0iRYoQEhLC8ePH//X4oKCgzPADUKZMmczj4+LiOHbsWGYYAfD29iYyMjJH72379u20bds2y762bduyfft2AO655x4uXLhA1apVeeSRR5g1axZpaWkAdOzYkUqVKlG1alUefPBBvvvuO86fP5+j188pXQESEbcXn5TKe/N3AjC8Q03KhAXyZf9m9Pp8JXuOn6P/pNX8+H9tCAvKfVt9KXiBvt5sG9XZstfOD5fP5nr66aeZP38+7733HtWrVycwMJBevXqRkpJyzfNcvgSEzWbD4XDk6PjLb+td3mAwu7f9rneOjH0VKlRg586dLFy4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wYIFC7jrrruYPHlyns4nIoVDg6BFxOmEhITw1FNPMWLECBwOBzfccAPx8fGsWLGC4OBgKlWqBMCoUaMoUaIEERERvPjii5QsWZIePXoA8J///IfmzZvz2muv0adPH1auXMmnn37K2LFjAahcuTL9+/dn4MCBmYOgDx48yPHjx+ndu/d1a7xw4QJPP/00vXr1okqVKhw+fJg1a9Zw9913F9h/FxHJR1YPQhIRuRqHw2F89NFHRq1atQxfX18jPDzc6Ny5s7FkyZLMAcq//vqrUa9ePcPPz89o3ry5ERUVleUcP/30k1G3bl3D19fXqFixovHuu+9mef7ChQvGiBEjjDJlyhh+fn5G9erVjUmTJhmGcXEQ9JkzZzKP37BhgwEY+/fvN5KTk417773XqFChguHn52eULVvWeOKJJzIHSIuIc7MZxiU3xUVEXMDixYtp3749Z86coWjRolaXIyIuSGOARERExOMoAImIiIjH0S0wERER8Ti6AiQiIiIeRwFIREREPI4CkIiIiHgcBSARERHxOApAIiIi4nEUgERERMTjKACJiIiIx1EAEhEREY+jACQiIiIe5/8BIlMXCtALIDwAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(1)\n",
    "plt.plot(history.history['loss'], label='training loss')\n",
    "plt.plot(history.history['val_loss'], label='val loss')\n",
    "plt.title('Loss')\n",
    "plt.xlabel('epochs')\n",
    "plt.ylabel('loss')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cc899980-6b2d-4610-bb27-aec3060016fd",
   "metadata": {
    "jp-MarkdownHeadingCollapsed": true
   },
   "source": [
    "# Testing accuracy on test dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "7d77ecdd-f0fe-45f0-be19-2ca7adbf74c7",
   "metadata": {},
   "outputs": [],
   "source": [
    "test_df=pd.read_csv('Test.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "979662de-0598-4d68-a7a9-5013439d5698",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m395/395\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m7s\u001b[0m 17ms/step\n"
     ]
    }
   ],
   "source": [
    "labels = test_df[\"ClassId\"].values\n",
    "imgs = test_df[\"Path\"].values\n",
    "data=[]\n",
    "for img in imgs:\n",
    "    image = Image.open(img)\n",
    "    image = image.resize((30,30))\n",
    "    data.append(np.array(image))\n",
    "x_test=np.array(data)\n",
    "pred = np.argmax(model.predict(x_test), axis=1)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d95cbcb8-434a-45df-993a-8e91c6cfee98",
   "metadata": {
    "jp-MarkdownHeadingCollapsed": true
   },
   "source": [
    "# Accuracy with the test data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "e561683f-5a14-417a-af4c-e0c4b5dbe4ac",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.9456057007125891\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import accuracy_score\n",
    "print(accuracy_score(labels, pred))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "59135a3a-be3a-4db6-96ee-f535c4922b3a",
   "metadata": {
    "jp-MarkdownHeadingCollapsed": true
   },
   "source": [
    "# Saving The Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "311e47b4-84d7-457e-986c-74b990fa0944",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.save(\"traffic_sign_classifier.keras\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
