{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "### Problem Statement:\n",
    " #Classification of cars is a process of evaluating of cars based upon the attributes Car's acceptability,Price,Tech etc. \n",
    "\n",
    "#Data Set Information:\n",
    " #This data set also known as “The Car Evaluation Database” contains examples with the structural information removed\n",
    " #Source of the data.The quality of cars is measured by two main groups of criteria:\n",
    " #price (PRICE) and technical characteristics (TECH). The price is determined by buying and maintenance price.\n",
    " #Technical characteristic are decomposed into safety and comfort, which further depends on number of doors, size of car and size of the luggage boot.\n",
    "\n",
    "### About Project¶\n",
    "  #In this project, I will analyze the Car Evaluation dataset and \n",
    "  #create a classification algorithm which can evaluate the car on the following attributes:\n",
    "  # CAR car acceptability\n",
    "  # . PRICE overall price\n",
    "  # . . buying buying price\n",
    "  # . . maint price of the maintenance\n",
    "  # . TECH technical characteristics\n",
    "  # . . COMFORT comfort\n",
    "  # . . . doors number of doors\n",
    "  # . . . persons capacity in terms of persons to carry\n",
    "  # . . . lug_boot the size of luggage boot\n",
    "  # . . safety estimated safety of the car\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Fr2CZl_dncHX"
   },
   "source": [
    "### Loading the Following Libraries & Data\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "_kg_hide-output": true,
    "execution": {
     "iopub.execute_input": "2021-11-23T12:33:55.466283Z",
     "iopub.status.busy": "2021-11-23T12:33:55.465713Z",
     "iopub.status.idle": "2021-11-23T12:34:06.113146Z",
     "shell.execute_reply": "2021-11-23T12:34:06.112084Z",
     "shell.execute_reply.started": "2021-11-23T12:33:55.466162Z"
    },
    "id": "ockVk97K2r1d",
    "outputId": "36439052-8d25-4aab-c1c6-325855aecbf4"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already up-to-date: category_encoders in c:\\users\\supriya\\anaconda3\\lib\\site-packages (2.5.0)\n",
      "Requirement already satisfied, skipping upgrade: pandas>=1.0.5 in c:\\users\\supriya\\anaconda3\\lib\\site-packages (from category_encoders) (1.3.5)\n",
      "Requirement already satisfied, skipping upgrade: scikit-learn>=0.20.0 in c:\\users\\supriya\\anaconda3\\lib\\site-packages (from category_encoders) (0.22.1)\n",
      "Requirement already satisfied, skipping upgrade: patsy>=0.5.1 in c:\\users\\supriya\\anaconda3\\lib\\site-packages (from category_encoders) (0.5.1)\n",
      "Requirement already satisfied, skipping upgrade: scipy>=1.0.0 in c:\\users\\supriya\\anaconda3\\lib\\site-packages (from category_encoders) (1.4.1)\n",
      "Requirement already satisfied, skipping upgrade: statsmodels>=0.9.0 in c:\\users\\supriya\\anaconda3\\lib\\site-packages (from category_encoders) (0.11.0)\n",
      "Requirement already satisfied, skipping upgrade: numpy>=1.14.0 in c:\\users\\supriya\\anaconda3\\lib\\site-packages (from category_encoders) (1.18.1)\n",
      "Requirement already satisfied, skipping upgrade: pytz>=2017.3 in c:\\users\\supriya\\anaconda3\\lib\\site-packages (from pandas>=1.0.5->category_encoders) (2019.3)\n",
      "Requirement already satisfied, skipping upgrade: python-dateutil>=2.7.3 in c:\\users\\supriya\\anaconda3\\lib\\site-packages (from pandas>=1.0.5->category_encoders) (2.8.1)\n",
      "Requirement already satisfied, skipping upgrade: joblib>=0.11 in c:\\users\\supriya\\anaconda3\\lib\\site-packages (from scikit-learn>=0.20.0->category_encoders) (0.14.1)\n",
      "Requirement already satisfied, skipping upgrade: six in c:\\users\\supriya\\anaconda3\\lib\\site-packages (from patsy>=0.5.1->category_encoders) (1.14.0)\n"
     ]
    }
   ],
   "source": [
    "!pip install --upgrade category_encoders"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-11-23T12:34:06.116025Z",
     "iopub.status.busy": "2021-11-23T12:34:06.115619Z",
     "iopub.status.idle": "2021-11-23T12:34:07.862927Z",
     "shell.execute_reply": "2021-11-23T12:34:07.861976Z",
     "shell.execute_reply.started": "2021-11-23T12:34:06.115981Z"
    },
    "id": "zsJcYGX6nc-H",
    "outputId": "d5b8c895-2e52-4580-e414-e7a83b2e5ed1"
   },
   "outputs": [],
   "source": [
    "import numpy as np #linear alegbra\n",
    "import pandas as pd #data processing CSV file I/O\n",
    "import matplotlib.pyplot as plt #data visualizaion\n",
    "import category_encoders as ce #data encoding\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import recall_score, precision_score, accuracy_score, plot_confusion_matrix, classification_report, f1_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-11-23T12:47:28.006503Z",
     "iopub.status.busy": "2021-11-23T12:47:28.006162Z",
     "iopub.status.idle": "2021-11-23T12:47:28.030351Z",
     "shell.execute_reply": "2021-11-23T12:47:28.02927Z",
     "shell.execute_reply.started": "2021-11-23T12:47:28.006466Z"
    },
    "id": "xwWymAnsngmG",
    "outputId": "5c05bc48-1ce6-4765-b4f7-1a84ef57ea35"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "      <th>4</th>\n",
       "      <th>5</th>\n",
       "      <th>6</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>vhigh</td>\n",
       "      <td>vhigh</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>small</td>\n",
       "      <td>low</td>\n",
       "      <td>unacc</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>vhigh</td>\n",
       "      <td>vhigh</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>small</td>\n",
       "      <td>med</td>\n",
       "      <td>unacc</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>vhigh</td>\n",
       "      <td>vhigh</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>small</td>\n",
       "      <td>high</td>\n",
       "      <td>unacc</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>vhigh</td>\n",
       "      <td>vhigh</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>med</td>\n",
       "      <td>low</td>\n",
       "      <td>unacc</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>vhigh</td>\n",
       "      <td>vhigh</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>med</td>\n",
       "      <td>med</td>\n",
       "      <td>unacc</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       0      1  2  3      4     5      6\n",
       "0  vhigh  vhigh  2  2  small   low  unacc\n",
       "1  vhigh  vhigh  2  2  small   med  unacc\n",
       "2  vhigh  vhigh  2  2  small  high  unacc\n",
       "3  vhigh  vhigh  2  2    med   low  unacc\n",
       "4  vhigh  vhigh  2  2    med   med  unacc"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "data = pd.read_csv(\"car_evaluation.csv\",header=None)\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-11-23T12:48:37.306228Z",
     "iopub.status.busy": "2021-11-23T12:48:37.305598Z",
     "iopub.status.idle": "2021-11-23T12:48:37.319438Z",
     "shell.execute_reply": "2021-11-23T12:48:37.318629Z",
     "shell.execute_reply.started": "2021-11-23T12:48:37.306192Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>buying</th>\n",
       "      <th>maint</th>\n",
       "      <th>doors</th>\n",
       "      <th>persons</th>\n",
       "      <th>lug_boot</th>\n",
       "      <th>safety</th>\n",
       "      <th>class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>vhigh</td>\n",
       "      <td>vhigh</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>small</td>\n",
       "      <td>low</td>\n",
       "      <td>unacc</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>vhigh</td>\n",
       "      <td>vhigh</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>small</td>\n",
       "      <td>med</td>\n",
       "      <td>unacc</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>vhigh</td>\n",
       "      <td>vhigh</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>small</td>\n",
       "      <td>high</td>\n",
       "      <td>unacc</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>vhigh</td>\n",
       "      <td>vhigh</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>med</td>\n",
       "      <td>low</td>\n",
       "      <td>unacc</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>vhigh</td>\n",
       "      <td>vhigh</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>med</td>\n",
       "      <td>med</td>\n",
       "      <td>unacc</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  buying  maint doors persons lug_boot safety  class\n",
       "0  vhigh  vhigh     2       2    small    low  unacc\n",
       "1  vhigh  vhigh     2       2    small    med  unacc\n",
       "2  vhigh  vhigh     2       2    small   high  unacc\n",
       "3  vhigh  vhigh     2       2      med    low  unacc\n",
       "4  vhigh  vhigh     2       2      med    med  unacc"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Changing column names for betterment\n",
    "col_names = ['buying','maint','doors','persons','lug_boot','safety','class']\n",
    "data.columns = col_names\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-11-23T12:48:40.917539Z",
     "iopub.status.busy": "2021-11-23T12:48:40.917248Z",
     "iopub.status.idle": "2021-11-23T12:48:40.931056Z",
     "shell.execute_reply": "2021-11-23T12:48:40.930118Z",
     "shell.execute_reply.started": "2021-11-23T12:48:40.91751Z"
    },
    "id": "gsOi_UoutsXM",
    "outputId": "e077219e-58be-470a-f3a1-e2e35ed5875d"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 1728 entries, 0 to 1727\n",
      "Data columns (total 7 columns):\n",
      " #   Column    Non-Null Count  Dtype \n",
      "---  ------    --------------  ----- \n",
      " 0   buying    1728 non-null   object\n",
      " 1   maint     1728 non-null   object\n",
      " 2   doors     1728 non-null   object\n",
      " 3   persons   1728 non-null   object\n",
      " 4   lug_boot  1728 non-null   object\n",
      " 5   safety    1728 non-null   object\n",
      " 6   class     1728 non-null   object\n",
      "dtypes: object(7)\n",
      "memory usage: 94.6+ KB\n"
     ]
    }
   ],
   "source": [
    "data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-11-23T12:49:01.095534Z",
     "iopub.status.busy": "2021-11-23T12:49:01.094928Z",
     "iopub.status.idle": "2021-11-23T12:49:01.105591Z",
     "shell.execute_reply": "2021-11-23T12:49:01.10469Z",
     "shell.execute_reply.started": "2021-11-23T12:49:01.095499Z"
    },
    "id": "d3qPDdAnM_u3",
    "outputId": "703fea37-a1da-4644-e328-ecfc34601337"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Feature: maint with ['vhigh' 'high' 'med' 'low'] Levels\n",
      "Feature: doors with ['2' '3' '4' '5more'] Levels\n",
      "Feature: persons with ['2' '4' 'more'] Levels\n",
      "Feature: lug_boot with ['small' 'med' 'big'] Levels\n",
      "Feature: safety with ['low' 'med' 'high'] Levels\n",
      "Feature: class with ['unacc' 'acc' 'vgood' 'good'] Levels\n"
     ]
    }
   ],
   "source": [
    "def show(data):\n",
    "  for i in data.columns[1:]:\n",
    "    print(\"Feature: {} with {} Levels\".format(i,data[i].unique()))\n",
    "\n",
    "show(data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-11-23T12:48:58.024883Z",
     "iopub.status.busy": "2021-11-23T12:48:58.024606Z",
     "iopub.status.idle": "2021-11-23T12:48:58.03405Z",
     "shell.execute_reply": "2021-11-23T12:48:58.033312Z",
     "shell.execute_reply.started": "2021-11-23T12:48:58.024855Z"
    },
    "id": "r8Y4B4r8t4Z9",
    "outputId": "b2fc97aa-e4ac-450d-913a-851f3974e686"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "buying      0\n",
       "maint       0\n",
       "doors       0\n",
       "persons     0\n",
       "lug_boot    0\n",
       "safety      0\n",
       "class       0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Checking for null values\n",
    "data.isnull().sum() "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Uf0zkaChtevM"
   },
   "source": [
    "### Feature Engineering"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-11-23T12:49:12.847066Z",
     "iopub.status.busy": "2021-11-23T12:49:12.846531Z",
     "iopub.status.idle": "2021-11-23T12:49:12.854457Z",
     "shell.execute_reply": "2021-11-23T12:49:12.853395Z",
     "shell.execute_reply.started": "2021-11-23T12:49:12.84701Z"
    },
    "id": "bvVAcWQl24zM",
    "outputId": "a5492980-4975-489c-ca54-60e12e4360e1"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "buying      object\n",
       "maint       object\n",
       "doors       object\n",
       "persons     object\n",
       "lug_boot    object\n",
       "safety      object\n",
       "class       object\n",
       "dtype: object"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-11-23T12:49:16.245054Z",
     "iopub.status.busy": "2021-11-23T12:49:16.244775Z",
     "iopub.status.idle": "2021-11-23T12:49:16.282627Z",
     "shell.execute_reply": "2021-11-23T12:49:16.281739Z",
     "shell.execute_reply.started": "2021-11-23T12:49:16.245026Z"
    },
    "id": "FQTQdtD1tNlQ"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>buying</th>\n",
       "      <th>maint</th>\n",
       "      <th>doors</th>\n",
       "      <th>persons</th>\n",
       "      <th>lug_boot</th>\n",
       "      <th>safety</th>\n",
       "      <th>class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   buying  maint  doors  persons  lug_boot  safety  class\n",
       "0       1      1      1        1         1       1      1\n",
       "1       1      1      1        1         1       2      1\n",
       "2       1      1      1        1         1       3      1\n",
       "3       1      1      1        1         2       1      1\n",
       "4       1      1      1        1         2       2      1"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "encoder = ce.OrdinalEncoder(cols = ['buying','maint','doors','persons','lug_boot','safety','class'])\n",
    "data = encoder.fit_transform(data)\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "wEZQhtrL3iUD"
   },
   "source": [
    "### Splitting Data into Train Test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-11-23T12:49:46.635054Z",
     "iopub.status.busy": "2021-11-23T12:49:46.634715Z",
     "iopub.status.idle": "2021-11-23T12:49:46.651407Z",
     "shell.execute_reply": "2021-11-23T12:49:46.650238Z",
     "shell.execute_reply.started": "2021-11-23T12:49:46.635019Z"
    },
    "id": "4uafW1Ug3oOS",
    "outputId": "3afab691-42da-424b-efad-df46707b9309"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "X_train: (1209, 6)\n",
      "X_test: (519, 6)\n",
      "Y_train: (1209,)\n",
      "Y_test: (519,)\n"
     ]
    }
   ],
   "source": [
    "x = data.drop(['class'], axis = 1)\n",
    "y = data['class']\n",
    "\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)\n",
    "print(\"X_train: {}\".format(x_train.shape))\n",
    "print(\"X_test: {}\".format(x_test.shape))\n",
    "print(\"Y_train: {}\".format(y_train.shape))\n",
    "print(\"Y_test: {}\".format(y_test.shape))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "0_hBbVkY4s69"
   },
   "source": [
    "### Data Modeling"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "5yj8bH9YADzk"
   },
   "source": [
    "##### Creating Evaluation Parametric Function"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-11-23T12:49:58.904886Z",
     "iopub.status.busy": "2021-11-23T12:49:58.904568Z",
     "iopub.status.idle": "2021-11-23T12:49:58.916589Z",
     "shell.execute_reply": "2021-11-23T12:49:58.91555Z",
     "shell.execute_reply.started": "2021-11-23T12:49:58.904857Z"
    },
    "id": "WZFPx2jR9ZHF"
   },
   "outputs": [],
   "source": [
    "def evaluation_parametrics(y_train,yp_train,y_test,yp_test):\n",
    "  print(\"--------------------------------------------------------------------------\")\n",
    "  print(\"Classification Report for Train Data\")\n",
    "  print(classification_report(y_train, yp_train))\n",
    "  print(\"Classification Report for Test Data\")\n",
    "  print(classification_report(y_test, yp_test))\n",
    "  print(\"--------------------------------------------------------------------------\")\n",
    "  # Accuracy\n",
    "  print(\"Accuracy on Train Data is: {}\".format(round(accuracy_score(y_train,yp_train),2)))\n",
    "  print(\"Accuracy on Test Data is: {}\".format(round(accuracy_score(y_test,yp_test),2)))\n",
    "  print(\"--------------------------------------------------------------------------\")\n",
    "  # Precision\n",
    "  print(\"Precision on Train Data is: {}\".format(round(precision_score(y_train,yp_train,average = \"weighted\"),2)))\n",
    "  print(\"Precision on Test Data is: {}\".format(round(precision_score(y_test,yp_test,average = \"weighted\"),2)))\n",
    "  print(\"--------------------------------------------------------------------------\")\n",
    "  # Recall \n",
    "  print(\"Recall on Train Data is: {}\".format(round(recall_score(y_train,yp_train,average = \"weighted\"),2)))\n",
    "  print(\"Recall on Test Data is: {}\".format(round(recall_score(y_test,yp_test,average = \"weighted\"),2)))\n",
    "  print(\"--------------------------------------------------------------------------\")\n",
    "  # F1 Score\n",
    "  print(\"F1 Score on Train Data is: {}\".format(round(f1_score(y_train,yp_train,average = \"weighted\"),2)))\n",
    "  print(\"F1 Score on Test Data is: {}\".format(round(f1_score(y_test,yp_test,average = \"weighted\"),2)))\n",
    "  print(\"--------------------------------------------------------------------------\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "AVOHurm_4wR9"
   },
   "source": [
    "##### 1. Logistics Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-11-23T12:50:01.765091Z",
     "iopub.status.busy": "2021-11-23T12:50:01.764764Z",
     "iopub.status.idle": "2021-11-23T12:50:02.131294Z",
     "shell.execute_reply": "2021-11-23T12:50:02.130434Z",
     "shell.execute_reply.started": "2021-11-23T12:50:01.765058Z"
    },
    "id": "nfHKo9-W4Yyt",
    "outputId": "148e3e3c-08d4-4034-a05f-07a8c7651d38"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "--------------------------------------------------------------------------\n",
      "Classification Report for Train Data\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           1       0.88      0.93      0.90       852\n",
      "           2       0.66      0.58      0.62       266\n",
      "           3       0.79      0.63      0.70        41\n",
      "           4       0.53      0.38      0.44        50\n",
      "\n",
      "    accuracy                           0.82      1209\n",
      "   macro avg       0.71      0.63      0.67      1209\n",
      "weighted avg       0.81      0.82      0.82      1209\n",
      "\n",
      "Classification Report for Test Data\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           1       0.87      0.93      0.90       358\n",
      "           2       0.66      0.58      0.62       118\n",
      "           3       0.75      0.75      0.75        24\n",
      "           4       0.55      0.32      0.40        19\n",
      "\n",
      "    accuracy                           0.82       519\n",
      "   macro avg       0.71      0.64      0.67       519\n",
      "weighted avg       0.81      0.82      0.81       519\n",
      "\n",
      "--------------------------------------------------------------------------\n",
      "Accuracy on Train Data is: 0.82\n",
      "Accuracy on Test Data is: 0.82\n",
      "--------------------------------------------------------------------------\n",
      "Precision on Train Data is: 0.81\n",
      "Precision on Test Data is: 0.81\n",
      "--------------------------------------------------------------------------\n",
      "Recall on Train Data is: 0.82\n",
      "Recall on Test Data is: 0.82\n",
      "--------------------------------------------------------------------------\n",
      "F1 Score on Train Data is: 0.82\n",
      "F1 Score on Test Data is: 0.81\n",
      "--------------------------------------------------------------------------\n"
     ]
    }
   ],
   "source": [
    "lr = LogisticRegression(max_iter = 1000,random_state = 48)\n",
    "lr.fit(x_train,y_train)\n",
    "\n",
    "yp_train = lr.predict(x_train)\n",
    "yp_test = lr.predict(x_test)\n",
    "\n",
    "evaluation_parametrics(y_train,yp_train,y_test,yp_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "uO66sXny9Q-9"
   },
   "source": [
    "##### 2. Decision Tree"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-11-23T12:50:07.05491Z",
     "iopub.status.busy": "2021-11-23T12:50:07.054411Z",
     "iopub.status.idle": "2021-11-23T12:50:07.095324Z",
     "shell.execute_reply": "2021-11-23T12:50:07.094357Z",
     "shell.execute_reply.started": "2021-11-23T12:50:07.054861Z"
    },
    "id": "VvLSqREy9Td8",
    "outputId": "93894ca3-c922-4e06-dd47-1b04bdbf3486"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "--------------------------------------------------------------------------\n",
      "Classification Report for Train Data\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           1       0.98      0.97      0.98       852\n",
      "           2       0.86      0.94      0.90       266\n",
      "           3       0.93      0.68      0.79        41\n",
      "           4       0.81      0.76      0.78        50\n",
      "\n",
      "    accuracy                           0.95      1209\n",
      "   macro avg       0.90      0.84      0.86      1209\n",
      "weighted avg       0.95      0.95      0.95      1209\n",
      "\n",
      "Classification Report for Test Data\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           1       0.97      0.98      0.97       358\n",
      "           2       0.86      0.81      0.83       118\n",
      "           3       0.77      0.83      0.80        24\n",
      "           4       0.52      0.58      0.55        19\n",
      "\n",
      "    accuracy                           0.92       519\n",
      "   macro avg       0.78      0.80      0.79       519\n",
      "weighted avg       0.92      0.92      0.92       519\n",
      "\n",
      "--------------------------------------------------------------------------\n",
      "Accuracy on Train Data is: 0.95\n",
      "Accuracy on Test Data is: 0.92\n",
      "--------------------------------------------------------------------------\n",
      "Precision on Train Data is: 0.95\n",
      "Precision on Test Data is: 0.92\n",
      "--------------------------------------------------------------------------\n",
      "Recall on Train Data is: 0.95\n",
      "Recall on Test Data is: 0.92\n",
      "--------------------------------------------------------------------------\n",
      "F1 Score on Train Data is: 0.95\n",
      "F1 Score on Test Data is: 0.92\n",
      "--------------------------------------------------------------------------\n"
     ]
    }
   ],
   "source": [
    "dt = DecisionTreeClassifier(max_depth = 7,random_state = 48) # Keeping max_depth = 7 to avoid overfitting\n",
    "dt.fit(x_train,y_train)\n",
    "\n",
    "yp_train = dt.predict(x_train)\n",
    "yp_test = dt.predict(x_test)\n",
    "\n",
    "evaluation_parametrics(y_train,yp_train,y_test,yp_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "CdisRlui-zXl"
   },
   "source": [
    "##### 3. Random Forest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-11-23T12:50:12.725532Z",
     "iopub.status.busy": "2021-11-23T12:50:12.725265Z",
     "iopub.status.idle": "2021-11-23T12:50:13.001276Z",
     "shell.execute_reply": "2021-11-23T12:50:13.000442Z",
     "shell.execute_reply.started": "2021-11-23T12:50:12.725504Z"
    },
    "id": "HV-5if3v-yAu",
    "outputId": "6d8cf0d8-1168-4cc4-b8e7-333e571f0728"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "--------------------------------------------------------------------------\n",
      "Classification Report for Train Data\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           1       0.99      0.99      0.99       852\n",
      "           2       0.88      0.98      0.93       266\n",
      "           3       0.97      0.76      0.85        41\n",
      "           4       0.97      0.74      0.84        50\n",
      "\n",
      "    accuracy                           0.97      1209\n",
      "   macro avg       0.96      0.86      0.90      1209\n",
      "weighted avg       0.97      0.97      0.97      1209\n",
      "\n",
      "Classification Report for Test Data\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           1       0.98      0.97      0.98       358\n",
      "           2       0.85      0.89      0.87       118\n",
      "           3       0.88      0.88      0.88        24\n",
      "           4       0.69      0.58      0.63        19\n",
      "\n",
      "    accuracy                           0.94       519\n",
      "   macro avg       0.85      0.83      0.84       519\n",
      "weighted avg       0.94      0.94      0.94       519\n",
      "\n",
      "--------------------------------------------------------------------------\n",
      "Accuracy on Train Data is: 0.97\n",
      "Accuracy on Test Data is: 0.94\n",
      "--------------------------------------------------------------------------\n",
      "Precision on Train Data is: 0.97\n",
      "Precision on Test Data is: 0.94\n",
      "--------------------------------------------------------------------------\n",
      "Recall on Train Data is: 0.97\n",
      "Recall on Test Data is: 0.94\n",
      "--------------------------------------------------------------------------\n",
      "F1 Score on Train Data is: 0.97\n",
      "F1 Score on Test Data is: 0.94\n",
      "--------------------------------------------------------------------------\n"
     ]
    }
   ],
   "source": [
    "rf = RandomForestClassifier(max_depth = 7,random_state = 48) # Keeping max_depth = 7 same as DT\n",
    "rf.fit(x_train,y_train)\n",
    "\n",
    "yp_train = rf.predict(x_train)\n",
    "yp_test = rf.predict(x_test)\n",
    "\n",
    "evaluation_parametrics(y_train,yp_train,y_test,yp_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "MBrph37EAK3s"
   },
   "source": [
    "### Conclusion"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "g7xT8qSfAuAt"
   },
   "source": [
    "\n",
    "1. Implemented Logistic Regression, Decision Tree & Random Forest models.\n",
    "2. Evaluated Evaluation Metrics (Confusion Matrix, Precision, Recall, Accuracy & F1-Score).\n",
    "3. Random Forest performed slightly better out of the 3 models.\n",
    "\n",
    "![Best Model.png](data:image/png;base64,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)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
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
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
