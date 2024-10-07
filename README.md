# YT-Spam-Comment-Classification

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Scikit--learn-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/Framework-XGBoost-success)](https://xgboost.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/Optimization-Optuna-lightgrey)](https://optuna.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Project Overview

YT-Spam-Comment-Classification is a machine learning project focused on detecting spam comments on YouTube using advanced natural language processing (NLP) techniques. The project employs an ensemble learning approach with XGBoost, combined with hyperparameter tuning using Optuna, to achieve robust spam detection performance.

## Table of Contents
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Technologies Used

The following key technologies and frameworks were used in the project:

- **Python 3.x**: Programming language.
- **Jupyter Notebook**: For creating and managing the codebase.
- **Pandas**: Data manipulation and analysis.
- **Scikit-learn**: Machine learning library used for feature extraction and model evaluation.
- **TF-IDF Vectorizer**: Converts text data into numerical form using the Term Frequency-Inverse Document Frequency technique.
- **XGBoost**: Gradient Boosting framework used for classification.
- **Optuna**: Hyperparameter optimization framework to tune the XGBoost model parameters.
- **Joblib**: For saving and loading trained models and other large data objects.

## Project Structure

The repository structure is organized as follows:

YT-Spam-Comment-Classification/
├── Model.ipynb                # Main Jupyter Notebook with all steps and code.
├── xgboost_spam_comment_classifier_optimized.json  # Optimized XGBoost model file.
├── tfidf_vectorizer.pkl       # Saved TF-IDF vectorizer.
├── README.md                  # Project README file.
└── LICENSE                    # Project License.

## Installation

To run the code in this repository, make sure you have the following installed:

- Python 3.x
- Jupyter Notebook
- The required Python packages, which can be installed using the following command:

```bash
pip install -r requirements.txt
```
**Note:** Create a `requirements.txt` file that includes the dependencies:

