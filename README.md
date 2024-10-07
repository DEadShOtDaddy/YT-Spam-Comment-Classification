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

- **YT-Spam-Comment-Classification/**
  - `Model.ipynb`: Main Jupyter Notebook with all steps and code.
  - `xgboost_spam_comment_classifier_optimized.json`: Optimized XGBoost model file.
  - `tfidf_vectorizer.pkl`: Saved TF-IDF vectorizer.
  - `README.md`: Project README file.
  - `LICENSE`: Project License.
  - `requirements.txt`: Required packages.


## Installation

To run the code in this repository, make sure you have the following installed:

- Python 3.x
- Jupyter Notebook
- The required Python packages, which can be installed using the following command:

```bash
pip install -r requirements.txt
```
**Note:** Create a `requirements.txt` file that includes the dependencies:
## Usage
**1.Clone the Repository:** 
```bash
git clone https://github.com/DEadShOtDaddy/YT-Spam-Comment-Classification.git
cd YT-Spam-Comment-Classification
```
**2. Run the Jupyter Notebook: Open the `Model.ipynb file` and execute the cells to see the code in action.**

**3.Model Training: The model can be retrained by running the cells related to XGBoost training and hyperparameter tuning using Optuna.**

**4.Inference: Use the saved model `xgboost_spam_comment_classifier_optimized.json` to make predictions on new data using the saved TF-IDF vectorizer.**

## DataSet
The dataset used in this project contains YouTube comments labeled as spam or not spam. This dataset is preprocessed to ensure that it suits the machine learning requirements, using techniques such as:
- Tokenization
- TF-IDF Vectorization
- Train-test split (80-20)

**Note:** Ensure that your dataset is properly formatted with two columns: `CONTENT` (the comment text) and `CLASS` (spam=1 or not_spam=0).

## Model Training
The model is trained using the following steps:

**1.TF-IDF Vectorization:** Converts the text into a sparse matrix representation.

**2.Hyperparameter Tuning:** Uses Optuna to find the best hyperparameters for the XGBoost classifier.

**3.Training:** Trains the XGBoost model with the optimized parameters on the dataset.

**4.Evaluation:** Evaluates the model's performance using metrics like accuracy and log loss.

## Results
The best XGBoost model achieved an accuracy of approximately 94% using the optimized hyperparameters obtained through Optuna. This demonstrates the model's effectiveness in distinguishing spam comments from legitimate ones.

## Contributing
Contributions to this project are welcome! If you have ideas for improvement or have found issues, feel free to open an issue or submit a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For any query,feedback or advise contact [DEadShOtDaddy](https://github.com/DEadShOtDaddy)

