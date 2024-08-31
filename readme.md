* [Presentation](https://github.com/MMJGGR/Phase_3_Project/blob/master/presentation.pdf)
* [Jupyter Notebook](https://github.com/MMJGGR/Phase_3_Project/blob/master/index.ipynb)
* [Notebook PDF](https://github.com/MMJGGR/Phase_3_Project/blob/master/index.pdf)
* [Data Sources](https://data.seattle.gov/Public-Safety/Terry-Stops/28ny-9ts8/about_data)



# **Terry Stops Classification Project**

## **Overview**

This project involves developing a predictive model to classify whether a Terry Stop in Seattle will result in an arrest. Utilizing historical data on Terry Stops, we aim to build a robust classification model that can help law enforcement and city authorities better understand the factors contributing to arrests during these stops. Our analysis explores the use of logistic regression and decision tree models to achieve this goal, ultimately selecting the most appropriate model based on its performance metrics and relevance to the stakeholder's needs.

## **Business and Data Understanding**

### **Stakeholder Audience**

Our primary stakeholders include the Seattle Police Department, city policymakers, and community advocacy groups. These stakeholders are interested in gaining insights into how various factors influence the likelihood of an arrest during a Terry Stop. By understanding these patterns, stakeholders can make informed decisions about law enforcement practices, resource allocation, and community engagement strategies.

### **Dataset Choice**

The dataset used in this project comprises historical records of Terry Stops in Seattle. The dataset includes various features such as the type of call, the race and gender of both the officer and subject, the presence of a weapon, the geographic location of the stop (Beat, Sector, Precinct), and the officer's squad. The target variable is whether the stop resulted in an arrest (True or False). This dataset was chosen because it provides a comprehensive view of the circumstances under which Terry Stops occur and their outcomes, making it suitable for building a predictive model.

## **Modeling**
### **Model Selection**

We experimented with two primary models for this classification task:

- **Logistic Regression:** A linear model that is well-suited for binary classification tasks and provides interpretability through the analysis of feature coefficients.

- **Decision Tree Classifier**: A non-linear model that can capture complex interactions between features and is particularly useful for understanding the decision-making process through its tree structure.

During the modeling phase, we initially developed a logistic regression model to establish a baseline performance. To address class imbalance, we employed Synthetic Minority Over-sampling Technique (SMOTE) to enhance the model's ability to correctly predict the minority class (arrests). Subsequently, we introduced a decision tree classifier, fine-tuning it by determining the optimal tree depth based on AUC (Area Under the Curve) scores.

### **Model evaluation**

The models were evaluated using several key metrics:

- Accuracy: The proportion of correct predictions over the total number of instances.
- Precision: The ratio of true positive predictions to the total number of positive predictions.
- Recall: The ratio of true positive predictions to the total number of actual positives.
- F1 Score: The harmonic mean of precision and recall, providing a balanced metric that accounts for both false positives and false negatives.
-AUC: The Area Under the Receiver Operating Characteristic Curve, a measure of the model's ability to distinguish between the classes.

### **Performance Summary**

- Logistic Regression: Achieved higher recall and AUC, making it more effective at correctly identifying stops that result in an arrest.
- Decision Tree: While providing competitive accuracy and AUC, the decision tree model showed a lower recall, indicating that it may miss some arrest cases.

## **Conclusion**
Based on the evaluation metrics and our stakeholder needs, we selected the Logistic Regression model as the final model for deployment. Despite the decision tree's interpretability, the logistic regression model offers superior recall and AUC, making it a more reliable tool for identifying potential arrests during Terry Stops. This decision aligns with our objective to minimize false negatives (i.e., failing to identify an arrest) while maintaining a high overall accuracy.

This model can serve as a decision-support tool for law enforcement, helping to prioritize stops that are more likely to require further action. Future improvements may involve exploring additional features, refining SMOTE application, and considering ensemble methods to further enhance prediction performance.