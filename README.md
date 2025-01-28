Advertising and Sales Prediction using Linear Regression

This project aims to analyze the impact of advertising spend across different media channels on sales and to build a linear regression model to predict sales based on advertising costs.

Table of Contents
Introduction

Dataset

Requirements

Usage

Visualizations

Model Training and Evaluation

Predictions

Conclusion

Introduction
In this project, we use a dataset containing advertising costs for TV, Radio, and Newspaper along with corresponding sales figures. Our goals are to:

Understand the relationship between advertising spend and sales.

Build a linear regression model to predict sales based on advertising costs.

Evaluate the performance of the model using various metrics.

Dataset
The dataset contains the following columns:

TV: Advertising cost spent on TV in dollars.

Radio: Advertising cost spent on Radio in dollars.

Newspaper: Advertising cost spent on Newspaper in dollars.

Sales: Number of units sold.

Sample data:

TV	Radio	Newspaper	Sales
230.1	37.8	69.2	22.1
44.5	39.3	45.1	10.4
17.2	45.9	69.3	12.0
151.5	41.3	58.5	16.5
180.8	10.8	58.4	17.9
Requirements
Python 3.x

NumPy

Pandas

Scikit-learn

Plotly

Usage

Load the dataset and preprocess the data.

Perform exploratory data analysis (EDA) to understand the relationships between variables.

Train a linear regression model using the training data.

Evaluate the model using the testing data.

Visualize the results using Plotly.

Visualizations

The project includes interactive visualizations to explore the relationships between sales and advertising costs across different media channels.

Scatter Plot for TV Advertising:

figure = px.scatter(data_frame=data, x='Sales', y='TV', size='TV', trendline='ols')
figure.show()
Scatter Plot for Newspaper Advertising:

figure = px.scatter(data_frame=data, x='Sales', y='Newspaper', size='Newspaper', trendline='ols')
figure.show()
Scatter Plot for Radio Advertising:

figure = px.scatter(data_frame=data, x='Sales', y='Radio', size='Radio', trendline='ols')
figure.show()
Model Training and Evaluation
Splitting the Data:

x = np.array(data.drop(["Sales"], axis=1))
y = np.array(data["Sales"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

Training the Model:

model = LinearRegression()
model.fit(xtrain, ytrain)
Evaluating the Model:

print(model.score(xtest, ytest))  # R-squared value
Calculating Evaluation Metrics:

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = model.predict(xtest)
mae = mean_absolute_error(ytest, y_pred)
mse = mean_squared_error(ytest, y_pred)
r2 = r2_score(ytest, y_pred)

print(f'MAE: {mae}, MSE: {mse}, R-squared: {r2}')
Predictions
You can make predictions using the trained model:

features = np.array([[230.1, 37.8, 69.2]])
print(model.predict(features))  # Predicted Sales: [21.37254028]

Conclusion
This project demonstrates the use of linear regression to analyze the impact of advertising spend on sales and to build a predictive model. The results can guide decision-making in advertising budget allocation to maximize sales.
