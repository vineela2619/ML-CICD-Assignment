from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_model():
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create and train the model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy


if __name__ == "__main__":
    model, accuracy = train_model()

    print("Machine Learning Model: Logistic Regression")
    print("Dataset: Iris")
    print(f"Model Accuracy: {accuracy:.2f}")

    sample_prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])
    print("Sample Prediction:", sample_prediction[0])
