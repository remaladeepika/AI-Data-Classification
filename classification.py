# ============================================================
# AI DATA CLASSIFICATION USING DECISION TREE
# DecodeLabs AI Internship - Project 2
# ============================================================

import os

import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

import matplotlib

# Use a non-GUI backend so this file works correctly with Flask.
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# STEP 1 : LOAD DATASET
# ============================================================

iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names


# Create assets directory if it does not exist
ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "assets"
)

os.makedirs(ASSETS_DIR, exist_ok=True)


# ============================================================
# STEP 2 : DATASET SUMMARY
# ============================================================

df = pd.DataFrame(
    X,
    columns=feature_names
)

df["Species"] = y


total_samples = len(X)

total_features = len(feature_names)

target_classes = list(target_names)


# ============================================================
# STEP 3 : SPLIT DATASET
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


training_samples = len(X_train)

testing_samples = len(X_test)


# ============================================================
# STEP 4 : TRAIN DECISION TREE MODEL
# ============================================================

model = DecisionTreeClassifier(
    random_state=42
)

model.fit(
    X_train,
    y_train
)


# ============================================================
# STEP 5 : MAKE PREDICTIONS
# ============================================================

predictions = model.predict(X_test)


# ============================================================
# STEP 6 : MODEL ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)


# ============================================================
# STEP 7 : CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    predictions,
    target_names=target_names,
    output_dict=True
)


# ============================================================
# STEP 8 : CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions
)


# ============================================================
# STEP 9 : CREATE CONFUSION MATRIX IMAGE
# ============================================================

confusion_matrix_path = os.path.join(
    ASSETS_DIR,
    "confusion_matrix.png"
)


fig, ax = plt.subplots(
    figsize=(7, 5)
)

ax.imshow(cm)

ax.set_title(
    "Confusion Matrix"
)

ax.set_xlabel(
    "Predicted Label"
)

ax.set_ylabel(
    "Actual Label"
)

ax.set_xticks(
    range(len(target_names))
)

ax.set_yticks(
    range(len(target_names))
)

ax.set_xticklabels(
    target_names
)

ax.set_yticklabels(
    target_names
)


# Display values inside the matrix
for i in range(len(cm)):

    for j in range(len(cm[i])):

        ax.text(
            j,
            i,
            cm[i][j],
            ha="center",
            va="center"
        )


plt.tight_layout()

plt.savefig(
    confusion_matrix_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# STEP 10 : CREATE DECISION TREE IMAGE
# ============================================================

decision_tree_path = os.path.join(
    ASSETS_DIR,
    "decision_tree.png"
)


fig, ax = plt.subplots(
    figsize=(15, 8)
)

plot_tree(
    model,
    feature_names=feature_names,
    class_names=target_names,
    filled=True,
    rounded=True,
    ax=ax
)

ax.set_title(
    "Decision Tree Classifier"
)

plt.tight_layout()

plt.savefig(
    decision_tree_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# STEP 11 : PREDICT CUSTOM FLOWER
# ============================================================

def predict_flower(
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
):
    """
    Predict the Iris flower species using four measurements.
    """

    sample = [[
        float(sepal_length),
        float(sepal_width),
        float(petal_length),
        float(petal_width)
    ]]

    prediction = model.predict(
        sample
    )

    predicted_class = int(
        prediction[0]
    )

    flower_name = target_names[
        predicted_class
    ]

    # Probability scores
    probabilities = model.predict_proba(
        sample
    )[0]

    confidence = float(
        probabilities[predicted_class] * 100
    )

    return {
        "class_id": predicted_class,
        "flower": flower_name,
        "confidence": round(confidence, 2)
    }


# ============================================================
# STEP 12 : GET MODEL INFORMATION
# ============================================================

def get_model_info():
    """
    Return useful information for the Flask frontend.
    """

    return {
        "dataset": "Iris Dataset",

        "total_samples": total_samples,

        "total_features": total_features,

        "features": feature_names,

        "classes": target_classes,

        "training_samples": training_samples,

        "testing_samples": testing_samples,

        "accuracy": round(
            accuracy * 100,
            2
        ),

        "confusion_matrix": cm.tolist(),

        "classification_report": report,

        "confusion_matrix_image":
            "/assets/confusion_matrix.png",

        "decision_tree_image":
            "/assets/decision_tree.png"
    }


# ============================================================
# STEP 13 : TERMINAL OUTPUT
# ============================================================

def display_results():
    """
    Display model results when classification.py
    is executed directly from the terminal.
    """

    print("=" * 60)

    print(
        "      AI DATA CLASSIFICATION USING DECISION TREE"
    )

    print("=" * 60)


    print("\nDataset Loaded Successfully!")

    print(
        "Total Samples :",
        total_samples
    )

    print(
        "Total Features:",
        total_features
    )

    print(
        "Target Classes:",
        target_classes
    )


    print("\nFeature Names:")

    for feature in feature_names:

        print(
            "-",
            feature
        )


    print("\n========== DATASET PREVIEW ==========")

    print(
        df.head()
    )


    print("\n========== DATA SPLIT ==========")

    print(
        "Training Samples :",
        training_samples
    )

    print(
        "Testing Samples  :",
        testing_samples
    )


    print("\n========== MODEL ACCURACY ==========")

    print(
        f"Accuracy : {accuracy * 100:.2f}%"
    )


    print("\n========== CLASSIFICATION REPORT ==========")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=target_names
        )
    )


    print("\n========== CONFUSION MATRIX ==========")

    print(cm)


    print("\n========== GENERATED FILES ==========")

    print(
        "Confusion Matrix:",
        confusion_matrix_path
    )

    print(
        "Decision Tree:",
        decision_tree_path
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    display_results()

    print(
        "\nModel is ready for predictions."
    )

    print(
        "The Flask application can now import "
        "predict_flower() and get_model_info()."
    )

