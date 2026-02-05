from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report


def tune_hyperparameters(X_train, y_train):
    param_grid = {
        'C': [0.1, 1, 3, 5, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }

    model = LogisticRegression(
        max_iter=2000,
        class_weight='balanced'
    )

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_, grid.best_score_


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return acc, report
