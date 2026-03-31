from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_svm(X_train, y_train, C, gamma):
    model = SVC(C=C, gamma=gamma)
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test, )
    return accuracy_score(y_test, y_pred)

def detailed_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))