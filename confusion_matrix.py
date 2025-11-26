
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load model and test data
rf, scaler, feature_cols, encoders = joblib.load(r"C:\Users\palak priya\OneDrive\Desktop\fraud_upi\models\fraud_model.pkl")
X_test, y_test = joblib.load(r'C:\Users\palak priya\OneDrive\Desktop\fraud_upi\models\test_data.pkl')

# Predict
y_pred = rf.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Genuine', 'Fraud'])

fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap='Reds', values_format='d')
plt.title('Confusion Matrix: Fraud Detection')
plt.show()

# Print metrics for insight
print('Confusion Matrix:')
print(cm)
print('\nTrue Negatives:', cm[0,0])
print('False Positives:', cm[0,1])
print('False Negatives:', cm[1,0])
print('True Positives:', cm[1,1])
