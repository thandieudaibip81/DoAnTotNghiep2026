import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import io
import base64

model = joblib.load('models/random_forest_smote.pkl')
scaler = joblib.load('models/robust_scaler.pkl')

df_input = pd.DataFrame([np.random.randn(30)], columns=["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"])

explainer = shap.TreeExplainer(model)
shap_values = explainer(df_input)

if len(shap_values.shape) == 3:
    explanation = shap_values[0, :, 1]
else:
    explanation = shap_values[0]

plt.figure()
shap.plots.waterfall(explanation, show=False)
buf = io.BytesIO()
plt.savefig(buf, format='png')
print("Base64 length:", len(base64.b64encode(buf.getvalue())))

print("Now testing logistic regression...")
model_lr = joblib.load('models/logistic_regression_smote.pkl')
background = np.zeros((1, 30))
explainer_lr = shap.Explainer(model_lr.predict_proba, background)
shap_values_lr = explainer_lr(df_input)
exp_lr = shap_values_lr[0, :, 1]
plt.figure()
shap.plots.waterfall(exp_lr, show=False)
buf_lr = io.BytesIO()
plt.savefig(buf_lr, format='png')
print("LR Base64 length:", len(base64.b64encode(buf_lr.getvalue())))
print("Done")
