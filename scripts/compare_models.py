import webbrowser

html = """
<!DOCTYPE html>
<html>
<head>
  <title>Model Comparison: XGBoost vs LightGBM</title>
  <style>
    body { font-family: Arial, sans-serif; padding: 20px; background: #f4f4f4; }
    h1 { color: #333; }
    .row { display: flex; gap: 20px; margin-bottom: 40px; }
    .column { flex: 1; background: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
    img { max-width: 100%; border: 1px solid #ccc; }
    iframe { width: 100%; height: 500px; border: 1px solid #ccc; }
  </style>
</head>
<body>

<h1>🧠 Model Comparison: XGBoost vs LightGBM</h1>

<h2>📊 SHAP Summary Plots</h2>
<div class="row">
  <div class="column">
    <h3>XGBoost</h3>
    <img src="shap_summary_xgboost.png" alt="SHAP XGBoost">
  </div>
  <div class="column">
    <h3>LightGBM</h3>
    <img src="shap_summary_lightgbm.png" alt="SHAP LightGBM">
  </div>
</div>

<h2>🔍 SHAP Force Plots (Sample Index 0)</h2>
<div class="row">
  <div class="column">
    <h3>XGBoost</h3>
    <img src="shap_force_xgboost_0.png" alt="Force XGBoost">
  </div>
  <div class="column">
    <h3>LightGBM</h3>
    <img src="shap_force_lightgbm_0.png" alt="Force LightGBM">
  </div>
</div>

<h2>📋 LIME Explanations (Sample Index 0)</h2>
<div class="row">
  <div class="column">
    <h3>XGBoost</h3>
    <iframe src="lime_explanation_xgboost_0.html"></iframe>
  </div>
  <div class="column">
    <h3>LightGBM</h3>
    <iframe src="lime_explanation_lightgbm_0.html"></iframe>
  </div>
</div>

</body>
</html>
"""

with open("model_comparison.html", "w", encoding="utf-8") as f:
    f.write(html)

# Open in browser
webbrowser.open("model_comparison.html")
