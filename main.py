import shap
import matplotlib.pyplot as plt

def explain_with_shap(model, X_train):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    # Summary Plot (global feature importance)
    print("Generating SHAP summary plot...")
    shap.summary_plot(shap_values, X_train, show=False)
    plt.savefig("shap_summary.png")
    print("Saved global SHAP plot as shap_summary.png")
