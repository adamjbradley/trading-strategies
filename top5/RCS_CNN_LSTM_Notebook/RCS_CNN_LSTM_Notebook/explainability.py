import shap
import numpy as np
import matplotlib.pyplot as plt

def explain_model(model, X_sample, output_dir="logs", max_display=10):
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)

    shap.summary_plot(shap_values, X_sample, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_summary.png")
    plt.close()

    return f"{output_dir}/shap_summary.png"
