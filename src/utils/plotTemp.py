import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


energy_consumption = [
    # K-Nearest Neighbors (KNN)
    [8.637069728639392e-01, 1.0972198247909547, 3.774757683277131e-01, 1.173370193552088e-02, 1.280250474570595e-01],
    # Support Vector Classifier (SVC)
    [2.1059927896217069e-01, 1.57587386391781, 2.1059927896217069e-01, 1.1152227719624838e-02, 4.457370859019513e-02],
    # AdaBoost
    [3.6121867431534667e-01, 1.8628148977403287, 1.835870345433553e-01, 1.2179157248249763e-02, 8.5350837384626e-03],
    # Decision Tree
    [3.149090541733636e-02, 1.2118435016384831e-01, 5.853026001541704e-02, 1.156253947152032e-02, 1.1391876600218557e-02],
    # Logistic Regression
    [2.4984007632290878e-02, 3.5572935916759354e-01, 2.2454760251221836e-01, 1.0857852520766081e-02, 6.525568079757283e-03],
    # Naive Bayes
    [4.07216317123837e-02, 1.8403367731306289, 7.09026222758823, 1.1149251902544942e-02, 1.8403367731306289],
    # Random Forest
    [4.792267834698713e-01, 6.190779783107617e-01, 9.216313152401536e-01, 1.0207845105065242e-02, 6.208873506825008e-02]
]

energy_consumption_np = np.array(energy_consumption)

# Find the largest value in the matrix
max_value = np.max(energy_consumption_np)
print(max_value)

np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.2f}'.format})

# Convert to percentage
energy_percentage = (energy_consumption_np / max_value) * 100

# Print the matrix as percentages
print(np.round(energy_percentage, 2))


# Plot heatmap with annotations
plt.figure(figsize=(10, 7))
ax = sns.heatmap(energy_percentage, annot=True, cmap="YlGnBu", fmt=".2f", cbar_kws={'label': 'Percentage'})

# Adding labels
ax.set_xticklabels(['Python', 'R', 'Matlab', 'Java', 'C++'])
ax.set_yticklabels(['KNN', 'SVC', 'AdaBoost', 'DecisionTree', 'LogReg', 'NaiveBayes', 'RandomForest'], rotation=0)

# Display the plot
plt.title('Energy Consumption as Percentage per Algorithm and Language')
plt.show()
