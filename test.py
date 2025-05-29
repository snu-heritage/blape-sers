# %%
import blape
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

# Download and load data
blape.download_data(path='data')
data = blape.read_data(path='data')

# Determine the common wavenumber range across all data samples
target_wn = blape.get_common_wavenumber_range(data)
print(f'Common wavenumber range: {target_wn[0]} cm-1 to {target_wn[-1]} cm-1')

# Apply BLaPE algorithm to all samples
print(f'Processing BLaPE...')
pbar = tqdm(data.items())
for code, d in pbar:
    d['blape'] = blape.blape(d['signal'], original_wn=d['wavenumbers'], target_wn=target_wn)
    pbar.set_postfix_str(f"Current sample: {code}")

# %%
# Visualize BLaPE for random sample
random_code = random.choice(list(data.keys()))
sample = data[random_code]
print(f'Selected sample: {random_code}')

plt.figure(figsize=(12, 10))

# Raw signal
subplot = plt.subplot(3, 1, 1)
plt.plot(sample['wavenumbers'], sample['signal'][:5].T, alpha=0.3, color='b')
plt.title(f'Raw Signal {random_code}')
plt.xlabel('Wavenumber (cm-1)')
plt.ylabel('Intensity')
plt.grid(True, alpha=0.3)

# Baseline removed
subplot = plt.subplot(3, 1, 2)
plt.plot(sample['wavenumbers'], sample['baseline_removed'][:5].T, alpha=0.3, color='b')
plt.title(f'Baseline Removed {random_code}')
plt.xlabel('Wavenumber (cm-1)')
plt.ylabel('Intensity')
plt.grid(True, alpha=0.3)

# BLaPE processed
subplot = plt.subplot(3, 1, 3)
plt.plot(target_wn, sample['blape'][:5].T, alpha=0.3, color='b')
plt.title(f'BLaPE Processed {random_code}')
plt.xlabel('Wavenumber (cm-1)')
plt.ylabel('Peak Enhancement')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# %%
# Quick evaluation of BLaPE features, with default sigma = 25-cm
# Prepare multilabel data
X, y_dict, label_encoders = blape.prepare_multilabel_data(data, feature_key='blape')
print(f"Feature matrix shape: {X.shape}")
print(f"Categories: {list(y_dict.keys())}")

# Show label distribution for each category
for category, labels in y_dict.items():
    unique_labels, counts = np.unique(labels, return_counts=True)
    class_names = label_encoders[category].classes_
    print(f"\n{category} distribution:")
    for i, (label, count) in enumerate(zip(unique_labels, counts)):
        print(f"  {class_names[label]}: {count} samples")

# Train multilabel models
print("\nTraining multilabel classification models...")
models, X_train, X_test, y_train_dict, y_test_dict = blape.train_multilabel_models(
    X, y_dict, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Evaluate models
print("\nEvaluating models...")
results = blape.evaluate_multilabel_models(models, X_test, y_test_dict, label_encoders)

# Print overall accuracy summary
print("\n=== OVERALL ACCURACY SUMMARY ===")
for category, result in results.items():
    print(f"{category}: {result['accuracy']:.3f}")
    
# %%
# Compare different regularization methods on baseline_removed features
print("Preparing baseline_removed data for regularization comparison...")

# Interpolate baseline_removed to target_wn for consistent dimensions
pbar = tqdm(data.items())
for code, d in pbar:
    if 'baseline_removed' in d and d['baseline_removed'] is not None:
        baseline_removed_interp = []
        for i in range(d['baseline_removed'].shape[0]):
            interp_spectrum = np.interp(target_wn, d['wavenumbers'], d['baseline_removed'][i])
            baseline_removed_interp.append(interp_spectrum)
        d['baseline_removed_interp'] = np.array(baseline_removed_interp)
    pbar.set_postfix_str(f"Interpolating: {code}")

print("Comparing regularization methods on baseline_removed features...")

regularization_methods = ['l1', 'l2', 'minmax', 'stdnorm']
accuracy_results = {}

for reg_method in regularization_methods:
    print(f"Testing regularization: {reg_method}")
    
    # Prepare data with specific regularization
    X_reg, y_dict_reg, label_encoders_reg = blape.prepare_multilabel_data(
        data, feature_key='baseline_removed_interp', regularization=reg_method
    )
    
    # Train models
    models_reg, X_train_reg, X_test_reg, y_train_dict_reg, y_test_dict_reg = blape.train_multilabel_models(
        X_reg, y_dict_reg, test_size=0.2
    )
    
    # Evaluate models
    results_reg = blape.evaluate_multilabel_models(
        models_reg, X_test_reg, y_test_dict_reg, label_encoders_reg, verbose=False
    )
    
    # Store accuracies
    accuracy_results[reg_method] = {}
    for category, result in results_reg.items():
        accuracy_results[reg_method][category] = result['accuracy']

# Display comparison results
print("\n=== REGULARIZATION COMPARISON RESULTS ===")
print("Method\t\tBase\tDye\tMordant\tAging")
print("-" * 50)
for reg_method, accuracies in accuracy_results.items():
    print(f"{reg_method:<10}\t{accuracies['base']:.3f}\t{accuracies['dye']:.3f}\t{accuracies['mordant']:.3f}\t{accuracies['aging']:.3f}")

# %%
# Test different sigma values for BLaPE
print("Testing different sigma values for BLaPE...")

sigma_values = np.arange(10, 51, 5)  # 10, 15, 20, ..., 50
sigma_results = {}

for sigma in tqdm(sigma_values, desc="Testing sigma values"):
    print(f"Testing sigma: {sigma}")
    
    # Apply BLaPE with different sigma
    for code, d in data.items():
        d['blape_sigma'] = blape.blape(d['signal'], original_wn=d['wavenumbers'], 
                                      target_wn=target_wn, sigma=sigma)
    
    # Prepare data and train models
    X_sigma, y_dict_sigma, label_encoders_sigma = blape.prepare_multilabel_data(
        data, feature_key='blape_sigma', regularization='stdnorm'
    )
    
    models_sigma, X_train_sigma, X_test_sigma, y_train_dict_sigma, y_test_dict_sigma = blape.train_multilabel_models(
        X_sigma, y_dict_sigma, test_size=0.2, random_state=42
    )
    
    results_sigma = blape.evaluate_multilabel_models(
        models_sigma, X_test_sigma, y_test_dict_sigma, label_encoders_sigma, verbose=False
    )
    
    # Store results
    sigma_results[sigma] = {}
    for category, result in results_sigma.items():
        sigma_results[sigma][category] = 1 - result['accuracy']  # Convert to error rate

# Print best sigma for each category
print("\n=== BEST SIGMA FOR EACH CATEGORY ===")
for category in categories:
    best_sigma = min(sigma_results.keys(), key=lambda x: sigma_results[x][category])
    best_error = sigma_results[best_sigma][category]
    best_acc = 1 - best_error
    print(f"{category}: sigma={best_sigma} (accuracy={best_acc:.3f}, error={best_error:.3f})")

# %%

# Calculate mean error rates for each regularization method
reg_mean_errors = {}
for reg_method in regularization_methods:
    reg_mean_errors[reg_method] = {}
    for category in categories:
        reg_mean_errors[reg_method][category] = (1 - accuracy_results[reg_method][category]) * 100

# BLaPE error rates
blape_errors = {}
for category in categories:
    blape_errors[category] = [sigma_results[sigma][category] * 100 for sigma in sigma_results.keys()]

# Plot results
plt.figure(figsize=(10, 8))

# Dye
plt.subplot(221)
plt.axhline(y=reg_mean_errors['stdnorm']['dye'], color='r', linestyle='--', label='stdnorm')
plt.axhline(y=reg_mean_errors['minmax']['dye'], color='g', linestyle='--', label='minmax')
plt.axhline(y=reg_mean_errors['l1']['dye'], color='b', linestyle='--', label='L1')
plt.axhline(y=reg_mean_errors['l2']['dye'], color='y', linestyle='--', label='L2')
plt.scatter(sigma_results.keys(), blape_errors['dye'], label='BLaPE', s=20, color='black')
plt.plot(sigma_results.keys(), blape_errors['dye'], color='black', alpha=0.5)
plt.ylim(0, 50)
plt.title('Dye')
plt.xlabel('Sigma')
plt.ylabel('Error Rate (%)')
plt.legend()

# Base
plt.subplot(222)
plt.axhline(y=reg_mean_errors['stdnorm']['base'], color='r', linestyle='--', label='stdnorm')
plt.axhline(y=reg_mean_errors['minmax']['base'], color='g', linestyle='--', label='minmax')
plt.axhline(y=reg_mean_errors['l1']['base'], color='b', linestyle='--', label='L1')
plt.axhline(y=reg_mean_errors['l2']['base'], color='y', linestyle='--', label='L2')
plt.scatter(sigma_results.keys(), blape_errors['base'], label='BLaPE', s=20, color='black')
plt.plot(sigma_results.keys(), blape_errors['base'], color='black', alpha=0.5)
plt.title('Base')
plt.xlabel('Sigma')
plt.ylabel('Error Rate (%)')
plt.legend()
plt.ylim(0, 50)

# Mordant
plt.subplot(223)
plt.axhline(y=reg_mean_errors['stdnorm']['mordant'], color='r', linestyle='--', label='stdnorm')
plt.axhline(y=reg_mean_errors['minmax']['mordant'], color='g', linestyle='--', label='minmax')
plt.axhline(y=reg_mean_errors['l1']['mordant'], color='b', linestyle='--', label='L1')
plt.axhline(y=reg_mean_errors['l2']['mordant'], color='y', linestyle='--', label='L2')
plt.scatter(sigma_results.keys(), blape_errors['mordant'], label='BLaPE', s=20, color='black')
plt.plot(sigma_results.keys(), blape_errors['mordant'], color='black', alpha=0.5)
plt.title('Mordant')
plt.xlabel('Sigma')
plt.ylabel('Error Rate (%)')
plt.ylim(0, 50)
plt.legend()

# Aging (degradation)
plt.subplot(224)
plt.axhline(y=reg_mean_errors['stdnorm']['aging'], color='r', linestyle='--', label='stdnorm')
plt.axhline(y=reg_mean_errors['minmax']['aging'], color='g', linestyle='--', label='minmax')
plt.axhline(y=reg_mean_errors['l1']['aging'], color='b', linestyle='--', label='L1')
plt.axhline(y=reg_mean_errors['l2']['aging'], color='y', linestyle='--', label='L2')
plt.scatter(sigma_results.keys(), blape_errors['aging'], label='BLaPE', s=20, color='black')
plt.plot(sigma_results.keys(), blape_errors['aging'], color='black', alpha=0.5)
plt.title('Aging')
plt.xlabel('Sigma')
plt.ylabel('Error Rate (%)')
plt.ylim(0, 50)
plt.legend()

plt.tight_layout()
plt.show()

# %%
