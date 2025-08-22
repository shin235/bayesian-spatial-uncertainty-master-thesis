import numpy as np
import os
from config import KL_WEIGHT, RESULTS_DIR
from .bayesian_model import build_bayesian_pm25


def mc_predict(model, cov_input, phi_input, n_samples=200):

    loc_list = []
    scale_list = []
    samples = []

    for _ in range(n_samples):
        dist = model([cov_input, phi_input], training=True)
        samples.append(dist.mean().numpy().squeeze())
        loc = dist.loc.numpy().squeeze()
        scale = dist.scale.numpy().squeeze()
        loc_list.append(loc)
        scale_list.append(scale)

    samples = np.array(samples)
    return samples, loc_list, scale_list


def predict_all_folds(saved_models, cov_input_all, phi_input_all, n_samples=200):
    """
    Predict PM2.5 values using all saved models from 5-fold training.
    Saves fold-wise samples and ensemble mean, std across all folds.

    Returns:
    - all_samples: All MC samples from all folds
    - mean_preds: Averaged predictions across all folds and samples
    - std_preds: Standard deviation (uncertainty) across all predictions
    """

    all_samples = []

    for fold_idx, (path, _) in enumerate(saved_models):
        prediction_path = os.path.join(
            RESULTS_DIR, f"fold{fold_idx+1}_samples.npy")
        if os.path.exists(prediction_path):
            print(
                f"Skipping fold {fold_idx+1}: Prediction file already exists.")
            samples = np.load(prediction_path)
            all_samples.append(samples)
            fold_idx += 1
            continue

        model = build_bayesian_pm25(
            input_dim_phi=phi_input_all.shape[1],
            input_dim_cov=cov_input_all.shape[1],
            kl_weight=KL_WEIGHT
        )
        model.load_weights(path)

        samples, _, _ = mc_predict(
            model, cov_input_all, phi_input_all, n_samples=n_samples)

        # Save fold-wise samples
        np.save(prediction_path, samples)
        all_samples.append(samples)

    N = phi_input_all.shape[0]
    all_samples = np.array(all_samples).reshape(-1, N)

    mean_preds = np.mean(all_samples, axis=0)
    std_preds = np.std(all_samples, axis=0)

    # Save ensemble predictions
    np.save(os.path.join(RESULTS_DIR, "ensemble_mean.npy"), mean_preds)
    np.save(os.path.join(RESULTS_DIR, "ensemble_std.npy"), std_preds)

    print("Prediction complete.")
    print(f"Prediction shape: {mean_preds.shape}")

    return all_samples, mean_preds, std_preds
