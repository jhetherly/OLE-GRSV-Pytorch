import numpy as np
import torch


def compute_projection_matrices_in_memory(dataloader, model, min_singular_value_fraction):
    features_dict = {}
    device = next(model.parameters()).device
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        features = model.feature_vector(inputs)
        features = np.squeeze(features.cpu().data.numpy())

        labels = labels.cpu().data.numpy()
        for label in np.unique(labels):
            indices = label == labels

            if label not in features_dict:
                features_dict[label] = features[indices]
            else:
                features_dict[label] = np.concatenate([features_dict[label], features[indices]], axis=0)

    transforms_list = []
    for label in sorted(features_dict.keys()):
        u,s,v = np.linalg.svd(features_dict[label], full_matrices=False)
        valid_indices = s/np.max(s) > min_singular_value_fraction
        v = v[valid_indices]
        transforms_list.append(np.dot(v.T, v))
    transforms = np.stack(transforms_list)

    return features_dict, transforms

def evaluate(dataloader, model, projections, ole_grsv):
    device = next(model.parameters()).device
    y_true = []
    y_pred = []
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        y_pred_model = ole_grsv.predict_from_projections(projections, model.feature_vector(inputs))

        y_true += labels.cpu().data.numpy().tolist()
        y_pred += torch.argmax(y_pred_model, dim=1).cpu().data.numpy().tolist()
    
    return y_true, y_pred