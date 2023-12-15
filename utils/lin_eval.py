import numpy as np
import torch
from utils.eval_helper import binary_metrics, prob_metrics
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier


def get_representations(algorithm, loader, device):
    ys, atts, zs = [], [], []

    algorithm.eval()
    with torch.no_grad():
        for _, x, y, a in loader:
            z = algorithm.return_feats(x.to(device)).detach().cpu().numpy()
            zs.append(z)
            ys.append(y)
            atts.append(a)

    return np.concatenate(zs, axis=0), np.concatenate(atts, axis=0), np.concatenate(ys, axis=0)


def fit_model(train_X, train_Y, val_X, val_Y, test_X, test_Y, model_type='lr'):
    if model_type == 'lr':
        pipe = Pipeline(steps=[
            ('model', LogisticRegression(random_state=42, n_jobs=-1))
        ])
        param_grid = {
            'model__C': 10**np.linspace(-5, 1, 10)
        }
    elif model_type == 'rf':
        pipe = Pipeline(steps=[
            ('model', RandomForestClassifier(random_state=42, n_jobs=-1))
            # ('model', XGBClassifier(random_state=42, n_jobs=-1))
        ])
        param_grid = {
            'model__max_depth': list(range(1, 7))
        }
    else:
        raise NotImplementedError

    pds = PredefinedSplit(test_fold=np.concatenate([np.ones((len(train_X),))*-1, np.zeros((len(val_X),))]))

    cv_lr = (GridSearchCV(pipe, param_grid, refit=False, cv=pds, scoring='roc_auc_ovr', verbose=10, n_jobs=-1).fit(
        np.concatenate((train_X, val_X)), np.concatenate((train_Y, val_Y))))

    pipe = clone(
        clone(pipe).set_params(**cv_lr.best_params_)
    )
    pipe = pipe.fit(train_X, train_Y)

    label_set = np.sort(np.unique(train_Y))
    res = {}
    for sset, X, Y in zip(['va', 'te'], [val_X, test_X], [val_Y, test_Y]):
        preds = pipe.predict_proba(X)
        if len(label_set) == 2:
            preds = preds[:, 1]
            preds_rounded = preds >= 0.5
        else:
            preds_rounded = preds.argmax(1)

        res[sset] = binary_metrics(Y, preds_rounded, label_set=label_set, return_arrays=True)
        prob_mets = prob_metrics(Y, preds, label_set=label_set, return_arrays=True)
        prob_mets['pred_probs'] = prob_mets['preds']
        del prob_mets['targets']
        res[sset] = {
            **res[sset],
            **prob_mets
        }

        # per class AUROC
        if preds.squeeze().ndim == 1:  # 2 classes
            res[sset][f'class_1_AUROC'] = roc_auc_score(Y, preds, labels=[0, 1])
            res[sset][f'class_0_AUROC'] = res[sset][f'class_1_AUROC']
        else:
            for y in np.unique(Y):
                new_label = Y == y
                new_preds = preds[:, int(y)]
                res[sset][f'class_{y}_AUROC'] = roc_auc_score(new_label, new_preds, labels=[0, 1])

    return res


def eval_lin_attr_pred(train_zs, train_atts, train_ys, val_zs, val_atts, val_ys, test_zs, test_atts, test_ys):
    res = {}
    for model_type in ['lr', 'rf']:
        res[f'{model_type}_uncond'] = fit_model(
            train_zs, train_atts, val_zs, val_atts, test_zs, test_atts, model_type=model_type)

        res[f'{model_type}_cond_0'] = fit_model(
            train_zs[train_ys == 0], train_atts[train_ys == 0], val_zs[val_ys == 0], val_atts[val_ys == 0],
            test_zs[test_ys == 0], test_atts[test_ys == 0], model_type=model_type)

        res[f'{model_type}_cond_1'] = fit_model(
            train_zs[train_ys == 1], train_atts[train_ys == 1], val_zs[val_ys == 1], val_atts[val_ys == 1],
            test_zs[test_ys == 1], test_atts[test_ys == 1], model_type=model_type)

        cond_avg = {'va': {}, 'te': {}}
        for sset in res[f'{model_type}_cond_0']:
            for met in res[f'{model_type}_cond_0'][sset]:
                if isinstance(res[f'{model_type}_cond_0'][sset][met], (float, np.floating, int)):
                    cond_avg[sset][met] = (res[f'{model_type}_cond_0'][sset][met] + res[f'{model_type}_cond_1'][sset][met])/2

        res[f'{model_type}_cond_avg'] = cond_avg
    return res
