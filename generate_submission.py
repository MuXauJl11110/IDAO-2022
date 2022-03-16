from pathlib import Path

import catboost as cat
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from baseline import load_data, make_submission, read_pymatgen_dict

threshold = 0.02


def error(output, target, criterion):
    loss = criterion(output, target)
    correct = (np.abs(output - target) < (np.ones_like(output) * threshold)).sum()

    return correct / len(output), loss


def use_model(model, criterion, X_tr, y_tr, kf, X_t, frame, name):
    kf.get_n_splits(X_tr)
    train_losses, train_acc, val_losses, val_acc = [], [], [], []
    for i, (train_index, val_index) in enumerate(kf.split(X_tr)):
        Xtrain, Xval = X_tr[train_index], X_tr[val_index]
        ytrain, yval = y_tr[train_index], y_tr[val_index]

        model.fit(Xtrain, ytrain)

        train_pred = model.predict(Xtrain)
        val_pred = model.predict(Xval)

        t_acc, t_loss = error(train_pred, ytrain, criterion)
        train_losses.append(t_loss)
        train_acc.append(t_acc)

        v_acc, v_loss = error(val_pred, yval, criterion)
        val_losses.append(v_loss)
        val_acc.append(v_acc)
        print(
            "Fold %d: train acc: %.4f train loss: %.4f val acc: %.4f val loss: %.4f"
            % (i + 1, t_acc, t_loss, v_acc, v_loss)
        )

    t_acc, t_loss = np.mean(train_acc), np.mean(train_losses)
    v_acc, v_loss = np.mean(val_acc), np.mean(val_losses)
    print(
        "Mean over folds: train acc: %.4f train loss: %.4f val acc: %.4f val loss: %.4f"
        % (t_acc, t_loss, v_acc, v_loss)
    )
    model.fit(X_tr, y_tr)
    res = model.predict(X_t)
    return {"train_acc": t_acc, "train_loss": t_loss, "val_acc": v_acc, "val_loss": v_loss}, make_submission(
        res, frame, name + ".csv"
    )


def make_submission(res, frame, name):
    f1 = frame.assign(predictions=res)
    f1["predictions"].to_csv(name, index_label="id")

    return f1


def prepare_data(dataset_path, with_targets=False):
    dataset_path = Path(dataset_path)
    if with_targets:
        targets = pd.read_csv(dataset_path / "targets.csv", index_col=0)
    struct = {item.name.strip(".json"): read_pymatgen_dict(item) for item in (dataset_path / "structures").iterdir()}

    data = pd.DataFrame(columns=["structures"], index=struct.keys())
    data = data.assign(structures=struct.values())
    if with_targets:
        data = data.assign(targets=targets)

    return data


def extract_full_data1(data, target=False, max_len=0, formula_to_num={}, max_n_bonds=0):
    it1 = []
    it2 = []
    it3 = []
    it4 = []
    it5 = []
    targets = []
    num_to_element = {i + 1: v for i, v in enumerate(e.keys())}

    for i in data.iterrows():
        s = i[1].structures
        lat = s.lattice
        c = g(s)

        it1.append(
            np.array(
                [
                    formula_to_num[s.formula],
                    len(s),
                    len(c["index1"]),
                    lat.a,
                    lat.b,
                    lat.c,
                    *lat.abc,
                    *lat.angles,
                    lat.volume,
                ]
            )[np.newaxis, ...]
        )
        it2l = np.zeros((max_len, 16))
        it3l = np.zeros((max_len, 1))
        it4l = np.zeros((max_len, 6))
        it5l = np.zeros((max_n_bonds, 3))
        for j in range(len(s)):
            it2l[j, :] = e[num_to_element[c["atom"][j]]]
            it3l[j, :] = c["atom"][j]
            it4l[j, :] = np.array([s.sites[j].a, s.sites[j].b, s.sites[j].c, s.sites[j].x, s.sites[j].y, s.sites[j].z])

        for j in range(len(c["index1"])):
            it5l[j, :] = np.array([c["index1"][j], c["index2"][j], c["bond"][j]])

        it2.append(it2l[np.newaxis, ...])
        it3.append(it3l[np.newaxis, ...])
        it4.append(it4l[np.newaxis, ...])
        it5.append(it5l[np.newaxis, ...])

        if target == True:
            t = i[1].targets
            targets.append(t)

    targets = np.array(targets)

    it1 = np.concatenate(it1)
    it2 = np.concatenate(it2)
    it3 = np.concatenate(it3)
    it4 = np.concatenate(it4)
    it5 = np.concatenate(it5)

    if target == True:
        return it1, it2, it3, it4, it5, targets
    else:
        return it1, it2, it3, it4, it5


def main(config):
    """
    print("=> start loading training embeddings")
    X_train, y_train, _ = load_data(config["embeddings"]["train_name"], test=False)
    print("=> end loading training embdeddings")

    print("=> start loading testing embeddings")
    X_test, ids_test = load_data(config["embeddings"]["test_name"], test=True)
    print("=> end loading testing embdeddings")

    test_data = pd.DataFrame({"id": ids_test})
    models = {
        "catboost_mae": cat.CatBoostRegressor(verbose=0, loss_function="MAE"),
        "catboost_rmse": cat.CatBoostRegressor(verbose=0),
        "xgboost": xgb.XGBRegressor(),
        "lightgbm": LGBMRegressor(),
        "random_forest": RandomForestRegressor(),
        "knn": KNeighborsRegressor(),
        "dec_tree_mae": DecisionTreeRegressor(criterion="absolute_error"),
        "dec_tree": DecisionTreeRegressor(),
        "linreg": LinearRegression(),
    }

    first_lavel_est = [(i, models[i]) for i in models]

    stacking = StackingRegressor(estimators=first_lavel_est, n_jobs=2, final_estimator=RandomForestRegressor())
    stacking.fit(X_train, y_train)
    res = stacking.predict(X_test)
    make_submission(res, test_data, "submission.csv")
    """
    train_data = prepare_data(config["datapath"], with_targets=True)
    test_data = prepare_data(config["test_datapath"])
    X_train_full = extract_full_data1(train_data, target=True)
    X_test_full = extract_full_data1(test_data)

    s2 = StandardScaler()
    s2.fit(np.concatenate([X_train_full[0], X_test_full[0]]))
    y_train = X_train_full[-1]
    X_tmp1 = X_train_full[0]
    X_train2 = X_tmp1
    X_tmp1 = s2.transform(X_tmp1)
    X_train2_scaled = X_tmp1
    X_tmp1 = X_test_full[0]
    X_test2 = X_tmp1
    X_tmp1 = s2.transform(X_tmp1)
    X_test2_scaled = X_tmp1

    kf = KFold(n_splits=5)

    res, frame = use_model(
        model=cat.CatBoostRegressor(verbose=0, loss_function="MAE"),
        criterion=mean_absolute_error,
        X_tr=X_train2,
        y_tr=y_train,
        kf=kf,
        X_t=X_test2,
        frame=test_data,
        name="submission",
    )


if __name__ == "__main__":
    with open("config.yaml") as file:
        config = yaml.safe_load(file)
    main(config)
