# Imports
from os.path import join
import pickle
from pennylane import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from glob import glob
import seaborn as sns
import pandas as pd

from qmlhep.utils.helper import GridSearch
from qmlhep.config import results_path
from qmlhep.config import analisys_results_path, figures_path
from qmlhep.config import qc_results_path

import warnings

warnings.filterwarnings("ignore")

# Set whitegrid
sns.set(style="whitegrid")

#################################################################
############ Plotting Configurations
#################################################################

SMALL_SIZE = 24
MEDIUM_SIZE = 32
BIGGER_SIZE = 38
LEGEND_SIZE = 30
TICK_SIZE = 30
sns.set(font_scale=1000)


plt.rc("font", size=SMALL_SIZE)         # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)    # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)   # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)   # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)   # fontsize of the tick labels
plt.rc("legend", fontsize=LEGEND_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE) # fontsize of the figure title


#################################################################
############ QML Clf for seamless integration
#################################################################


class qml_clf:
    def __init__(self, trainer, weights):
        self.weights = weights
        self.trainer = trainer

    def __str__(self):
        return f"QML Classifier"

    def predict_proba(self, X):
        prob = np.array([self.trainer.classifier(self.weights, x) for x in X])
        prob = (prob + 1) / 2
        return np.array([[1 - p, p] for p in prob])

    def predict(self, X):
        prob = self.predict_proba(X)
        return np.array([1 if p[1] > 0.5 else 0 for p in prob])


#################################################################
############ Painter main class
#################################################################


class Painter:
    def __init__(self, use_test_data=False) -> None:
        # Create dataframe for sns
        self.df = pd.DataFrame()

        # Stuff tp alternate between test and validation
        if use_test_data:
            self.y_scores_key = "Y_test_scores"
            self.x_data_key = "X_test"
            self.y_data_key = "Y_test"
            self.w_data_key = "W_test"
            self.save_path_extra = "_test"
        else:
            self.y_scores_key = "Y_val_scores"
            self.x_data_key = "X_val"
            self.y_data_key = "Y_val"
            self.w_data_key = "W_val"
            self.save_path_extra = "_val"

        self.use_test_data = use_test_data

        for study in ["AdamModel", "OptunaModel"]:
            results = glob(join(analisys_results_path, study, f"*.pkl"))

            for result in results:
                with open(result, "rb") as f:
                    qml_worlds, svm_worlds, log_worlds = pickle.load(f)

                for i, qml_world in enumerate(qml_worlds):
                    hp = qml_world
                    hp["model"] = "QML"
                    hp["run"] = i
                    hp["name"] = int(hp["name"].split("_")[0])
                    hp["auc"] = roc_auc_score(hp[self.y_data_key], hp[self.y_scores_key], sample_weight=hp[self.w_data_key])

                    self.df = pd.concat([self.df, pd.DataFrame.from_dict(hp, orient="index").T])

                for i, svm_world in enumerate(svm_worlds):
                    hp = svm_world
                    hp["model"] = "SVM"
                    hp["run"] = i
                    hp["name"] = int(hp["name"].split("_")[0])
                    hp["auc"] = roc_auc_score(hp[self.y_data_key], hp[self.y_scores_key], sample_weight=hp[self.w_data_key])

                    self.df = pd.concat([self.df, pd.DataFrame.from_dict(hp, orient="index").T])

                for i, log_world in enumerate(log_worlds):
                    hp = log_world
                    hp["model"] = "LR"
                    hp["run"] = i
                    hp["name"] = int(hp["name"].split("_")[0])
                    hp["auc"] = roc_auc_score(hp[self.y_data_key], hp[self.y_scores_key], sample_weight=hp[self.w_data_key])

                    self.df = pd.concat([self.df, pd.DataFrame.from_dict(hp, orient="index").T])

        self.df.reset_index(inplace=True, drop=True)

        # Infer
        self.df = self.df.infer_objects()

        # Rename columns
        self.df.rename(
            columns={
                "study_name": "Study",
                "feature_method": "Feature Method",
                "n_datapoints": "#Datapoints",
                "auc": "AUC",
                "n_features": "#Features",
                "n_layers": "#Layers",
                "model": "Model",
            },
            inplace=True,
        )

    def plot_all_studys(self):
        fig, ax = plt.subplots(4, 4, figsize=(28, 28), sharey=True)

        palette = {"SBS": "C0", "PCA": "C1"}
        
        for n_features in range(4):
            for n_layers in range(4):

                sns.lineplot(
                    x="#Datapoints",
                    y="AUC",
                    hue="Feature Method",
                    style="Study",
                    data=self.df[self.df["#Features"] == n_features + 1][self.df["#Layers"] == n_layers + 1][
                        self.df["Model"] == "QML"
                    ], 
                    markers=True,
                    dashes=True,
                    markersize=8,
                    ax=ax[n_features, n_layers],
                    palette=palette,
                    legend=True,
                )

                ax[n_features, n_layers].set_title(f"{n_features+1} features, {n_layers+1} layers", fontsize=BIGGER_SIZE)
                ax[n_features, n_layers].set_xlabel("#Datapoints", fontsize=MEDIUM_SIZE)
                ax[n_features, n_layers].set_ylabel("AUC Score", fontsize=MEDIUM_SIZE)
                ax[n_features, n_layers].legend(fontsize=LEGEND_SIZE)
                ax[n_features, n_layers].tick_params(axis="both", which="major", labelsize=TICK_SIZE)

        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes[:1]]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels, loc="lower center", ncol=2, fontsize=LEGEND_SIZE, bbox_to_anchor=(0.5, -0.08))

        # Remove all legends
        for ax in fig.axes:
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        fig.tight_layout()

        # Save fig pdf
        fig.savefig(join(figures_path, f"adam-vs-optuna{self.save_path_extra}.pdf"), bbox_inches='tight')

    def plot_study(self, study):
        fig, ax = plt.subplots(4, 4, figsize=(28, 28), sharey=True)

        palette = {"SBS": "C0", "PCA": "C1"}
        
        for n_features in range(4):
            for n_layers in range(4):
                sns.lineplot(
                    x="#Datapoints",
                    y="AUC",
                    hue="Feature Method",
                    data=self.df[self.df["Study"] == study][self.df["#Features"] == n_features + 1][self.df["#Layers"] == n_layers + 1][
                        self.df["Model"] == "QML"
                    ],
                    markers=True,
                    dashes=True,
                    markersize=8,
                    ax=ax[n_features, n_layers],
                    palette=palette,
                    ci=0,
                    linewidth=3,
                )

                sns.lineplot(
                    x="#Datapoints",
                    y="AUC",
                    hue="Feature Method",
                    style="run",
                    data=self.df[self.df["Study"] == study][self.df["#Features"] == n_features + 1][self.df["#Layers"] == n_layers + 1][
                        self.df["Model"] == "QML"
                    ],
                    markers=True,
                    dashes=True,
                    markersize=8,
                    ax=ax[n_features, n_layers],
                    palette=palette,
                    ci=0,
                    alpha=0.25,
                    legend=False,
                    linewidth=2,
                )

                ax[n_features, n_layers].set_title(f"{n_features+1} features, {n_layers+1} layers", fontsize=BIGGER_SIZE)
                ax[n_features, n_layers].set_xlabel("#Datapoints", fontsize=MEDIUM_SIZE)
                ax[n_features, n_layers].set_ylabel("AUC Score", fontsize=MEDIUM_SIZE)
                ax[n_features, n_layers].legend(fontsize=LEGEND_SIZE)
                ax[n_features, n_layers].tick_params(axis="both", which="major", labelsize=TICK_SIZE)

        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes[:1]]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels, loc="lower center", ncol=2, fontsize=LEGEND_SIZE, bbox_to_anchor=(0.5, -0.04))

        # Remove all legends
        for ax in fig.axes:
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        fig.tight_layout()

        # Save fig 
        fig.savefig(join(figures_path, f"{study}{self.save_path_extra}.pdf"), bbox_inches='tight')

    def plot_shallow(self, study):
        # Shallow
        fig, ax = plt.subplots(2, 2, figsize=(15, 15), sharey=True)

        palette = {"SBS": "C0", "PCA": "C1"}

        for n_features in range(4):

            sns.lineplot(
                x="#Datapoints",
                y="AUC",
                hue="Feature Method",
                style="Model",
                data=self.df[self.df["Study"] == study][self.df["#Features"] == n_features + 1][self.df["Model"] != "QML"],
                markers=True,
                dashes=True,
                markersize=8,
                ax=ax[n_features // 2, n_features % 2],
                palette=palette,
            ) 

            ax[n_features // 2, n_features % 2].set_title(f"{n_features+1} features", fontsize=BIGGER_SIZE)
            ax[n_features // 2, n_features % 2].set_xlabel("#Datapoints", fontsize=MEDIUM_SIZE)
            ax[n_features // 2, n_features % 2].set_ylabel("AUC Score", fontsize=MEDIUM_SIZE)
            ax[n_features // 2, n_features % 2].legend(fontsize=LEGEND_SIZE)
            ax[n_features // 2, n_features % 2].tick_params(axis="both", which="major", labelsize=TICK_SIZE)

        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes[:1]]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels, loc="lower center", ncol=2, fontsize=LEGEND_SIZE, bbox_to_anchor=(0.5, -0.15))

        # Remove all legends
        for ax in fig.axes:
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        fig.tight_layout()

        # Save fig 
        fig.savefig(join(figures_path, f"shallow-{study}{self.save_path_extra}.pdf"), bbox_inches='tight')

    def plot_best_roc(self, study):
        # Get the number trial of the best model
        best_name, best_run, (qml_world, svm_world, log_world) = self.get_best_name(study)

        fig = plt.figure(figsize=(15, 15))

        gs = gridspec.GridSpec(4, 4)
        ax_qml_roc = plt.subplot(gs[:2, :2])
        ax_svm_roc = plt.subplot(gs[:2, 2:])
        ax_log_roc = plt.subplot(gs[2:4, 1:3])

        # Plot Rocs for QML runs
        aucs_qml = []
        aucs_svm = []
        aucs_log = []
        for idx in range(len(qml_world)):
            qml_run = qml_world[idx]
            svm_run = svm_world[idx]
            log_run = log_world[idx]

            qml_auc = roc_auc_score(y_true=qml_run[self.y_data_key], y_score=qml_run[self.y_scores_key], sample_weight=qml_run[self.w_data_key])
            qml_fpr, qml_tpr, _ = roc_curve(
                y_true=qml_run[self.y_data_key], y_score=qml_run[self.y_scores_key], sample_weight=qml_run[self.w_data_key]
            )

            svm_auc = roc_auc_score(y_true=svm_run[self.y_data_key], y_score=svm_run[self.y_scores_key], sample_weight=svm_run[self.w_data_key])
            svm_fpr, svm_tpr, _ = roc_curve(
                y_true=svm_run[self.y_data_key], y_score=svm_run[self.y_scores_key], sample_weight=svm_run[self.w_data_key]
            )

            log_auc = roc_auc_score(y_true=log_run[self.y_data_key], y_score=log_run[self.y_scores_key], sample_weight=log_run[self.w_data_key])
            log_fpr, log_tpr, _ = roc_curve(
                y_true=log_run[self.y_data_key], y_score=log_run[self.y_scores_key], sample_weight=log_run[self.w_data_key]
            )

            # Add to list
            aucs_qml.append(qml_auc)
            aucs_svm.append(svm_auc)
            aucs_log.append(log_auc)

            # Plot ROC for diferent instances
            ax_qml_roc.plot(qml_fpr, qml_tpr, alpha=0.5)
            ax_svm_roc.plot(svm_fpr, svm_tpr, alpha=0.5)
            ax_log_roc.plot(log_fpr, log_tpr, alpha=0.5)

        # Plot random classifier lines
        ax_qml_roc.plot([0, 1], [0, 1], linestyle="--")
        ax_svm_roc.plot([0, 1], [0, 1], linestyle="--")

        # Titles
        ax_qml_roc.set_title(
            f"QML\nAUC: {np.mean(aucs_qml):.3f} +/- {np.std(aucs_qml):.3f}",
            fontsize=MEDIUM_SIZE,
        )
        ax_svm_roc.set_title(
            f"SVM\nAUC: {np.mean(aucs_svm):.3f} +/- {np.std(aucs_svm):.3f}",
            fontsize=MEDIUM_SIZE,
        )
        ax_log_roc.set_title(
            f"Log. Reg.\nAUC: {np.mean(aucs_log):.3f} +/- {np.std(aucs_log):.3f}",
            fontsize=MEDIUM_SIZE,
        )

        # Font size for axes
        ax_qml_roc.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
        ax_svm_roc.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
        ax_log_roc.tick_params(axis="both", which="major", labelsize=TICK_SIZE)

        # Labels
        ax_qml_roc.set_xlabel("False Positive Rate", fontsize=MEDIUM_SIZE)
        ax_qml_roc.set_ylabel("True Positive Rate", fontsize=MEDIUM_SIZE)
        ax_svm_roc.set_xlabel("False Positive Rate", fontsize=MEDIUM_SIZE)
        ax_svm_roc.set_ylabel("True Positive Rate", fontsize=MEDIUM_SIZE)
        ax_log_roc.set_xlabel("False Positive Rate", fontsize=MEDIUM_SIZE)
        ax_log_roc.set_ylabel("True Positive Rate", fontsize=MEDIUM_SIZE)

        # Tight layout
        plt.tight_layout()

        # Save figure
        fig.savefig(join(figures_path, f"{study}-best-roc{self.save_path_extra}.pdf"), bbox_inches='tight')

    def plot_histogram_separation(self, study):
        best_name, best_run, (qml_world, svm_world, log_world) = self.get_best_name(study)

        qml_world = qml_world[best_run]

        X_val = qml_world[self.x_data_key]
        Y_val = qml_world[self.y_data_key]
        W_val = qml_world[self.w_data_key]
        y_scores = qml_world[self.y_scores_key]

        fig, ax = plt.subplots(sharex=True, figsize=(20, 10))
        auc = roc_auc_score(Y_val, y_scores, sample_weight=W_val)
        ax.hist(y_scores[Y_val == 0], histtype="step", fill=False, label="Background", alpha=0.9, linewidth=1.6)
        ax.hist(y_scores[Y_val == 1], histtype="step", fill=False, label="Signal", alpha=0.9, linewidth=1.6)
        plt.title(f"AUC: {auc:.4f}", fontsize=BIGGER_SIZE)
        plt.yscale("log")
        plt.legend(fontsize=MEDIUM_SIZE)
        # More fonts
        ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)

        plt.xlabel("Model Prediction", fontsize=MEDIUM_SIZE)
        plt.ylabel("#Events", fontsize=MEDIUM_SIZE)

        fig.savefig(join(figures_path, f"{study}-histogram_separation{self.save_path_extra}.pdf"), bbox_inches='tight')
        plt.show()

    def plot_best_score_epoch(self, study):
        from qmlhep.config import hp_space

        df = pd.DataFrame()
        path = join(results_path, study, "models")
        for hp in GridSearch(hp_space):
            name = hp["name"]
            
            for file in glob(join(path, f"{name}_*info.pkl")):
                with open(file, "rb") as f:
                    results = pickle.load(f)

                hp["best_score_epoch"] = results["best_score_epoch"]

                # Concatenate results
                df = pd.concat([df, pd.DataFrame(hp, index=[0])])

        df.rename(
            columns={
                "feature_method": "Feature Method",
                "n_datapoints": "#Datapoints",
                "n_features": "#Features",
                "n_layers": "#Layers",
            },
            inplace=True,
        )

        df = pd.DataFrame()
        path = join(results_path, study, "models")
        for hp in GridSearch(hp_space):
            name = hp["name"]
            
            for file in glob(join(path, f"{name}_*info.pkl")):
                with open(file, "rb") as f:
                    results = pickle.load(f)

                hp["best_score_epoch"] = results["best_score_epoch"]

                # Concatenate results
                df = pd.concat([df, pd.DataFrame(hp, index=[0])])

        df.rename(
            columns={
                "feature_method": "Feature Method",
                "n_datapoints": "#Datapoints",
                "n_features": "#Features",
                "n_layers": "#Layers",
            },
            inplace=True,
        )

        fig, ax = plt.subplots(5, 5, figsize=(28, 28), sharey=True)

        palette = {"SBS": "C0", "PCA": "C1"}

        for n_features in range(5):
            for n_layers in range(5):
                df_ = df[(df["#Features"] == n_features + 1) & (df["#Layers"] == n_layers + 1)]
                df_.reset_index(inplace=True, drop=True)

                sns.lineplot(
                    x="#Datapoints",
                    y="best_score_epoch",
                    hue="Feature Method",
                    data=df_,
                    markers=True,
                    dashes=True,
                    markersize=8,
                    ax=ax[n_features, n_layers],
                    palette=palette,
                )

                ax[n_features, n_layers].set_title(f"{n_features+1} features, {n_layers+1} layers")
                ax[n_features, n_layers].set_xlabel("#Datapoints")
                ax[n_features, n_layers].set_ylabel("Best Score Epoch")
                ax[n_features, n_layers].legend()
                ax[n_features, n_layers].tick_params(axis="both", which="major")

        fig.tight_layout()

        fig.savefig(join(figures_path, f"{study}-best_score_epoch{self.save_path_extra}.pdf"), bbox_inches='tight')

    def plot_qc(self, model):
        best_name, best_run, (qml_worlds, svm_worlds, log_worlds) = self.get_best_name(model)

        sufix = "test" if self.use_test_data else "val"

        systems = set([x.split("_")[-4] for x in glob(join(qc_results_path, f"*{sufix}.pkl"))])
        seeds = [qml_world["random_seed"] for qml_world in qml_worlds]

        fig = plt.figure(figsize=(15, 15 * 3 / 2))

        for i, system in enumerate(systems):
            runs = [x for x in glob(join(qc_results_path, f"*{sufix}.pkl")) if system in x]
            print(f"System {system} has {len(runs)} runs.")

            ax = fig.add_subplot(int(f"32{i+1}"))
            aucs = []
            for i, random_seed in enumerate(seeds):
                # This needs to be improved
                # This is to guarantee that the same run has the same
                # color on figerent subplots
                for run in runs:
                    run_name = run.split("_")[-3]
                    run_random_seed = run.split("_")[-2]

                    if int(run_random_seed) == int(random_seed):
                        break

                for qml_world in qml_worlds:
                    if int(qml_world["random_seed"]) == int(random_seed):
                        break

                assert qml_world["name"].split("_")[0] == run_name, f"Name {qml_world['name'].split('_')[0]} does not match {run_name}"

                X = qml_world[f"X_{sufix}"]
                Y = qml_world[f"Y_{sufix}"]
                W = qml_world[f"W_{sufix}"]

                with open(run, "rb") as f:
                    results = pickle.load(f)

                # Plot ROC
                fpr, tpr, _ = roc_curve(Y, results, sample_weight=W)
                auc = roc_auc_score(Y, results, sample_weight=W)
                aucs.append(auc)
                ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
                ax.plot([0, 1], [0, 1], linestyle="--", color="grey")

            # Style
            aucs = np.array(aucs)

            # Labels
            ax.set_xlabel("False Positive Rate", fontsize=MEDIUM_SIZE)
            ax.set_ylabel("True Positive Rate", fontsize=MEDIUM_SIZE)
            ax.set_title(f"IBM's {system[0].upper() + system[1:]} system\nAUC: {aucs.mean():.3f} +- {aucs.std():.3f}", fontsize=MEDIUM_SIZE)
            ax.tick_params(axis="both", which="major", labelsize=MEDIUM_SIZE)

        plt.tight_layout()

        plt.savefig(join(figures_path, f"best_roc_real_qc_{model}_{sufix}.pdf"), bbox_inches='tight')

    def plot_all_from_study(self, study):
        self.plot_study(study)
        self.plot_shallow(study)
        self.plot_best_roc(study)
        self.plot_histogram_separation(study)

    def get_best_name(self, study):
        best_name = self.df[self.df["Study"] == study][self.df["Model"] == "QML"].groupby(["name"])["AUC"].mean().idxmax()
        best_run = self.df[self.df["Study"] == study][self.df["Model"] == "QML"][self.df["name"] == best_name].groupby(["run"])["AUC"].mean().idxmax()

        with open(join(analisys_results_path, study, f"{best_name}.pkl"), "rb") as f:
            qml_world, svm_world, log_world = pickle.load(f)

        return best_name, best_run, (qml_world, svm_world, log_world)
