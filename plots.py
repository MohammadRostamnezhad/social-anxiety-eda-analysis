import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import math
from sklearn.linear_model import LogisticRegression


def draw_plot(plot_data, plot_bins=30, plt_type="Histogram", kde_plot=False,
              plot_label=None, x_label=None, y_label=None, ax=None,
              target=None, target_col="Target"):

    plot_data = np.asarray(plot_data)
    if target is not None:
        target = np.asarray(target)

    if ax is None:
        fig, ax = plt.subplots()
        own_fig = True
    else:
        own_fig = False

    plt_type = plt_type.lower()
    histnorm = "probability density" if kde_plot else None

    if plt_type == "histogram":
        if target is None:
            ax.hist(plot_data, bins=plot_bins,
                    density=(histnorm == "probability density"),
                    label="Histogram", alpha=0.75)
        else:
            for t in np.unique(target):
                mask = target == t
                ax.hist(plot_data[mask], bins=plot_bins,
                        density=(histnorm == "probability density"),
                        alpha=0.5, label=f"{target_col}={t}")

    elif plt_type == "scatter":
        ax.scatter(np.arange(plot_data.size), plot_data,
                   label="Scatter", color="steelblue", s=6, alpha=0.7)

    elif plt_type == "line":
        ax.plot(np.arange(plot_data.size), plot_data,
                label="Line", color="steelblue", linewidth=2)

    elif plt_type == "boxplot":
        if target is None:
            ax.boxplot(plot_data)
        else:
            groups = [plot_data[target == t] for t in np.unique(target)]
            labels = [f"{target_col}={t}" for t in np.unique(target)]
            ax.boxplot(groups, labels=labels) 

    elif plt_type == "bar":
        if target is None:
            values, counts = np.unique(plot_data, return_counts=True)
            ax.bar(np.arange(len(values)), counts,
                   label="Bar", color="steelblue", alpha=0.75)
            ax.set_xticks(np.arange(len(values)))
            ax.set_xticklabels(values, rotation=45, ha="right")
        else:
            values = np.unique(plot_data)
            unique_targets = np.unique(target)
            n_t = len(unique_targets)
            width = 0.8 / n_t
            x = np.arange(len(values))
            for k, t in enumerate(unique_targets):
                counts = [np.sum((plot_data == v) & (target == t))
                          for v in values]
                ax.bar(x + (k - (n_t - 1) / 2) * width, counts,
                       width=width, alpha=0.85,
                       label=f"{target_col}={t}")
            ax.set_xticks(x)
            ax.set_xticklabels(values, rotation=45, ha="right")

    if kde_plot and plt_type in ("histogram", "line", "scatter"):
        try:
            numeric_data = plot_data.astype(float)
        except (ValueError, TypeError):
            numeric_data = None

        if numeric_data is not None and numeric_data.size > 1:
            pad = 0.05 * (numeric_data.max() - numeric_data.min())
            x_range = np.linspace(numeric_data.min() - pad,
                                  numeric_data.max() + pad, 300)

            if target is None:
                kde = stats.gaussian_kde(numeric_data)
                ax.plot(x_range, kde(x_range), label="KDE",
                        color="red", linewidth=2)
            else:
                for t in np.unique(target):
                    mask = target == t
                    d = numeric_data[mask]
                    if d.size < 2:
                        continue
                    kde = stats.gaussian_kde(d)
                    ax.plot(x_range, kde(x_range),
                            label=f"KDE {target_col}={t}", linewidth=2)

    ax.set_title(plot_label, fontsize=9)
    if plt_type != "violin":
        ax.set_xlabel(x_label, fontsize=8)
        
    ax.set_ylabel(y_label if y_label else ("Density" if kde_plot else "Count"),
                  fontsize=8)
    ax.tick_params(labelsize=7)

    if target is not None:
        ax.legend(fontsize=7, title=target_col, title_fontsize=7)

    if own_fig:
        plt.show()

    return ax


def draw_plots_grid(df, cols, target_col=None, n_cols=6, figsize=(16, 8),
                    plot_bins=20, plt_type=None, kde_plot=None):
    n = len(cols)
    n_rows = -(-n // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=figsize,
                             tight_layout=True, squeeze=False)
    axes_flat = axes.flatten()

    for i, col in enumerate(cols):
        wanted = [col] + ([target_col] if target_col else [])
        sub = df[wanted].dropna()
        s = sub[col]
        t = sub[target_col].values if target_col else None

        is_cat = (
            s.dtype == object
            or str(s.dtype) in ("category", "bool")
            or s.nunique() <= 10
        )
        local_type = plt_type if plt_type is not None else (
            "bar" if is_cat else "histogram")
        local_kde = kde_plot if kde_plot is not None else (not is_cat)

        draw_plot(
            s.values,
            plot_bins=plot_bins,
            plt_type=local_type,
            kde_plot=local_kde,
            plot_label=str(col),
            x_label=None,
            y_label=None,               
            ax=axes_flat[i],
            target=t,
            target_col=target_col or "Target",
        )

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.show()


def draw_scatter_pairs_grid_mpl(df, cols, target_col, n_cols=4,
                                figsize_per_cell=(3.2, 2.8),
                                marker_size=6, alpha=0.5):
    pairs = [(cols[i], cols[j])
             for i in range(len(cols))
             for j in range(i + 1, len(cols))]

    n = len(pairs)
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_cell[0] * n_cols,
                 figsize_per_cell[1] * n_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for k, (x_col, y_col) in enumerate(pairs):
        ax = axes_flat[k]
        for cls, g in df.groupby(target_col):
            ax.scatter(g[x_col], g[y_col],
                       s=marker_size, alpha=alpha, label=str(cls))
        ax.set_title(f"{x_col} vs {y_col}", fontsize=8)
        ax.set_xlabel(x_col, fontsize=7)
        ax.set_ylabel(y_col, fontsize=7)
        ax.tick_params(labelsize=6)
        if k == 0:
            ax.legend(title=target_col, fontsize=6, title_fontsize=6)

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.tight_layout()
    plt.show()


def draw_scatter_3d(df, x_col, y_col, z_col, target_col='Target',
                    title=None, sample=5000, opacity=0.7, size=4):
    d = df.sample(sample, random_state=0) if len(df) > sample else df

    fig = px.scatter_3d(
        d, x=x_col, y=y_col, z=z_col,
        color=d[target_col].astype(str),
        opacity=opacity,
        title=title or f"{x_col} vs {y_col} vs {z_col} by {target_col}",
        labels={target_col: target_col},
    )
    fig.update_traces(marker=dict(size=size))
    fig.update_layout(
        legend_title_text=target_col,
        scene=dict(xaxis_title=x_col, yaxis_title=y_col, zaxis_title=z_col),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    fig.show()


def draw_logistic_curves_grid(df, combinations_list, target_col="Target", 
                              n_cols=3, figsize_per_cell=(6, 4)):
    
    n = len(combinations_list)
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows),
        squeeze=False,
        tight_layout=True
    )
    axes_flat = axes.flatten()

    log_model = LogisticRegression(solver='lbfgs', max_iter=500)

    for i, features in enumerate(combinations_list):
        ax = axes_flat[i]
        
        wanted = features + [target_col]
        temp_df = df[wanted].dropna()
        y = temp_df[target_col]
        
        if len(features) == 1:
            x_plot = temp_df[features[0]]
            x_label = features[0]
            title_prefix = "1-Column"
        else:
            X_features = temp_df[features]
            X_features = pd.get_dummies(X_features, drop_first=True)
            
            log_model.fit(X_features, y)
            x_plot = log_model.predict_proba(X_features)[:, 1]
            x_label = "Combined Probability Score"
            title_prefix = f"{len(features)}-Column"

        sns.regplot(
            x=x_plot, 
            y=y, 
            logistic=True, 
            n_boot=0,   
            ci=None,    
            scatter_kws={'alpha': 0.4, 'color': 'steelblue', 's': 15}, 
            line_kws={'color': 'darkred', 'linewidth': 2.5},
            ax=ax
        )
        

        feature_name = " +\n".join(features)
        ax.set_title(f"Rank {i+1}: {title_prefix}\n{feature_name}", fontsize=9, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=8)
        ax.set_ylabel(f"Prob({target_col}=1)", fontsize=8)
        ax.set_yticks([0, 1])
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)

   
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.show()


