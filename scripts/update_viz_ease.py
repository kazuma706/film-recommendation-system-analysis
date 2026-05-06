"""
Regenerate ndcg_sparsity_curves.html and model_dashboard.html with EASE added.
Run from the project root:
    python scripts/update_viz_ease.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

OUTPUT_DIR = 'outputs/'

# 
# 1.  NDCG@10 vs Sparsity Curves
# 

def build_sparsity_curves():
    # load existing model results
    df_a = pd.read_csv(OUTPUT_DIR + 'sparsity_expA_all_models.csv')
    df_b = pd.read_csv(OUTPUT_DIR + 'sparsity_expB_all_models.csv')

    # normalise Exp B: drop 'experiment' column if present
    if 'experiment' in df_b.columns:
        df_b = df_b.drop(columns=['experiment'])

    # load EASE results
    ease_a = pd.read_csv(OUTPUT_DIR + 'ease_expA_results.csv')
    ease_b = pd.read_csv(OUTPUT_DIR + 'ease_expB_results.csv')
    ease_a['model'] = 'ease'
    ease_b['model'] = 'ease'

    # merge EASE into combined DataFrames
    df_a = pd.concat([df_a, ease_a], ignore_index=True)
    df_b = pd.concat([df_b, ease_b], ignore_index=True)

    for df in [df_a, df_b]:
        df['log_n'] = np.log10(df['n_ratings'])

    print('Exp A shape (with EASE):', df_a.shape)
    print('Exp B shape (with EASE):', df_b.shape)
    print('Models in A:', sorted(df_a['model'].unique()))
    print('Models in B:', sorted(df_b['model'].unique()))

    # style config
    MODEL_ORDER = ['popularity_baseline', 'svd', 'penalized_svd',
                   'content_based', 'hybrid', 'ease']
    MODEL_LABELS = {
        'popularity_baseline': 'Popularity Baseline',
        'svd':                 'SVD',
        'penalized_svd':       'Penalised SVD (λ=0.3)',
        'content_based':       'Content-Based',
        'hybrid':              'Hybrid (α=0.6)',
        'ease':                'EASE (λ=500)',
    }
    COLOURS = {
        'popularity_baseline': '#636EFA',
        'svd':                 '#EF553B',
        'penalized_svd':       '#00CC96',
        'content_based':       '#AB63FA',
        'hybrid':              '#FFA15A',
        'ease':                '#00BCD4',
    }
    DASH = {
        'popularity_baseline': 'solid',
        'svd':                 'dot',
        'penalized_svd':       'dash',
        'content_based':       'dashdot',
        'hybrid':              'longdash',
        'ease':                'longdashdot',
    }

    # crossover detection
    def find_winner_crossovers(df):
        n_vals = sorted(df['n_ratings'].unique())

        def winner_at(n):
            sub = df[df['n_ratings'] == n].dropna(subset=['ndcg'])
            if sub.empty:
                return None
            return sub.loc[sub['ndcg'].idxmax(), 'model']

        winners = [(n, winner_at(n)) for n in n_vals]
        winners = [(n, w) for n, w in winners if w is not None]
        crossovers = []

        for i in range(len(winners) - 1):
            n0, w0 = winners[i]
            n1, w1 = winners[i + 1]
            if w0 == w1:
                continue

            x0 = np.log10(n0)
            x1 = np.log10(n1)

            def ndcg(model, n):
                row = df[(df['model'] == model) & (df['n_ratings'] == n)]
                if row.empty or row['ndcg'].isna().all():
                    return np.nan
                return float(row['ndcg'].values[0])

            ya0, ya1 = ndcg(w0, n0), ndcg(w0, n1)
            yb0, yb1 = ndcg(w1, n0), ndcg(w1, n1)

            if any(np.isnan(v) for v in [ya0, ya1, yb0, yb1]):
                crossovers.append(((x0 + x1) / 2, w0, w1))
                continue

            denom = (ya1 - ya0) - (yb1 - yb0)
            if abs(denom) < 1e-10:
                crossovers.append(((x0 + x1) / 2, w0, w1))
            else:
                t = float(np.clip((yb0 - ya0) / denom, 0, 1))
                crossovers.append((x0 + t * (x1 - x0), w0, w1))

        return crossovers

    crossovers_a = find_winner_crossovers(df_a)
    crossovers_b = find_winner_crossovers(df_b)

    # build figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            'Experiment A: Shrinking Catalogue',
            'Experiment B: Fixed Catalogue (62,423 films)',
        ],
        horizontal_spacing=0.10,
    )

    legend_shown = set()

    def add_model_traces(df, col, crossovers):
        for model in MODEL_ORDER:
            sub = df[df['model'] == model].sort_values('log_n')
            if sub.empty:
                continue

            # drop NaN NDCG rows for plotting (keep points that exist)
            sub_valid = sub.dropna(subset=['ndcg'])
            if sub_valid.empty:
                continue

            x   = sub_valid['log_n'].tolist()
            y   = sub_valid['ndcg'].tolist()
            sp  = sub_valid['sparsity'].tolist()
            nr  = sub_valid['n_ratings'].tolist()
            ylo = sub_valid['ci_lower'].tolist() if 'ci_lower' in sub_valid.columns else None
            yhi = sub_valid['ci_upper'].tolist() if 'ci_upper' in sub_valid.columns else None

            colour   = COLOURS[model]
            show_leg = model not in legend_shown
            label    = MODEL_LABELS[model]

            hover = [
                f'<b>{label}</b><br>'
                f'NDCG@10: {yi:.4f}<br>'
                f'95% CI: [{lo:.4f}, {hi:.4f}]<br>'
                f'Sparsity: {si}<br>'
                f'n_ratings: {ni:,}'
                for yi, lo, hi, si, ni in zip(
                    y,
                    ylo if ylo else [float('nan')] * len(y),
                    yhi if yhi else [float('nan')] * len(y),
                    sp, nr,
                )
            ]

            # CI shaded band
            if ylo and yhi:
                fig.add_trace(go.Scatter(
                    x=x + x[::-1],
                    y=yhi + ylo[::-1],
                    fill='toself',
                    fillcolor=colour,
                    opacity=0.15,
                    line=dict(width=0),
                    hoverinfo='skip',
                    showlegend=False,
                    name=label + ' CI',
                ), row=1, col=col)

            # main line
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines+markers',
                name=label,
                line=dict(color=colour, width=2.5, dash=DASH[model]),
                marker=dict(size=8, color=colour,
                            line=dict(width=1, color='white')),
                hovertemplate='%{customdata}<extra></extra>',
                customdata=hover,
                legendgroup=model,
                showlegend=show_leg,
            ), row=1, col=col)

            if show_leg:
                legend_shown.add(model)

        # crossover vertical lines — minimal labels, no overlapping text
        SHORT = {
            'popularity_baseline': 'Pop',
            'svd':                 'SVD',
            'penalized_svd':       'PenSVD',
            'content_based':       'CB',
            'hybrid':              'Hybrid',
            'ease':                'EASE',
        }
        # filter to unique x positions that are far enough apart (min 0.15 log units)
        filtered = []
        for xc, wb, wa in crossovers:
            if not filtered or (xc - filtered[-1][0]) >= 0.15:
                filtered.append((xc, wb, wa))

        for i, (xc, w_before, w_after) in enumerate(filtered):
            wb_short = SHORT.get(w_before, w_before)
            wa_short = SHORT.get(w_after, w_after)
            fig.add_vline(
                x=xc,
                line=dict(color='rgba(80,80,80,0.5)', dash='dot', width=1.2),
                annotation_text=f'{wb_short}→{wa_short}',
                annotation_position='top right' if i % 2 == 0 else 'top left',
                annotation_font_size=9,
                annotation_font_color='rgba(80,80,80,0.85)',
                row=1, col=col,
            )

    add_model_traces(df_a, col=1, crossovers=crossovers_a)
    add_model_traces(df_b, col=2, crossovers=crossovers_b)

    # layout
    tick_log = [np.log10(v) for v in [1_000, 10_000, 100_000, 1_000_000,
                                       10_000_000, 25_000_000]]
    tick_txt = ['1K', '10K', '100K', '1M', '10M', '25M']
    axis_common = dict(
        tickvals=tick_log, ticktext=tick_txt,
        gridcolor='rgba(200,200,200,0.4)', showgrid=True, zeroline=False,
        title_font_size=12,
    )

    fig.update_xaxes(title_text='Number of Ratings (log scale)', **axis_common)
    fig.update_yaxes(
        title_text='NDCG@10',
        gridcolor='rgba(200,200,200,0.4)', showgrid=True,
        zeroline=True, zerolinecolor='rgba(150,150,150,0.5)', zerolinewidth=1,
        rangemode='tozero', title_font_size=12, row=1, col=1,
    )
    fig.update_yaxes(
        gridcolor='rgba(200,200,200,0.4)', showgrid=True,
        zeroline=True, zerolinecolor='rgba(150,150,150,0.5)', zerolinewidth=1,
        rangemode='tozero', row=1, col=2,
    )
    fig.update_layout(
        title=dict(
            text=(
                'NDCG@10 vs. Rating Density — Sparsity Experiments<br>'
                '<sup>Shaded bands = 95% bootstrap CI (n=500 users, 1,000 resamples) · '
                'Dashed lines = crossover between leading models · '
                'EASE: NaN points omitted (min_ratings filter or memory skip)</sup>'
            ),
            x=0.5, font_size=15,
        ),
        legend=dict(
            title='Model', orientation='h', yanchor='bottom', y=-0.22,
            xanchor='center', x=0.5, font_size=12, itemsizing='constant',
        ),
        hovermode='closest',
        plot_bgcolor='white', paper_bgcolor='white',
        height=520, width=1200,
        margin=dict(t=100, b=150, l=70, r=40),
    )

    out_path = OUTPUT_DIR + 'ndcg_sparsity_curves.html'
    fig.write_html(out_path, include_plotlyjs='cdn')
    print(f'Saved  →  {out_path}')


# 
# 2.  Model Dashboard  (add EASE to Panel 1 — NDCG bar only)
# 

def build_dashboard():
    summary     = pd.read_csv(OUTPUT_DIR + 'hybrid_summary.csv')
    fairness    = pd.read_csv(OUTPUT_DIR + 'fairness_comparison.csv')
    alpha_sweep = pd.read_csv(OUTPUT_DIR + 'hybrid_alpha_sweep.csv')

    # add EASE row if not already in hybrid_summary.csv
    if 'ease' not in summary['model'].values:
        ease_row = pd.DataFrame([{
            'model':  'ease',
            'alpha':  None,
            'ndcg10': 0.4346,
            'ci_lo':  0.4111,
            'ci_hi':  0.4572,
            'cat_cov': None,
            'lt_cov':  None,
        }])
        summary = pd.concat([summary, ease_row], ignore_index=True)

    MODEL_ORDER = ['popularity_baseline', 'svd', 'penalized_svd',
                   'content_based', 'hybrid_weighted (α=0.6)', 'hybrid_rrf', 'ease']
    MODEL_LABEL = {
        'popularity_baseline':     'Pop. Baseline',
        'svd':                     'SVD',
        'penalized_svd':           'Penalised SVD',
        'content_based':           'Content-Based',
        'hybrid_weighted (α=0.6)': 'Hybrid (α=0.6)',
        'hybrid_rrf':              'Hybrid RRF',
        'ease':                    'EASE (λ=500)',
    }
    MODEL_COLOUR = {
        'popularity_baseline':     '#636EFA',
        'svd':                     '#EF553B',
        'penalized_svd':           '#00CC96',
        'content_based':           '#AB63FA',
        'hybrid_weighted (α=0.6)': '#FFA15A',
        'hybrid_rrf':              '#19D3F3',
        'ease':                    '#00BCD4',
    }

    summary = summary.set_index('model').reindex(MODEL_ORDER).reset_index()
    summary['label']  = summary['model'].map(MODEL_LABEL)
    summary['colour'] = summary['model'].map(MODEL_COLOUR)

    # mean Popularity Rank (SVD & Penalised SVD only)
    fair_idx = fairness.set_index('Metric')
    pop_rank_map = {
        'svd':           float(fair_idx.loc['Mean Popularity Rank', 'SVD Baseline']),
        'penalized_svd': float(fair_idx.loc['Mean Popularity Rank', 'Penalized (λ=0.3)']),
    }
    summary['mean_pop_rank'] = summary['model'].map(pop_rank_map)
    summary['inv_pop_rank']  = 1.0 - summary['mean_pop_rank']

    def minmax(series):
        s = series.dropna()
        lo, hi = s.min(), s.max()
        return (series - lo) / (hi - lo) if hi > lo else series * 0 + 1.0

    summary['ndcg_norm']   = minmax(summary['ndcg10'])
    summary['catcov_norm'] = minmax(summary['cat_cov'])
    summary['ltcov_norm']  = minmax(summary['lt_cov'])

    known_pr = summary['inv_pop_rank'].dropna()
    pr_lo, pr_hi = known_pr.min(), known_pr.max()
    def norm_pr(v):
        if pd.isna(v): return 0.0
        return float((v - pr_lo) / (pr_hi - pr_lo)) if pr_hi > pr_lo else 1.0
    summary['pr_norm'] = summary['inv_pop_rank'].apply(norm_pr)

    # build 2×2 dashboard
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            '<b>NDCG@10 by Model</b>',
            '<b>Catalogue vs Long-Tail Coverage</b>',
            '<b>Hybrid: NDCG@10 & Coverage vs α</b>',
            '<b>Model Radar — Normalised Scores (0→1)</b>',
        ],
        specs=[
            [{'type': 'xy'},    {'type': 'xy'}   ],
            [{'type': 'xy'},    {'type': 'polar'}],
        ],
        vertical_spacing=0.22,
        horizontal_spacing=0.13,
    )

    # panel 1: NDCG@10 bars
    for _, row in summary.iterrows():
        ease_note = '<br><i>Exp A, sparsity=1.0, 25M ratings</i>' if row['model'] == 'ease' else ''
        fig.add_trace(go.Bar(
            x=[row['label']], y=[row['ndcg10']],
            error_y=dict(
                type='data', symmetric=False,
                array=[max(0.0, float(row['ci_hi']) - float(row['ndcg10']))],
                arrayminus=[max(0.0, float(row['ndcg10']) - float(row['ci_lo']))],
                color='rgba(50,50,50,0.6)', thickness=2, width=6,
            ),
            marker=dict(color=row['colour'], line=dict(width=0)),
            name=row['label'],
            legendgroup=row['model'],
            showlegend=True,
            hovertemplate=(
                f"<b>{row['label']}</b><br>"
                f"NDCG@10: {row['ndcg10']:.4f}<br>"
                f"95% CI: [{row['ci_lo']:.4f}, {row['ci_hi']:.4f}]"
                f"{ease_note}<extra></extra>"
            ),
        ), row=1, col=1)

    # panel 2: Coverage scatter (EASE skipped — no coverage metrics)
    # pixel offsets per model to avoid label overlap
    ANNOTATION_OFFSET = {
        'popularity_baseline':     ( 45,  -18),  # right of bottom-left point
        'svd':                     ( 45,   18),  # right, above svd
        'penalized_svd':           (-50,  -18),  # left of pen_svd
        'hybrid_rrf':              ( 45,   18),  # right of rrf
        'hybrid_weighted (α=0.6)': ( 10,  -28),  # below hybrid_weighted
        'content_based':           (  0,  -28),  # above content_based
    }
    cov_rows = summary[summary['cat_cov'].notna()].reset_index(drop=True)
    for _, row in cov_rows.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['cat_cov']], y=[row['lt_cov']],
            mode='markers',
            marker=dict(color=row['colour'], size=14,
                        line=dict(color='white', width=1.5)),
            name=row['label'],
            legendgroup=row['model'],
            showlegend=False,
            hovertemplate=(
                f"<b>{row['label']}</b><br>"
                f"Catalogue Coverage: {row['cat_cov']:.2f}%<br>"
                f"Long-Tail Coverage: {row['lt_cov']:.2f}%<extra></extra>"
            ),
        ), row=1, col=2)

        ax, ay = ANNOTATION_OFFSET.get(row['model'], (0, -28))
        fig.add_annotation(
            x=row['cat_cov'], y=row['lt_cov'],
            xref='x2', yref='y2',
            text=row['label'],
            showarrow=True,
            arrowhead=0,
            arrowwidth=1,
            arrowcolor='rgba(120,120,120,0.6)',
            ax=ax, ay=ay,
            font=dict(size=10, color='#222222'),
            bgcolor='rgba(255,255,255,0.85)',
            borderpad=2,
        )

    # panel 3: Alpha sweep
    x_fwd = alpha_sweep['alpha'].tolist()
    x_rev = x_fwd[::-1]

    fig.add_trace(go.Scatter(
        x=x_fwd + x_rev,
        y=alpha_sweep['ci_hi'].tolist() + alpha_sweep['ci_lo'].tolist()[::-1],
        fill='toself', fillcolor='rgba(99,110,250,0.15)',
        line=dict(width=0), hoverinfo='skip', showlegend=False,
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=alpha_sweep['alpha'], y=alpha_sweep['ndcg10'],
        mode='lines+markers', name='NDCG@10 (α sweep)',
        line=dict(color='#636EFA', width=2.5),
        marker=dict(size=7, color='#636EFA', line=dict(color='white', width=1)),
        legendgroup='alpha_ndcg', showlegend=True,
        hovertemplate='<b>α = %{x:.1f}</b><br>NDCG@10: %{y:.4f}<extra>NDCG@10</extra>',
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=alpha_sweep['alpha'], y=alpha_sweep['cat_cov'],
        mode='lines+markers', name='Cat. Coverage % (α sweep)',
        line=dict(color='#AB63FA', width=2, dash='dot'),
        marker=dict(size=6, color='#AB63FA'),
        legendgroup='alpha_cat', showlegend=True,
        hovertemplate='<b>α = %{x:.1f}</b><br>Catalogue Coverage: %{y:.2f}%<extra>Cat. Coverage</extra>',
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=alpha_sweep['alpha'], y=alpha_sweep['lt_cov'],
        mode='lines+markers', name='LT Coverage % (α sweep)',
        line=dict(color='#FFA15A', width=2, dash='dash'),
        marker=dict(size=6, color='#FFA15A'),
        legendgroup='alpha_lt', showlegend=True,
        hovertemplate='<b>α = %{x:.1f}</b><br>Long-Tail Coverage: %{y:.2f}%<extra>LT Coverage</extra>',
    ), row=2, col=1)

    fig.add_vline(x=0.6, line=dict(color='black', dash='dash', width=1.2),
                  annotation_text='α=0.6\n(selected)', annotation_font_size=9,
                  annotation_position='top right', row=2, col=1)

    # panel 4: Radar — include EASE if coverage metrics are available
    RADAR_DIMS = ['NDCG@10', 'Cat. Coverage', 'LT Coverage', 'Inv. Pop. Rank']
    radar_rows = summary[summary['cat_cov'].notna()].reset_index(drop=True)

    for _, row in radar_rows.iterrows():
        r_vals = [row['ndcg_norm'], row['catcov_norm'],
                  row['ltcov_norm'], row['pr_norm']]
        r_closed    = r_vals + [r_vals[0]]
        dims_closed = RADAR_DIMS + [RADAR_DIMS[0]]

        has_pr  = not pd.isna(row.get('mean_pop_rank'))
        pr_line = (f"Inv. Pop. Rank: {row['inv_pop_rank']:.3f} (norm {row['pr_norm']:.2f})"
                   if has_pr else "Inv. Pop. Rank: N/A")
        hover = (
            f"<b>{row['label']}</b><br>"
            f"NDCG@10: {row['ndcg10']:.4f} → norm {row['ndcg_norm']:.2f}<br>"
            f"Cat. Coverage: {row['cat_cov']:.2f}% → norm {row['catcov_norm']:.2f}<br>"
            f"LT Coverage: {row['lt_cov']:.2f}% → norm {row['ltcov_norm']:.2f}<br>"
            f"{pr_line}"
        )
        c    = row['colour']
        rgba = (f"rgba({int(c[1:3],16)},"
                f"{int(c[3:5],16)},"
                f"{int(c[5:7],16)},0.12)")

        fig.add_trace(go.Scatterpolar(
            r=r_closed, theta=dims_closed,
            fill='toself', fillcolor=rgba,
            line=dict(color=row['colour'], width=2),
            name=row['label'],
            legendgroup=row['model'],
            showlegend=False,
            hovertemplate=hover + '<extra></extra>',
        ), row=2, col=2)

    # global layout
    fig.update_layout(
        title=dict(
            text=(
                'Model Comparison Dashboard<br>'
                '<sup>Top: NDCG@10 bars | Coverage scatter  ·  '
                'Bottom: Hybrid α sweep | Normalised radar  ·  '
                'EASE shown in NDCG bar only (coverage metrics not computed)</sup>'
            ),
            x=0.5, font_size=15,
        ),
        barmode='group',
        plot_bgcolor='white', paper_bgcolor='white',
        height=940, width=1220,
        margin=dict(t=115, b=140, l=60, r=60),
        legend=dict(
            orientation='h', yanchor='bottom', y=-0.16,
            xanchor='center', x=0.5,
            font_size=11, itemsizing='constant', tracegroupgap=4,
        ),
        hovermode='closest',
    )

    grid = dict(showgrid=True, gridcolor='rgba(200,200,200,0.4)', zeroline=False)
    fig.update_xaxes(title_text='Model', **grid, row=1, col=1)
    fig.update_yaxes(title_text='NDCG@10', showgrid=True,
                     gridcolor='rgba(200,200,200,0.4)',
                     zeroline=True, zerolinecolor='rgba(150,150,150,0.4)', row=1, col=1)
    fig.update_xaxes(title_text='Catalogue Coverage (%)', **grid, row=1, col=2)
    fig.update_yaxes(title_text='Long-Tail Coverage (%)', showgrid=True,
                     gridcolor='rgba(200,200,200,0.4)',
                     zeroline=True, zerolinecolor='rgba(150,150,150,0.4)', row=1, col=2)
    fig.update_xaxes(title_text='Content-Based Weight α', dtick=0.1, **grid, row=2, col=1)
    fig.update_yaxes(title_text='Value', showgrid=True,
                     gridcolor='rgba(200,200,200,0.4)', zeroline=False, row=2, col=1)
    fig.update_polars(
        radialaxis=dict(range=[0, 1.05], showticklabels=True,
                        tickfont_size=8, gridcolor='rgba(180,180,180,0.5)'),
        angularaxis=dict(tickfont_size=10),
        bgcolor='white',
    )

    out_path = OUTPUT_DIR + 'model_dashboard.html'
    fig.write_html(out_path, include_plotlyjs='cdn')
    print(f'Saved  →  {out_path}')


# 
if __name__ == '__main__':
    build_sparsity_curves()
    build_dashboard()
    print('\nDone.')
