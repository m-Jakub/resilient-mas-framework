import argparse
import pandas as pd


def compute_prr_pivot(csv_path: str, threshold: float = -0.20):
    df = pd.read_csv(csv_path)

    if 'run_id' not in df.columns:
        for alt in ('session_id', 'sessionId', 'session'):
            if alt in df.columns:
                df = df.rename(columns={alt: 'run_id'})
                break

    if 'turn' not in df.columns:
        for alt in ('turn_number', 'turnNumber', 'turn_idx'):
            if alt in df.columns:
                df = df.rename(columns={alt: 'turn'})
                break

    if 'judge_score' not in df.columns:
        for alt in ('overall', 'judge', 'score'):
            if alt in df.columns:
                df = df.rename(columns={alt: 'judge_score'})
                break

    # normalize scenario and condition column names
    if 'scenario' not in df.columns:
        for alt in ('scenario_id', 'scenarioId', 'scenario_name', 'scenarioName'):
            if alt in df.columns:
                df = df.rename(columns={alt: 'scenario'})
                break

    if 'condition' not in df.columns:
        for alt in ('cond', 'condition_id', 'conditionId', 'condition_name', 'group'):
            if alt in df.columns:
                df = df.rename(columns={alt: 'condition'})
                break

    if 'is_shock_turn' not in df.columns:
        if 'shock_type_turn' in df.columns:
            df['is_shock_turn'] = ~df['shock_type_turn'].isna()
        else:
            raise KeyError("Data must include 'is_shock_turn' or 'shock_type_turn' to detect shock turns")

    # ensure deterministic ordering for diff
    df = df.sort_values(['run_id', 'turn'])

    # turn-by-turn difference within each run
    df['delta'] = df.groupby('run_id')['judge_score'].diff()

    # filter shock turns with drop <= threshold
    mask = (df['is_shock_turn'] == True) & (df['delta'] <= threshold)
    df_triggers = df.loc[mask]

    # group by scenario and condition and count unique run_id
    grouped = (
        df_triggers
        .groupby(['scenario', 'condition'])['run_id']
        .nunique()
        .reset_index(name='prr_run_count')
    )

    # pivot to scenario x condition
    pivot = (
        grouped.pivot(index='scenario', columns='condition', values='prr_run_count')
        .fillna(0)
        .astype(int)
    )

    return pivot


def main():
    parser = argparse.ArgumentParser(description='Compute PRR activations pivot by scenario and condition')
    parser.add_argument('--csv', required=True, help='Path to turns CSV')
    parser.add_argument('--threshold', type=float, default=-0.20, help='Delta threshold (default -0.20)')
    parser.add_argument('--out', help='Optional path to save pivot CSV')
    args = parser.parse_args()

    pivot = compute_prr_pivot(args.csv, threshold=args.threshold)

    print(pivot)

    if args.out:
        pivot.to_csv(args.out)


if __name__ == '__main__':
    main()
