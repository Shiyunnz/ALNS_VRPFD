"""
导出无人机数量敏感性分析的可视化数据到 CSV 文件
"""

import pandas as pd
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def export_visualization_data():
    """导出可视化所需的所有数据"""

    # 读取原始结果
    results_dir = Path(__file__).parent / "results_new" / "drone_count"
    results_file = results_dir / "drone_count_sensitivity_results.csv"

    if not results_file.exists():
        print(f"错误: 找不到结果文件 {results_file}")
        return

    df = pd.read_csv(results_file)
    print(f"读取了 {len(df)} 条记录")

    # 1. 按无人机数量汇总的统计数据 (用于左右图表)
    summary = df.groupby('drone_count').agg({
        'best_cost': 'mean',
        'truck_distance_cost': 'mean',
        'drone_distance_cost': 'mean',
        'cost_reduction_percent': 'mean',
        'drone_customers': 'mean'
    }).reset_index()

    summary.columns = [
        'drone_count',
        'avg_total_cost',
        'avg_truck_distance_cost',
        'avg_drone_distance_cost',
        'avg_cost_reduction_percent',
        'avg_drone_customers'
    ]

    # 保存汇总数据
    summary_file = results_dir / "drone_count_summary_for_plot.csv"
    summary.to_csv(summary_file, index=False, encoding='utf-8')
    print(f"\n汇总数据已保存到: {summary_file}")
    print("\n=== 汇总数据 (图表数据) ===")
    print(summary.to_string(index=False))

    # 2. 按实例分组的数据 (用于每个实例的详细分析)
    per_instance = df[['instance', 'drone_count', 'best_cost',
                       'truck_distance_cost', 'drone_distance_cost',
                       'cost_reduction_percent', 'drone_customers']].copy()

    per_instance_file = results_dir / "drone_count_per_instance_for_plot.csv"
    per_instance.to_csv(per_instance_file, index=False, encoding='utf-8')
    print(f"\n每实例数据已保存到: {per_instance_file}")
    print(
        f"共 {len(per_instance)} 条记录 ({len(df['instance'].unique())} 个实例 × {len(df['drone_count'].unique())} 种无人机配置)")

    # 3. 打印详细的汇总统计
    print("\n" + "="*60)
    print("图表数据详情")
    print("="*60)

    print("\n【左图 - 成本分解与降低率】")
    print("-" * 50)
    for _, row in summary.iterrows():
        print(f"无人机数量 {int(row['drone_count'])}:")
        print(f"  平均总成本: {row['avg_total_cost']:.2f}")
        print(f"  卡车距离成本: {row['avg_truck_distance_cost']:.2f}")
        print(f"  无人机距离成本: {row['avg_drone_distance_cost']:.2f}")
        print(f"  成本降低率: {row['avg_cost_reduction_percent']:.1f}%")

    print("\n【右图 - 无人机服务客户数】")
    print("-" * 50)
    for _, row in summary.iterrows():
        print(
            f"无人机数量 {int(row['drone_count'])}: 平均服务 {row['avg_drone_customers']:.1f} 个客户")

    print("\n" + "="*60)
    print("数据导出完成!")
    print("="*60)


if __name__ == "__main__":
    export_visualization_data()
