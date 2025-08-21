import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体和样式
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def load_jsonl_to_df(filepath):
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            model = data['model']
            dataset = data['dataset']
            for t in data['timelist']:
                t['model'] = model
                t['dataset'] = dataset
                records.append(t)
    return pd.DataFrame(records)

df = load_jsonl_to_df('./timelist.jsonl')

df['iteration'] = df['iteration'].astype(int)

print(f"共加载 {len(df['model'].unique())} 个模型, {len(df['dataset'].unique())} 个数据集")
print("模型:", df['model'].unique())
print("数据集:", df['dataset'].unique())

for dataset_name in df['dataset'].unique():
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df[df['dataset'] == dataset_name],
        x='iteration',
        y='average_time',
        hue='model',
        linewidth=2.5,
        palette='Set1'
    )
    plt.title(f'Average Inference Time vs Iteration\nDataset: {dataset_name}')
    plt.xlabel('Iteration')
    plt.ylabel('Average Time per Sample (s)')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

for model_name in df['model'].unique():
    plt.figure(figsize=(10, 6))

    # 筛选当前模型的数据
    subset = df[df['model'] == model_name]

    # 绘制该模型在不同数据集上的 average_time 曲线
    sns.lineplot(
        data=subset,
        x='iteration',
        y='average_time',
        hue='dataset',  # 不同数据集用不同颜色
        style='dataset',  # 不同数据集用不同线型（可选）
        markers=False,
        linewidth=2.5,
        palette='Dark2'
    )

    plt.title(f'Average Inference Time vs Iteration\nModel: {model_name.upper()}')
    plt.xlabel('Iteration')
    plt.ylabel('Average Time per Sample (s)')
    plt.legend(title='Dataset', loc='upper right')
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(12, 8))
sns.lineplot(
    data=df,
    x='iteration',
    y='average_time',
    hue='model',
    style='dataset',
    markers=False,
    linewidth=2.5,
    palette='Set1'
)
plt.title('Average Inference Time vs Iteration (All Models & Datasets)')
plt.xlabel('Iteration')
plt.ylabel('Average Time per Sample (s)')
plt.legend(title='Model & Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

g = sns.FacetGrid(
    df,
    col='dataset',
    hue='model',
    sharey=False,
    col_wrap=2,
    height=5,
    aspect=1.2
)
g.map(sns.lineplot, 'iteration', 'average_time', linewidth=2.5)
g.add_legend()
g.set_axis_labels('Iteration', 'Average Time (s)')
g.set_titles('Dataset: {col_name}')
g.fig.suptitle('Model Comparison by Dataset', y=1.02, fontsize=14)
plt.show()

summary = df.groupby(['model', 'dataset']).agg(
    final_avg_time=('average_time', 'last'),  # 最后一次的 average_time
    total_time_500=('total_time', 'last'),   # 第500次的 total_time
    throughput=('average_time', lambda x: 1 / x.iloc[-1])  # 吞吐量 = 1 / avg_time
).round(3)

print("\n=== 性能汇总（第500个样本）=== ")
print(summary)


throughput_df = summary.reset_index()[['model', 'dataset', 'throughput']]
throughput_df['label'] = throughput_df['model'] + ' @ ' + throughput_df['dataset']

plt.figure(figsize=(10, 6))
bars = sns.barplot(data=throughput_df, x='label', y='throughput', palette='viridis')
plt.title('Throughput Comparison (samples/sec) at Iteration 500')
plt.ylabel('Throughput (samples/sec)')
plt.xlabel('Model @ Dataset')
plt.xticks(rotation=30)


for i, bar in enumerate(bars.patches):
    height = bar.get_height()
    bars.annotate(f'{height:.2f}',
                  xy=(bar.get_x() + bar.get_width() / 2, height),
                  xytext=(0, 5),
                  textcoords='offset points',
                  ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()