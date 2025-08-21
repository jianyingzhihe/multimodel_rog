import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取jsonl文件
records = []
with open('timelist.jsonl', 'r') as file:
    for line in file:
        entry = json.loads(line.strip())
        for time_entry in entry['timelist']:
            records.append({
                'model': entry['model'],
                'dataset': entry['dataset'],
                'lora': entry['lora'],
                'iteration': time_entry['iteration'],
                'average_time': time_entry['average_time']
            })

df = pd.DataFrame(records)

# 绘制图形并保存
for model_name in df['model'].unique():
    subset = df[df['model'] == model_name]

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=subset,
        x='iteration',
        y='average_time',
        hue='dataset',
        style='lora',
        markers=True,
        linewidth=2.5,
        palette='Dark2'
    )

    plt.title(f'Average Inference Time vs Iteration\nModel: {model_name.upper()}')
    plt.xlabel('Iteration')
    plt.ylabel('Average Time per Sample (s)')
    plt.legend(title='Dataset & LoRA', loc='upper right')

    # 保存图片
    filename = f"{model_name}_inference_time.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()


# 提取第 500 次迭代的数据
final_df = df[df['iteration'] == 500].copy()
final_df['model_lora'] = final_df['model'] + ' (L:' + final_df['lora'].str.upper() + ')'
final_df['throughput'] = 1 / final_df['average_time']

# 按数据集画柱状图
for dataset_name in final_df['dataset'].unique():
    plt.figure(figsize=(10, 6))
    subset = final_df[final_df['dataset'] == dataset_name]

    bars = sns.barplot(
        data=subset,
        x='model_lora',
        y='throughput',
        hue='lora',
        palette='Set2'
    )

    plt.title(f'Throughput Comparison at Iteration 500\nDataset: {dataset_name.upper()}')
    plt.ylabel('Throughput (samples/sec)')
    plt.xlabel('Model & LoRA')
    plt.legend(title='LoRA', loc='upper right')

    # 添加数值标签
    for bar in bars.patches:
        height = bar.get_height()
        bars.annotate(f'{height:.2f}',
                      xy=(bar.get_x() + bar.get_width() / 2, height),
                      xytext=(0, 5),
                      textcoords='offset points',
                      ha='center', va='bottom', fontsize=9)

    filename = f"{dataset_name}_throughput_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"已保存: {filename}")

    plt.show()
    plt.close()