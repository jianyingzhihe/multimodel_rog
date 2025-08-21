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
    plt.close()  # 关闭当前图像，防止内存占用过高