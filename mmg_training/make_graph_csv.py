import pandas as pd
import matplotlib.pyplot as plt
import ast

# 파일 경로
csv_path = '/home/work/kby_hgh/MMG_01/eval_results.csv'
output_path = '/home/work/kby_hgh/MMG_01/metric_plot_separated.png'

# CSV 불러오기
df = pd.read_csv(csv_path)

# 문자열 파싱
df["FVD"] = df["FVD"].apply(lambda x: ast.literal_eval(x)["final"])
df["ImageBind"] = df["ImageBind"].apply(lambda x: ast.literal_eval(x)[0])

# step 및 learning rate 추출
df["step"] = df["inference_save_path"].apply(lambda x: int([s for s in x.split("_") if s.isdigit()][-1]))
df["lr"] = df["inference_save_path"].apply(lambda x: "1e-4" if "_1e-4_" in x else "1e-5")

# 메트릭 목록
metrics = ["FAD", "CLAP", "FVD", "CLIP", "ImageBind"]

# 그래프 설정: (행 = 메트릭 개수, 열 = 2개: 1e-4 / 1e-5)
fig, axes = plt.subplots(len(metrics), 2, figsize=(14, 18), sharex='col')

for i, metric in enumerate(metrics):
    for j, lr in enumerate(["1e-4", "1e-5"]):
        sub_df = df[df["lr"] == lr].sort_values(by="step")
        axes[i, j].plot(sub_df["step"], sub_df[metric], marker='o', label=lr)
        axes[i, j].set_title(f"{metric} ({lr})")
        axes[i, j].set_ylabel(metric)
        axes[i, j].grid(True)
        axes[i, j].legend()
        if i == len(metrics) - 1:
            axes[i, j].set_xlabel("Step")

plt.tight_layout()
plt.savefig(output_path)
print(f"그래프가 저장되었습니다: {output_path}")
