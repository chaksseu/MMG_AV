import os
import math
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

# 로그 경로 설정
log_dir = 'tensorboard/0325_MMG_1e-5_HJ_8gpu'

# 이벤트 로딩
event_acc = EventAccumulator(log_dir)
event_acc.Reload()

# 저장 디렉토리 생성
output_dir = log_dir
os.makedirs(output_dir, exist_ok=True)

# 모든 스칼라 태그 가져오기
scalar_tags = event_acc.Tags()['scalars']

# 개별 그래프 저장
all_scalars = {}
for tag in scalar_tags:
    scalars = event_acc.Scalars(tag)
    steps = [s.step for s in scalars]
    values = [s.value for s in scalars]
    all_scalars[tag] = (steps, values)

    # 개별 저장
    plt.figure()
    plt.plot(steps, values)
    plt.xlabel("Step")
    plt.ylabel(tag)
    plt.title(f"{tag} over time")
    plt.tight_layout()

    safe_tag = tag.replace('/', '_')
    save_path = os.path.join(output_dir, f"{safe_tag}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved individual plot: {save_path}")

# 🔥 통합 subplot 이미지 저장
num_tags = len(all_scalars)
cols = 3  # 한 행에 3개씩
rows = math.ceil(num_tags / cols)

fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
axes = axes.flatten()  # 2D -> 1D 배열로

for i, (tag, (steps, values)) in enumerate(all_scalars.items()):
    axes[i].plot(steps, values)
    axes[i].set_title(tag, fontsize=10)
    axes[i].set_xlabel("Step")
    axes[i].set_ylabel("Value")
    axes[i].tick_params(axis='both', labelsize=8)

# 남은 subplot은 숨기기
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
combined_path = os.path.join(output_dir, "all_scalars_subplots.png")
plt.savefig(combined_path)
plt.close()

print(f"Saved combined subplot image: {combined_path}")
