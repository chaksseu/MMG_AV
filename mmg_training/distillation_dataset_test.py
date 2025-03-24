import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os

class ABCTextDataset(Dataset):
    """
    세 개의 CSV 파일에서 데이터를 읽어,
    - 각 데이터셋의 전체 개수 중 최대값을 한 epoch의 길이로 정합니다.
    - A, B, C 각각은 자신만의 full cycle(한 바퀴)를 반복하며, 
      full cycle마다 랜덤 순서로 섞인 인덱스를 사용합니다.
    """
    def __init__(self, csv_a, csv_b, csv_c, text_column="text"):
        self.data_A = pd.read_csv(csv_a)[text_column].tolist()
        self.data_B = pd.read_csv(csv_b)[text_column].tolist()
        self.data_C = pd.read_csv(csv_c)[text_column].tolist()

        # epoch_length는 가장 샘플 수가 많은 데이터셋 기준으로 결정합니다.
        self.epoch_length = max(len(self.data_A), len(self.data_B), len(self.data_C))
        self.on_epoch_end()  # 초기 페어링 설정

    def _generate_permutation(self, data_length):
        """데이터 개수가 data_length인 데이터셋에 대해,
           epoch_length 길이만큼 full cycle 반복 랜덤 permutation 생성"""
        num_full_cycles = self.epoch_length // data_length
        remainder = self.epoch_length % data_length
        perm = []
        for _ in range(num_full_cycles):
            perm.extend(np.random.permutation(data_length).tolist())
        if remainder:
            perm.extend(np.random.permutation(data_length)[:remainder].tolist())
        return np.array(perm)

    def on_epoch_end(self):
        self.permutation_A = self._generate_permutation(len(self.data_A))
        self.permutation_B = self._generate_permutation(len(self.data_B))
        self.permutation_C = self._generate_permutation(len(self.data_C))

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, idx):
        a_sample = self.data_A[self.permutation_A[idx]]
        b_sample = self.data_B[self.permutation_B[idx]]
        c_sample = self.data_C[self.permutation_C[idx]]
        return a_sample, b_sample, c_sample

def main():
    # 테스트용 간단한 CSV 데이터 생성
    data_A = {'text': [f"Sample A {i}" for i in range(10, 18)]}   # 10개 샘플
    data_B = {'text': [f"Sample B {i}" for i in range(20, 24)]}   # 9개 샘플
    data_C = {'text': [f"Sample C {i}" for i in range(30, 32)]}   # 3개 샘플

    df_A = pd.DataFrame(data_A)
    df_B = pd.DataFrame(data_B)
    df_C = pd.DataFrame(data_C)

    csv_file_A = "data_A.csv"
    csv_file_B = "data_B.csv"
    csv_file_C = "data_C.csv"

    df_A.to_csv(csv_file_A, index=False)
    df_B.to_csv(csv_file_B, index=False)
    df_C.to_csv(csv_file_C, index=False)

    # DataLoader에서는 내부에서 이미 순서를 섞은 permutation에 따라 데이터를 읽으므로 shuffle=False
    dataset = ABCTextDataset(csv_file_A, csv_file_B, csv_file_C, text_column="text")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)

    num_epochs = 5
    for epoch in range(num_epochs):
        dataset.on_epoch_end()  # epoch마다 순서를 새로 생성
        print(f"\n=== Epoch {epoch + 1} (epoch_length = {len(dataset)}) ===")
        for batch_idx, (a_batch, b_batch, c_batch) in enumerate(dataloader):
            print(f"[Epoch {epoch + 1}, Batch {batch_idx + 1}]")
            print("  A:", list(a_batch))
            print("  B:", list(b_batch))
            print("  C:", list(c_batch))

    # 테스트 후 임시 CSV 파일 삭제
    os.remove(csv_file_A)
    os.remove(csv_file_B)
    os.remove(csv_file_C)

if __name__ == "__main__":
    main()
