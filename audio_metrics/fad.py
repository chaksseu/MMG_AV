"""
Calculate Frechet Audio Distance betweeen two audio directories.

Frechet distance implementation adapted from: https://github.com/mseitzer/pytorch-fid
VGGish adapted from: https://github.com/harritaylor/torchvggish
"""
import os
import numpy as np
import torch

from torch import nn
from scipy import linalg
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from utils.load_mel import WaveDataset
from torch.utils.data import DataLoader
from audio_metrics.torchvggish.vggish import VGGish


class FrechetAudioDistance:
    def __init__(
        self,
        use_pca=False,
        use_activation=False,
        verbose=False,
        audio_load_worker=8,
        device="cpu",  # 기본값을 CPU로 설정, 필요한 경우 "cuda"로 변경 가능
    ):
        # device 설정
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.verbose = verbose
        self.audio_load_worker = audio_load_worker
        self.__get_model(use_pca=use_pca, use_activation=use_activation)

    def __get_model(self, use_pca=False, use_activation=False):
        """
        Load the VGGish model from torch hub.
        """
        # map_location을 self.device로 설정해 모델이 CPU 또는 GPU 원하는 곳으로 로드되도록 함
        # self.model = torch.hub.load(
        #     "harritaylor/torchvggish",
        #     "vggish",
        #     device=self.device
        # )



        dependencies = ['torch', 'numpy', 'resampy', 'soundfile']


        model_urls = {
            'vggish': 'https://github.com/harritaylor/torchvggish/'
                    'releases/download/v0.1/vggish-10086976.pth',
            'pca': 'https://github.com/harritaylor/torchvggish/'
                'releases/download/v0.1/vggish_pca_params-970ea276.pth'
        }

        self.model = VGGish(urls=model_urls, device=self.device)


        if not use_pca:
            self.model.postprocess = False

        # 중간 feature만 사용하려면 마지막 레이어 제거
        if not use_activation:
            self.model.embeddings = nn.Sequential(
                *list(self.model.embeddings.children())[:-1]
            )

        self.model.eval()
        self.model.to(self.device)

    def load_audio_data(self, x):
        """
        주어진 폴더(혹은 리스트)에 있는 오디오 파일들을 WaveDataset으로 로드하고,
        DataLoader를 통해 (audio_tensor, sr) 형태로 추출합니다.
        """
        outputloader = DataLoader(
            WaveDataset(
                x,
                16000,
                limit_num=None,
            ),
            batch_size=1,
            sampler=None,
            num_workers=0,  # 멀티프로세싱 사용 시 변경 가능
        )
        data_list = []
        for batch in tqdm(outputloader):
            # batch[0] shape 예: (1, 1, waveform_length)
            # batch[0][0, 0] -> 오디오 (1D 텐서)
            data_list.append((batch[0][0, 0], 16000))  # (audio_tensor, sr)
        return data_list

    def get_embeddings(self, x, sr=16000, limit_num=None):
        """
        Get embeddings using VGGish model.
        Params:
        -- x    : Either
            (i)  a string which is the directory of a set of audio files, or
            (ii) a list of np.ndarray audio samples
        -- sr   : Sampling rate, if x is a list of audio samples. Default value is 16000.
        """
        embd_lst = []
        # x가 폴더 경로라면 load_audio_data를 통해 리스트 형태로 변환
        x = self.load_audio_data(x)

        # x가 list 형식인지 확인
        if not isinstance(x, list):
            raise AttributeError("Input x should be a list of audio tensors.")

        # 모델 디바이스 확인 (디버깅용)
        # print(f"Model device: {next(self.model.parameters()).device}")

        for audio, sr in tqdm(x, disable=(not self.verbose)):
            # audio는 기본적으로 CPU 텐서일 가능성이 높으나,
            # 혹시 모를 상황을 대비해 명시적으로 self.device로 이동
            audio = audio.to(self.device, dtype=torch.float32)

            # VGGish 모델은 내부적으로 CPU 텐서를 사용하는 구조일 수 있으므로
            # forward 전에 audio를 다시 CPU로 내리고, numpy로 변환
            audio_np = audio.cpu().numpy()

            # 모델 forward (NumPy 배열 입력)
            embd = self.model.forward(audio_np, sr)

            # 모델의 출력 embd는 torch.Tensor 형태이므로 CPU 텐서로 변환 후 numpy()
            embd = embd.detach().cpu().numpy()
            embd_lst.append(embd)

        return np.concatenate(embd_lst, axis=0)

    def calculate_embd_statistics(self, embd_lst):
        """
        Calculate the mean (mu) and covariance (sigma) of the embeddings.
        """
        if isinstance(embd_lst, list):
            embd_lst = np.array(embd_lst)
        mu = np.mean(embd_lst, axis=0)
        sigma = np.cov(embd_lst, rowvar=False)
        return mu, sigma

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        Adapted from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
        """
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert (
            mu1.shape == mu2.shape
        ), "Training and test mean vectors have different lengths"
        assert (
            sigma1.shape == sigma2.shape
        ), "Training and test covariances have different dimensions"

        diff = mu1 - mu2

        # Product might be almost singular
        from numpy import isfinite
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; "
                f"adding {eps} to diagonal of cov estimates"
            )
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f"Imaginary component {m}")
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        return (diff.dot(diff)
                + np.trace(sigma1)
                + np.trace(sigma2)
                - 2 * tr_covmean)

    def score(self, background_dir, eval_dir, limit_num=None):
        """
        Calculate FAD between two directories of audio files.
        background_dir: generated samples
        eval_dir: groundtruth samples
        """
        embds_background = self.get_embeddings(background_dir, limit_num=limit_num)
        embds_eval = self.get_embeddings(eval_dir, limit_num=limit_num)

        if len(embds_background) == 0:
            print("[Frechet Audio Distance] background set dir is empty, exiting...")
            return -1

        if len(embds_eval) == 0:
            print("[Frechet Audio Distance] eval set dir is empty, exiting...")
            return -1

        mu_background, sigma_background = self.calculate_embd_statistics(embds_background)
        mu_eval, sigma_eval = self.calculate_embd_statistics(embds_eval)

        fad_score = self.calculate_frechet_distance(
            mu_background,
            sigma_background,
            mu_eval,
            sigma_eval
        )

        return fad_score
