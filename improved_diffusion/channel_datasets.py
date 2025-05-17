import torch
from torch.utils.data import Dataset, DataLoader
import hdf5storage
import numpy as np

def load_data(
    *, data_dir, batch_size, image_size=None, pilot_length=1, snr_db=20.0,
    quantize_y=True, n_bits=4, deterministic=False, train=True
):
    """
    Create a generator over (H_tensor, kwargs) batches using ChannelMatrixDataset.

    H_tensor: shape [B, 2, Tx, Rx]
    kwargs: dict containing:
        - "y": [B, 2, Tx, L] (real y)
        - "y_quant": [B, 2, Tx, L] (quantized y)
        - "p": [B, 2, Rx, L] (pilot)
        - "h_max": [B]
        - "scale_y": [B]
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    # 只取第一份文件（目前数据集中假设所有样本在一个大文件中）
    file_path = data_dir

    dataset = ChannelMatrixDataset(
        file_path=file_path,
        pilot_length=pilot_length,
        key="output_h",
        snr_db=snr_db,
        quantize_y=quantize_y,
        n_bits=n_bits,
        normalize=True,
        subcarrier_index=0,
        seed=42
    )

    # 如果指定了image_size，检查数据尺寸是否匹配
    if image_size is not None:
        if isinstance(image_size, tuple):
            # 确保转换为整数
            height = int(image_size[0])
            width = int(image_size[1])
            if height != dataset.tx or width != dataset.rx:
                raise ValueError(f"Specified image size ({height}, {width}) does not match data dimensions ({dataset.tx}, {dataset.rx})")
        else:
            # 单个数字的情况
            size = int(image_size)
            if size != dataset.tx or size != dataset.rx:
                raise ValueError(f"Specified image size {size} does not match data dimensions ({dataset.tx}, {dataset.rx})")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=1,
        drop_last=True
    )

    # 无限生成器
    while True:
        for batch in loader:
            H = batch["H"]  # [B, 2, Tx, Rx]
            if train==False:
                kwargs = {
                    "y": batch["y"],                     # [B, 2, Tx, L]
                    "y_quant": batch["y_quant"],         # [B, 2, Tx, L]
                    "p": batch["p"],                     # [B, 2, Rx, L]
                    "h_max": batch["h_max"],         # [B]
                    "scale_y": batch["scale_y"]          # [B]
                }
            else:
                kwargs = {}
            yield H, kwargs


class ChannelMatrixDataset(Dataset):
    def __init__(self, file_path, key="H", pilot_length=1, snr_db=20.0,
                 quantize_y=True, n_bits=4, normalize=True,
                 subcarrier_index=0, seed=42):
        self.file_path = file_path
        self.key = key
        self.pilot_length = pilot_length
        self.snr_db = snr_db
        self.quantize_y = quantize_y
        self.n_bits = n_bits
        self.normalize = normalize
        self.subcarrier_index = subcarrier_index
        self.seed = seed

        mat = hdf5storage.loadmat(self.file_path)
        H_full = mat[self.key]  # [N, S, Tx, Rx]
        assert H_full.ndim == 4
        self.H_all = H_full[:, self.subcarrier_index, :, :]
        self.num_samples, self.tx, self.rx = self.H_all.shape

        self.pilot = self._generate_qpsk_pilot(self.rx, self.pilot_length)
        self.p_concat = np.stack([np.real(self.pilot), np.imag(self.pilot)], axis=0)  # [2, Rx, L]

        # 计算H的最大值用于归一化
        self.h_max = np.max(np.abs(self.H_all))
        if self.normalize:
            self.H_all_norm = self.H_all / self.h_max

        self.scale_y = self._compute_y_quant_scale()
        self.rng = np.random.default_rng(seed)

        print(f"[Dataset] Samples: {self.num_samples}, Tx: {self.tx}, Rx: {self.rx}")
        print(f"[Pilot] Shape: {self.pilot.shape}, SNR: {self.snr_db} dB")
        print(f"[Global scale] h_max: {self.h_max:.4f}, scale_y (0.99) {self.scale_y:.4f}")

    def _generate_qpsk_pilot(self, rx_dim, L):
        rng = np.random.default_rng(self.seed)
        phases = rng.choice([0, np.pi/2, np.pi, 3*np.pi/2], size=(rx_dim, L))
        return np.exp(1j * phases)

    def _compute_y_quant_scale(self):
        y_vals = []
        for i in range(min(self.num_samples, 500)):
            if self.normalize:
                H = self.H_all_norm[i]
            else:
                H = self.H_all[i]
            y = H @ self.pilot
            y_vals.append(np.abs(np.real(y)).ravel())
            y_vals.append(np.abs(np.imag(y)).ravel())
        return np.quantile(np.concatenate(y_vals), 0.99)

    def _uniform_quantize(self, x, scale, n_bits):
        levels = 2 ** n_bits
        x_clipped = np.clip(x / scale, -1.0, 1.0)
        x_scaled = (x_clipped + 1) * (levels - 1) / 2
        x_rounded = np.round(x_scaled)
        x_dequant = x_rounded / (levels - 1) * 2 - 1
        return x_dequant * scale

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.normalize:
            H = self.H_all_norm[idx]  # [Tx, Rx]
        else:
            H = self.H_all[idx]  # [Tx, Rx]

        y = H @ self.pilot

        signal_power = np.mean(np.abs(y) ** 2)
        snr_linear = 10 ** (self.snr_db / 10)
        noise_power = signal_power / snr_linear
        noise_std = np.sqrt(noise_power / 2)
        noise = self.rng.normal(0, noise_std, y.shape) + 1j * self.rng.normal(0, noise_std, y.shape)
        y = y + noise

        H_concat = np.stack([np.real(H), np.imag(H)], axis=0)  # [2, Tx, Rx]
        y_concat = np.stack([np.real(y), np.imag(y)], axis=0)  # [2, Tx, L]

        # y 量化（基于全局 scale_y）
        y_quant = self._uniform_quantize(y_concat, self.scale_y, self.n_bits) if self.quantize_y else y_concat

        return {
            "H": torch.from_numpy(H_concat.astype(np.float32)),
            "y": torch.from_numpy(y_concat.astype(np.float32)),
            "y_quant": torch.from_numpy(y_quant.astype(np.float32)),
            "p": torch.from_numpy(self.p_concat.astype(np.float32)),
            "h_max": torch.tensor(self.h_max, dtype=torch.float32),
            "scale_y": torch.tensor(self.scale_y, dtype=torch.float32),
        }



def test_loader():
    dataset = ChannelMatrixDataset(
        file_path="../datasets/diffusion_4321.mat",
        pilot_length=8,
        quantize_y=True,
        n_bits=4,
        key="output_h",
        snr_db=10
    )
    loader = DataLoader(dataset, batch_size=2)

    for batch in loader:
        print("H:", batch["H"].shape)           # [2, 64, 16]
        print("y:", batch["y"].shape)           # [2, 64, 8]
        print("y_quant:", batch["y_quant"].shape)
        print("h_max:", batch["h_max"])
        print("scale_y:", batch["scale_y"])
        print("p:", batch["p"].shape)           # [2, 16, 8]
        print("y vs y_quant example:", batch["y"][0, 0, :4], batch["y_quant"][0, 0, :4])
        break

        
def test_loader_bk():
    file_path = "../datasets/diffusion_4321.mat"
    dataset = ChannelMatrixDataset(
        file_path=file_path,
        pilot_length=8,
        snr_db=10,
        key="output_h",
        quantize_y=True,
        n_bits=4
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    for batch in loader:
        print("H:", batch["H"].shape)           # [B, 2, 64, 16]
        print("y:", batch["y"].shape)           # [B, 2, 64, 8]
        print("y_quant:", batch["y_quant"].shape)  # [B, 2, 64, 8]
        print("h_max:", batch["h_max"])
        print("scale_y:", batch["scale_y"])
        print("y[0][0][:4]:", batch["y"][0, 0, 0, :4])
        print("y_quant[0][0][:4]:", batch["y_quant"][0, 0, 0, :4])
        break


if __name__ == "__main__":
    test_loader()
