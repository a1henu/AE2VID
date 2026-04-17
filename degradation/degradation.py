import numpy as np
import cv2

class Degradation:
    def __init__(self, 
                 gamma=2.2, 
                 C_min=0.1, C_max=0.4, C_var_range=(0.02, 0.08),
                 time_err_range=(10.0, 1000.0),
                 hot_pixels_prob=0.01, cold_pixels_prob=0.01,
                 gaussian_noise_range=(1/255, 25/255),
                 salt_pepper_prob=0.01,
                 alpha=1.0,
                 t_end=3e5,
                 enable_hot_cold=False,
                 enable_down=False,
                 down_factors=None,
                 rng_seed=None,
                 noise_ref_resolution=(5120, 2880),
    ):
        self.gamma = float(gamma)

        # threshold parameter
        self.C_min = float(C_min)
        self.C_max = float(C_max)
        self.C_var_range = C_var_range

        # timestamp error
        self.time_err_range = time_err_range

        # hot/cold pixels
        self.hot_pixels_prob = float(hot_pixels_prob)
        self.cold_pixels_prob = float(cold_pixels_prob)
        self.enable_hot_cold = bool(enable_hot_cold)

        # spatial noises
        self.gaussian_noise_range = gaussian_noise_range
        self.salt_pepper_prob = float(salt_pepper_prob)

        self.alpha = float(alpha)
        self.t_end = float(t_end)
        
        self.enable_down = bool(enable_down)
        self.down_factors = down_factors if down_factors is not None else []

        self.noise_ref_resolution = tuple(noise_ref_resolution)
        self.rng = np.random.RandomState(rng_seed)

    # ---------- helpers ----------
    def _to_gray01(self, image):
        img = image.astype(np.float32)
        if img.ndim == 3:
            img = 0.2126*img[...,0] + 0.7152*img[...,1] + 0.0722*img[...,2]
        if img.max() > 1.0:
            img = img / 255.0
        return np.clip(img, 0.0, 1.0).astype(np.float32)

    def linearize_image(self, image):
        img = self._to_gray01(image)
        return np.power(img, self.gamma)

    def delinearize_image(self, linear_image):
        lin = np.clip(linear_image.astype(np.float32), 0.0, 1.0)
        return np.power(lin, 1.0 / self.gamma)

    # ---------- temporal modeling ----------
    def threshold_degradation(self, shape):
        C_base = self.rng.uniform(self.C_min, self.C_max)
        sigma = self.rng.uniform(*self.C_var_range)
        deltaC = self.rng.normal(loc=0.0, scale=sigma, size=shape).astype(np.float32)
        return C_base, deltaC

    def timestamp_degradation(self, temp_matrix):
        lam = self.rng.uniform(*self.time_err_range)
        timestamp_noise = self.rng.poisson(lam=lam, size=temp_matrix.shape).astype(np.float32)
        return np.clip(temp_matrix + timestamp_noise, 1.0, self.t_end).astype(np.float32)

    def hot_cold_pixels_degradation(self, temp_matrix):
        hot_pixels = self.rng.rand(*temp_matrix.shape) < self.hot_pixels_prob
        cold_pixels = self.rng.rand(*temp_matrix.shape) < self.cold_pixels_prob
        tmin = float(np.min(temp_matrix))
        tmax = float(np.max(temp_matrix))
        out = temp_matrix.copy()
        out[hot_pixels] = tmin
        out[cold_pixels] = tmax
        return out

    # ---------- intensity <-> temporal with alpha ----------
    def temp2gray(self, temp_matrix, C):
        """
        I = (exp(C)-1)/h(t), h(t)=t^{α+1}/((α+1) t_end^α)
          = ((α+1) t_end^α)*(exp(C)-1) / t^{α+1}
        归一化到 [0,1]
        """
        h_t = (temp_matrix**(self.alpha + 1)) / ((self.alpha + 1) * self.t_end**self.alpha)
        gray = (np.exp(C) - 1) / h_t
        return gray

    def gray2temp(self, gray_image, C):
        """
        t = ( ((α+1) t_end^α)*(exp(C)-1) / I )^{1/(α+1)}
        """
        h_t = (np.exp(C) - 1) / gray_image
        t = ( ((self.alpha + 1) * self.t_end**self.alpha) * h_t )**(1/(self.alpha + 1))
        return t

    # ---------- spatial degradations ----------
    def _size_scale(self, shape):
        """根据当前图像尺寸相对参考分辨率，给出噪声缩放系数（≤1）。"""
        h, w = shape[:2]
        ref_w, ref_h = self.noise_ref_resolution
        ref_pix = max(1, ref_w * ref_h)
        cur_pix = max(1, w * h)
        # 用像素数开方（等效于线性尺寸比）；并且不超过 1
        scale = cur_pix / ref_pix
        return float(min(1.0, scale))

    def gaussian_noise_degradation(self, gray_image):
        # scale = self._size_scale(gray_image.shape)   # 低分辨率 -> scale<1
        sigma_base = self.rng.uniform(*self.gaussian_noise_range)
        sigma = sigma_base / 16
        noise = self.rng.normal(0.0, sigma, size=gray_image.shape).astype(np.float32)
        return np.clip(gray_image.astype(np.float32) + noise, 0.0, 1.0)


    def poisson_gaussian_noise(self, img_lin, 
                            k_shot=(0.01, 0.03), 
                            sigma_read=(2/255, 6/255)): 
        x = img_lin.astype(np.float32)
        k = self.rng.uniform(*k_shot)
        sr = self.rng.uniform(*sigma_read) * self._size_scale(x.shape)
        shot = k * np.sqrt(np.clip(x, 0, 1)) * self.rng.normal(0.0, 1.0, x.shape).astype(np.float32)
        read = self.rng.normal(0.0, sr, x.shape).astype(np.float32)
        y = x + shot + read
        return np.clip(y, 0.0, 1.0)

    def blur_degradation(self, gray_image):
        ksize = int(self.rng.choice([3, 5, 7]))
        sigma = self.rng.uniform(0.4, 2.0)
        return cv2.GaussianBlur(gray_image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma,
                                borderType=cv2.BORDER_REFLECT101)
        
    # ---------- downsampling (only down, no up) ----------
    def _down_once(self, img01, factor: float):
        if not (0 < factor < 1):
            return img01
        h, w = img01.shape[:2]
        nh, nw = max(1, int(round(h*factor))), max(1, int(round(w*factor)))
        if nh == h and nw == w:
            return img01
        small = cv2.resize(img01, (nw, nh), interpolation=cv2.INTER_AREA)
        return small.astype(np.float32)

    def _apply_downs(self, img01, factors):
        out = img01
        for factor, times in factors:
            if not (0 < float(factor) < 1) or int(times) <= 0:
                continue
            for _ in range(int(times)):
                out = self._down_once(out, float(factor))
        return out

    def salt_pepper_degradation(self, img):
        p = self.salt_pepper_prob
        if p <= 0:
            return img
        noisy = img.copy()
        rnd = self.rng.rand(*img.shape)
        salt = rnd < (p / 2.0)
        pepper = rnd > (1.0 - p / 2.0)
        noisy[salt] = 1.0
        noisy[pepper] = 0.0
        return noisy

    # ---------- main pipeline ----------
    def degrade(self, HQ_img, apply_gamma_correction=True):
        # 1) 线性化
        HQ_lin = self.linearize_image(HQ_img) if apply_gamma_correction else self._to_gray01(HQ_img)
        HQ_lin = np.clip(HQ_lin, 1e-4, 1.0).astype(np.float32)

        # 2) 阈值采样 + gray->temp
        C_base, deltaC = self.threshold_degradation(HQ_lin.shape)
        tstar = self.gray2temp(HQ_lin, C_base)
        tstar = tstar * (1.0 + deltaC)

        # 3) 时间戳退化
        tstar = self.timestamp_degradation(tstar)

        # 4) 可选热/冷像素
        if self.enable_hot_cold and (self.hot_pixels_prob > 0 or self.cold_pixels_prob > 0):
            tstar = self.hot_cold_pixels_degradation(tstar)

        # 5) temp->gray
        gray_lin = self.temp2gray(tstar, C_base)

        # 6) 空间域退化
        gray_lin = self.gaussian_noise_degradation(gray_lin)
        # gray_lin = self.poisson_gaussian_noise(gray_lin, sigma_read=self.gaussian_noise_range)
        gray_lin = self.blur_degradation(gray_lin)
        if self.enable_down and self.down_factors:
            gray_lin = self._apply_downs(gray_lin, self.down_factors)
        
        # 7) 归一化
        vmin, vmax = float(gray_lin.min()), float(gray_lin.max())
        if vmax - vmin > 1e-12:
            gray_lin = (gray_lin - vmin) / (vmax - vmin)

        # 8) 椒盐噪声
        gray_lin = self.salt_pepper_degradation(gray_lin)
 
        # 9) 伽马
        gray = self.delinearize_image(gray_lin) if apply_gamma_correction else gray_lin.astype(np.float32)
        return np.clip(gray, 0.0, 1.0).astype(np.float32)


