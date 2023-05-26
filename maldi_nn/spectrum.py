from scipy.signal import savgol_filter
from scipy import sparse
from scipy.linalg import norm
import pandas as pd
import numpy as np
from scipy.stats import binned_statistic
from maldi_nn import topf


class SpectrumObject:
    def __init__(self, file=None, mz=None, intensity=None):
        if file is not None:
            s = pd.read_table(
                file, sep=" ", index_col=None, comment="#", header=None
            ).values
            self.mz = s[:, 0]
            self.intensity = s[:, 1]
        else:
            self.mz = mz
            self.intensity = intensity
        if self.intensity is not None:
            if np.issubdtype(self.intensity.dtype, np.unsignedinteger):
                self.intensity = self.intensity.astype(int)
        if self.mz is not None:
            if np.issubdtype(self.mz.dtype, np.unsignedinteger):
                self.mz = self.mz.astype(int)

    def __getitem__(self, index):
        return SpectrumObject(mz=self.mz[index], intensity=self.intensity[index])

    def __len__(self):
        if self.mz is not None:
            return self.mz.shape[0]
        else:
            return 0


class Binner:
    def __init__(self, start=2000, stop=20000, step=3, aggregation="sum"):
        self.bins = np.arange(start, stop + 1e-8, step)
        self.mz_bins = self.bins[:-1] + step / 2
        self.agg = aggregation

    def __call__(self, SpectrumObj):
        if self.agg == "sum":
            bins, _ = np.histogram(
                SpectrumObj.mz, self.bins, weights=SpectrumObj.intensity
            )
        else:
            bins = binned_statistic(
                SpectrumObj.mz,
                SpectrumObj.intensity,
                bins=self.bins,
                statistic=self.agg,
            ).statistic
            bins = np.nan_to_num(bins)

        s = SpectrumObject(intensity=bins, mz=self.mz_bins)
        return s
    
class EqualFreqBinner:
    def __init__(self, n_intervals):
        self.n = n_intervals
        
    def __call__(self, SpectrumObj):
        n_per = len(SpectrumObj.mz) // self.n

        intensity = SpectrumObj.intensity[:(n_per * self.n)].reshape(-1, n_per).mean(1)
        mz = SpectrumObj.mz[:(n_per * self.n)].reshape(-1, n_per).mean(1)

        s = SpectrumObject(
            intensity= intensity,
            mz=mz
        )
        return s

class Normalizer:
    def __init__(self, sum=1):
        self.sum = sum

    def __call__(self, SpectrumObj):

        s = SpectrumObject()

        s = SpectrumObject(
            intensity=SpectrumObj.intensity / SpectrumObj.intensity.sum() * self.sum,
            mz=SpectrumObj.mz,
        )
        return s


class Trimmer:
    def __init__(self, min=2000, max=20000):
        self.range = [min, max]

    def __call__(self, SpectrumObj):
        indices = (2000 < SpectrumObj.mz) & (SpectrumObj.mz < 20000)

        s = SpectrumObject(
            intensity=SpectrumObj.intensity[indices], mz=SpectrumObj.mz[indices]
        )
        return s


class VarStabilizer:
    def __init__(self, method="sqrt"):
        methods = {"sqrt": np.sqrt, "log": np.log, "log2": np.log2, "log10": np.log10}
        self.fun = methods[method]

    def __call__(self, SpectrumObj):
        s = SpectrumObject(intensity=self.fun(SpectrumObj.intensity), mz=SpectrumObj.mz)
        return s


class BaselineCorrecter:
    def __init__(
        self,
        method=None,
        als_lam=1e8,
        als_p=0.01,
        als_max_iter=10,
        als_tol=1e-6,
        snip_n_iter=10,
    ):
        self.method = method
        self.lam = als_lam
        self.p = als_p
        self.max_iter = als_max_iter
        self.tol = als_tol
        self.n_iter = snip_n_iter

    def __call__(self, SpectrumObj):
        if "LS" in self.method:
            baseline = self.als(
                SpectrumObj.intensity,
                method=self.method,
                lam=self.lam,
                p=self.p,
                max_iter=self.max_iter,
                tol=self.tol,
            )
        elif self.method == "SNIP":
            baseline = self.snip(SpectrumObj.intensity, self.n_iter)

        s = SpectrumObject(
            intensity=SpectrumObj.intensity - baseline, mz=SpectrumObj.mz
        )
        return s

    def als(self, y, method="ArPLS", lam=1e8, p=0.01, max_iter=10, tol=1e-6):
        L = len(y)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
        D = lam * D.dot(
            D.transpose()
        )  # Precompute this term since it does not depend on `w`

        w = np.ones(L)
        W = sparse.spdiags(w, 0, L, L)

        crit = 1
        count = 0
        while crit > tol:
            z = sparse.linalg.spsolve(W + D, w * y)

            if method == "AsLS":
                w_new = p * (y > z) + (1 - p) * (y < z)
            elif method == "ArPLS":
                d = y - z
                dn = d[d < 0]
                m = np.mean(dn)
                s = np.std(dn)
                w_new = 1 / (1 + np.exp(np.minimum(2 * (d - (2 * s - m)) / s, 70)))

            crit = norm(w_new - w) / norm(w)
            w = w_new
            W.setdiag(w)
            count += 1
            if count > max_iter:
                break
        return z

    def snip(self, y, n_iter):
        y_prepr = np.log(np.log(np.sqrt(y + 1) + 1) + 1)
        for i in range(1, n_iter + 1):
            rolled = np.pad(y_prepr, (i, i), mode="edge")
            new = np.minimum(
                y_prepr, (np.roll(rolled, i) + np.roll(rolled, -i))[i:-i] / 2
            )
            y_prepr = new
        return (np.exp(np.exp(y_prepr) - 1) - 1) ** 2 - 1


class Smoother:
    def __init__(self, halfwindow=10, polyorder=3):
        self.window = halfwindow * 2 + 1
        self.poly = polyorder

    def __call__(self, SpectrumObj):
        s = SpectrumObject(
            intensity=np.maximum(savgol_filter(SpectrumObj.intensity, self.window, self.poly), 0),
            mz=SpectrumObj.mz,
        )
        return s


class PersistenceTransformer:
    def __init__(self, extract_nonzero=False):
        self.filter = extract_nonzero

    def __call__(self, SpectrumObj):
        a = np.stack([SpectrumObj.mz, SpectrumObj.intensity]).T
        b = topf.PersistenceTransformer().fit_transform(a)

        s = SpectrumObject()
        if self.filter:
            peaks = b[:, 1] != 0
            s = SpectrumObject(intensity=b[peaks, 1], mz=b[peaks, 0])
        else:
            s = SpectrumObject(intensity=b[:, 1], mz=b[:, 0])
        return s


class PeakFilter:
    def __init__(self, max_number=None, min_intensity=None):
        self.max_number = max_number
        self.min_intensity = min_intensity

    def __call__(self, SpectrumObj):
        s = SpectrumObject(intensity=SpectrumObj.intensity, mz=SpectrumObj.mz)

        if self.max_number is not None:
            indices = np.argsort(-s.intensity, kind="stable")

            take = np.sort(indices[: self.max_number])

            s.mz = s.mz[take]
            s.intensity = s.intensity[take]

        if self.min_intensity is not None:
            take = s.intensity >= self.min_intensity

            s.mz = s.mz[take]
            s.intensity = s.intensity[take]

        return s


class RandomPeakShifter:
    def __init__(self, std=1):
        self.std = std

    def __call__(self, SpectrumObj):
        s = SpectrumObject(
            intensity=SpectrumObj.intensity,
            mz=SpectrumObj.mz
            + np.random.normal(scale=self.std, size=SpectrumObj.mz.shape),
        )
        return s

class UniformPeakShifter:
    def __init__(self, range=1.5):
        self.range = range

    def __call__(self, SpectrumObj):
        s = SpectrumObject(
            intensity=SpectrumObj.intensity,
            mz=SpectrumObj.mz
            + np.random.uniform(low = -self.range, high = self.range, size=SpectrumObj.mz.shape),
        )
        return s

class Binarizer:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, SpectrumObj):
        s = SpectrumObject(
            intensity=(SpectrumObj.intensity > self.threshold).astype(SpectrumObj.intensity.dtype),
            mz=SpectrumObj.mz
        )
        return s


class SequentialPreprocessor:
    def __init__(self, *args):
        self.preprocessors = args

    def __call__(self, SpectrumObj):
        for step in self.preprocessors:
            SpectrumObj = step(SpectrumObj)
        return SpectrumObj
