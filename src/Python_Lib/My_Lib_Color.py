# -*- coding: utf-8 -*-
__author__ = 'LiYuanhe'

import subprocess
import numpy as np
import colour

from Python_Lib.My_Lib_Stock import *


# ===========================================================================
#  CIE data from colour-science (backward-compatible aliases)
# ===========================================================================
_CIE_CMFS = colour.colorimetry.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
_D65_SD   = colour.SDS_ILLUMINANTS["D65"]

# Kept for backward compatibility — now derived from colour-science data
CIE1931_CMF_WL_5NM   = np.arange(360, 785, 5, dtype=float)
CIE1931_CMF_XBAR_5NM = _CIE_CMFS[CIE1931_CMF_WL_5NM, 0]
CIE1931_CMF_YBAR_5NM = _CIE_CMFS[CIE1931_CMF_WL_5NM, 1]
CIE1931_CMF_ZBAR_5NM = _CIE_CMFS[CIE1931_CMF_WL_5NM, 2]
D65_WL_5NM  = CIE1931_CMF_WL_5NM
D65_SPD_5NM = _D65_SD[CIE1931_CMF_WL_5NM]


# ===========================================================================
#  CIE L*a*b* conversion utilities
# ===========================================================================

def f_lab(t):
    """CIE L*a*b* nonlinear compression function (vectorized).

    Kept for backward compatibility.
    """
    delta = 6.0 / 29.0
    return np.where(t > delta ** 3,
                    np.cbrt(t),
                    t / (3.0 * delta ** 2) + 4.0 / 29.0)


def XYZ_to_Lab(X, Y, Z, Xn, Yn, Zn):
    """Convert CIE XYZ tristimulus values to L*a*b*.

    Delegates to ``colour.XYZ_to_Lab``.

    Parameters
    ----------
    X, Y, Z : float or ndarray
        Tristimulus values.
    Xn, Yn, Zn : float
        Reference white tristimulus values.

    Returns
    -------
    L, a_star, b_star : float or ndarray
    """
    illuminant_xy = np.array([Xn, Yn]) / (Xn + Yn + Zn)
    XYZ_scaled = np.array([X, Y, Z], dtype=float) / float(Yn)
    Lab = colour.XYZ_to_Lab(XYZ_scaled, illuminant_xy)
    return float(Lab[0]), float(Lab[1]), float(Lab[2])


def Lab_to_hue_angle(a_star, b_star):
    """Compute hue angle h_ab = atan2(b*, a*) in degrees."""
    return float(np.degrees(np.arctan2(b_star, a_star)))


class HueAngleEvaluator:
    """
    Converts UV-Vis absorption spectra to CIE L*a*b* hue angle.

    The absorption spectrum is treated as absorbance; transmission outside
    the measurement range is assumed to be 1 (fully transparent).
    Only the overlap region with CIE CMFs (typically 360–780 nm) contributes
    to perceived color.

    Periodicity handling
    --------------------
    A reference center hue angle is computed from training data via circular
    mean.  All output hue angles are shifted to lie within ±180° of this
    center, ensuring both predicted and experimental values use a consistent
    branch and can be plotted on a standard linear axis.

    Usage with Train_NN_Network
    ---------------------------
    ::

        evaluator = HueAngleEvaluator(WAVELENGTH_START, WAVELENGTH_END)
        evaluator.set_center_from_data(Y_train)
        Train_NN_Network(..., function_for_numerical_evaluation=evaluator, ...)
    """

    def __init__(self, wavelength_start: int = 230, wavelength_end: int = 400):
        self.wl_start = wavelength_start
        self.wl_end = wavelength_end
        self.spectrum_length = wavelength_end - wavelength_start + 1
        self.center_hue = None

        # CMFs and D65 at native 1 nm resolution from colour-science
        wl_1nm = np.arange(360.0, 781.0, 1.0)  # 421 points
        self.xbar = _CIE_CMFS[wl_1nm, 0]
        self.ybar = _CIE_CMFS[wl_1nm, 1]
        self.zbar = _CIE_CMFS[wl_1nm, 2]
        self.d65  = _D65_SD[wl_1nm]
        self.wl_1nm = wl_1nm

        # Pre-compute reference white  (D65, T = 1 everywhere)
        self.Xn = np.sum(self.d65 * self.xbar)
        self.Yn = np.sum(self.d65 * self.ybar)
        self.Zn = np.sum(self.d65 * self.zbar)

        # Indices in the 1 nm grid [360..780] that overlap with the spectrum
        overlap_start = max(360, wavelength_start)
        overlap_end   = min(780, wavelength_end)
        if overlap_start > overlap_end:
            self._overlap_grid_idx = np.array([], dtype=int)
            self._overlap_spec_idx = np.array([], dtype=int)
        else:
            grid_idx_start = int(overlap_start - 360)
            grid_idx_end   = int(overlap_end - 360)
            spec_idx_start = int(overlap_start - wavelength_start)
            spec_idx_end   = int(overlap_end - wavelength_start)
            self._overlap_grid_idx = np.arange(grid_idx_start, grid_idx_end + 1)
            self._overlap_spec_idx = np.arange(spec_idx_start, spec_idx_end + 1)

    # ------------------------------------------------------------------

    def _spectrum_to_hue(self, absorbance: np.ndarray) -> float:
        """
        Single spectrum (1-D array of length spectrum_length) → hue angle in degrees.

        Treats the spectrum values as absorbance:  T(λ) = 10^{−A(λ)}.
        Outside the measured range, T = 1.
        """
        A_overlap = absorbance[self._overlap_spec_idx]
        A_overlap = np.maximum(A_overlap, 0.0)
        delta_T = 1.0 - np.power(10.0, -A_overlap)

        overlap_grid_indices = self._overlap_grid_idx
        dX = np.sum(self.d65[overlap_grid_indices] * delta_T * self.xbar[overlap_grid_indices])
        dY = np.sum(self.d65[overlap_grid_indices] * delta_T * self.ybar[overlap_grid_indices])
        dZ = np.sum(self.d65[overlap_grid_indices] * delta_T * self.zbar[overlap_grid_indices])

        X = self.Xn - dX
        Y = self.Yn - dY
        Z = self.Zn - dZ

        _, a_star, b_star = XYZ_to_Lab(X, Y, Z, self.Xn, self.Yn, self.Zn)
        return Lab_to_hue_angle(a_star, b_star)

    def compute_hue_angles(self, Ys: np.ndarray) -> list:
        """Batch of spectra (N, spectrum_length) → list of hue angles (degrees)."""
        return [self._spectrum_to_hue(Ys[i]) for i in range(len(Ys))]

    def set_center_from_data(self, Y_data):
        """
        Compute the circular-mean hue angle from a set of spectra and store
        it as the periodicity reference center.

        Call this once with Y_train before training starts.
        """
        try:
            import torch
            if isinstance(Y_data, torch.Tensor):
                Y_data = Y_data.detach().cpu().numpy()
        except ImportError:
            pass
        hues = np.array(self.compute_hue_angles(Y_data))
        rad = np.deg2rad(hues)
        center_rad = np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))
        self.center_hue = np.degrees(center_rad)
        print(f"HueAngleEvaluator: center hue = {self.center_hue:.2f}°")

    def _adjust_periodicity(self, h: float) -> float:
        """Shift h to be within ±180° of self.center_hue."""
        if self.center_hue is None:
            return h
        diff = h - self.center_hue
        h -= 360.0 * round(diff / 360.0)
        return h

    def __call__(self, Ys):
        """
        Called by Train_NN_Network.function_for_numerical_evaluation.

        Args:
            Ys: (batch, spectrum_length) tensor or ndarray.

        Returns:
            list of hue angles (degrees), adjusted for periodicity.
        """
        try:
            import torch
            if isinstance(Ys, torch.Tensor):
                Ys = Ys.detach().cpu().numpy()
        except ImportError:
            pass
        hues = self.compute_hue_angles(Ys)
        return [self._adjust_periodicity(h) for h in hues]


class BandIntegralEvaluator:
    """
    Splits a spectrum into *n* equal-width bands and returns the ratio of
    adjacent bands' integrals (band[i+1] / band[i]).

    For *num_bands* bands this produces *num_bands − 1* ratio values per
    spectrum, giving a shape-only descriptor (total intensity is
    normalized away).

    Usage with Train_NN_Network
    ---------------------------
    ::

        evaluator = BandIntegralEvaluator(WAVELENGTH_START, WAVELENGTH_END, num_bands=3)
        Train_NN_Network(..., function_for_numerical_evaluation=evaluator, ...)

    This produces *num_bands − 1* separate y = x scatter plots.
    """

    def __init__(self, wavelength_start, wavelength_end,
                 num_bands: int = 3):
        self.wl_start = wavelength_start
        self.wl_end = wavelength_end
        self.spectrum_length = wavelength_end - wavelength_start + 1
        self.num_bands = num_bands

        # Compute band boundaries (indices into the spectrum array)
        points_per_band = self.spectrum_length / num_bands
        self.band_slices = []
        self.labels = []
        for i in range(num_bands):
            start_idx = int(round(i * points_per_band))
            end_idx = int(round((i + 1) * points_per_band))
            self.band_slices.append((start_idx, end_idx))
            wl_lo = wavelength_start + start_idx
            wl_hi = wavelength_start + end_idx - 1
        for i in range(num_bands - 1):
            lo1 = wavelength_start + self.band_slices[i][0]
            hi1 = wavelength_start + self.band_slices[i][1] - 1
            lo2 = wavelength_start + self.band_slices[i + 1][0]
            hi2 = wavelength_start + self.band_slices[i + 1][1] - 1
            self.labels.append(f"Band {lo2}-{hi2} / {lo1}-{hi1} nm")

    def __call__(self, Ys):
        """
        Args:
            Ys: (N, spectrum_length) tensor or ndarray.

        Returns:
            list of lists, shape (N, num_bands-1) — ratio of adjacent band integrals.
        """
        try:
            import torch
            if isinstance(Ys, torch.Tensor):
                Ys = Ys.detach().cpu().numpy()
        except ImportError:
            pass
        Ys = np.array(Ys)
        if Ys.ndim == 1:
            Ys = Ys[np.newaxis, :]

        results = []
        for spectrum in Ys:
            band_integrals = []
            for start_idx, end_idx in self.band_slices:
                band_integrals.append(np.sum(np.abs(spectrum[start_idx:end_idx])))
            ratios = []
            for i in range(self.num_bands - 1):
                denom = band_integrals[i]
                if denom < 1e-12:
                    ratios.append(1.0)
                else:
                    ratios.append(band_integrals[i + 1] / denom)
            results.append(ratios)
        return results


# ===========================================================================
#  Spectrum ↔ CIE 1931 XYZ ↔ sRGB conversion
# ===========================================================================

def normalize_spectrum(spec_X, spec_Y, method="none", clip_negative=True):
    """Normalize a spectrum before color conversion.

    Parameters
    ----------
    spec_X : array-like
        Wavelengths in nm.
    spec_Y : array-like
        Spectral values.
    method : {"none", "max", "area", "l2", "percentile95"}
        Normalization strategy.
    clip_negative : bool
        If True, negative values are clipped to 0 before normalization.

    Returns
    -------
    ndarray
        Normalized spectral values.
    """
    spec_X = np.asarray(spec_X, dtype=float)
    values = np.asarray(spec_Y, dtype=float)

    if clip_negative:
        values = np.clip(values, 0.0, None)

    method = str(method).lower()
    if method == "none":
        return values

    if method == "max":
        scale = float(np.max(values))
    elif method == "area":
        scale = float(np.trapezoid(values, spec_X))
    elif method == "l2":
        scale = float(np.linalg.norm(values))
    elif method == "percentile95":
        scale = float(np.percentile(values, 95))
    else:
        raise ValueError(
            "method must be one of 'none', 'max', 'area', 'l2', 'percentile95', "
            f"got '{method}'"
        )

    if scale <= 0.0:
        return values
    return values / scale


def spec_to_CIE_1931_XYZ(
    spec_X,
    spec_Y,
    mode="absorption",
    normalize=True,
    spectrum_normalization="none",
    absorption_output="transmitted_light",
):
    """Convert a measured spectrum to CIE 1931 XYZ tristimulus values.

    Delegates to ``colour.sd_to_XYZ`` for the core integration.  The input
    spectrum is represented as a ``colour.SpectralDistribution``; outside
    the measured range, emission defaults to 0 and absorption defaults to
    transmittance = 1 (fully transparent).

    Parameters
    ----------
    spec_X : array-like
        Wavelengths in nm (must be sorted ascending).
    spec_Y : array-like
        Spectral values.

        - mode="absorption": absorbance *A*(λ).  Transmittance is computed
          as T(λ) = 10^{−A(λ)}.  Outside the measured range T = 1 (fully
          transparent).
        - mode="emission": spectral power distribution S(λ).  Outside the
          measured range S = 0.
    mode : {"absorption", "emission"}
        "absorption" — D65 white light through a filter / sample.
        "emission"   — self-luminous source.
    normalize : bool
        If True (default), apply the CIE normalization factor
        k = 100 / ∫ illuminant(λ) ȳ(λ) dλ so that Y = 100 for the
        reference white (D65 for absorption, or self-luminous peak for
        emission).  If False, return raw integration values (useful for
        comparing multiple emission spectra on a common intensity scale).
    spectrum_normalization : {"none", "max", "area", "l2", "percentile95"}
        Normalize the input spectrum before tristimulus integration.
    absorption_output : {"transmitted_light", "absorbed_light"}
        Only used when ``mode='absorption'``.

        - ``transmitted_light``: evaluate the transmitted D65 light.
        - ``absorbed_light``: evaluate the D65-weighted absorbed portion,
          which is often more useful when comparing spectra by color spread.

    Returns
    -------
    X, Y, Z : float
    """
    spec_X = np.asarray(spec_X, dtype=float)
    spec_Y = normalize_spectrum(spec_X, spec_Y, method=spectrum_normalization, clip_negative=True)

    # Build a 1 nm grid covering the full CIE CMF range
    wl = np.arange(360.0, 781.0, 1.0)

    if mode == "emission":
        S = np.zeros_like(wl)
        mask = (wl >= spec_X[0]) & (wl <= spec_X[-1])
        S[mask] = np.interp(wl[mask], spec_X, spec_Y)
        S = np.maximum(S, 0.0)

        sd = colour.SpectralDistribution(dict(zip(wl, S)))
        k_val = None if normalize else 1
        XYZ = colour.sd_to_XYZ(sd, k=k_val)
        X, Y, Z = float(XYZ[0]), float(XYZ[1]), float(XYZ[2])

    elif mode == "absorption":
        T = np.ones_like(wl)
        mask = (wl >= spec_X[0]) & (wl <= spec_X[-1])
        A_interp = np.interp(wl[mask], spec_X, spec_Y)
        T[mask] = np.power(10.0, -np.maximum(A_interp, 0.0))

        D65 = colour.SDS_ILLUMINANTS["D65"]
        k_val = None if normalize else 1
        if absorption_output == "transmitted_light":
            sd_T = colour.SpectralDistribution(dict(zip(wl, T)))
            XYZ = colour.sd_to_XYZ(sd_T, illuminant=D65, k=k_val)
        elif absorption_output == "absorbed_light":
            absorbed_spd = (1.0 - T) * D65[wl]
            sd_absorbed = colour.SpectralDistribution(dict(zip(wl, absorbed_spd)))
            XYZ = colour.sd_to_XYZ(sd_absorbed, k=k_val)
        else:
            raise ValueError(
                "absorption_output must be 'transmitted_light' or 'absorbed_light', "
                f"got '{absorption_output}'"
            )
        X, Y, Z = float(XYZ[0]), float(XYZ[1]), float(XYZ[2])

    else:
        raise ValueError(f"mode must be 'absorption' or 'emission', got '{mode}'")

    return X, Y, Z


def CIE_1931_XYZ_to_RGB(X, Y, Z, brightness=1.0, normalize_brightness=True, xyz_reference_y=100.0):
    """Convert CIE 1931 XYZ tristimulus values to sRGB (0–255).

    XYZ 的绝对尺度被丢弃——只保留色度 (X:Y:Z 的比例)，输出整体亮度
    由 ``brightness`` 参数控制，其定义为 HSV 空间中的 V 分量
    (= ``max(R, G, B) / 255``，在 gamma 编码后的 sRGB 上计算)。

    Parameters
    ----------
    X, Y, Z : float
        Tristimulus values, typically from :func:`spec_to_CIE_1931_XYZ`.
        Only the *ratios* matter; the absolute scale is discarded.
    brightness : float in [0, 1]
        输出颜色的目标 HSV V 值。

        - 1.0 (默认): 该色度下最饱和的渲染——峰值通道正好等于 255
        - 0.5: 峰值通道等于 128 (= ``round(0.5 * 255)``)，整体更暗但
          色度保持不变
        - 0.0: 黑色

        缩放在线性 RGB 空间进行（以保持色度），目标线性峰值取
        ``eotf_sRGB(brightness)``，这样经过 sRGB gamma 编码后最大
        通道恰好落在 ``brightness`` 上。
    normalize_brightness : bool
        If True, rescale RGB so the peak channel is fixed by ``brightness``.
        If False, keep the XYZ-derived luminance and treat ``brightness`` as a
        direct linear gain factor.
    xyz_reference_y : float
        Reference Y value corresponding to diffuse white. CIE-normalized XYZ
        typically uses 100.

    Returns
    -------
    (R, G, B) : tuple of int
        sRGB values in [0, 255], with ``max(R, G, B) == round(brightness * 255)``
        when the chromaticity is in-gamut.
    """
    XYZ_arr = np.array([X, Y, Z], dtype=float) / float(xyz_reference_y)
    brightness = float(np.clip(brightness, 0.0, 1.0))

    # Linear RGB (no gamma). Out-of-gamut colours may yield negatives; clip
    # them to zero so clipping desaturates toward the nearest in-gamut hue.
    linear = colour.XYZ_to_sRGB(XYZ_arr, apply_cctf_encoding=False)
    linear = np.clip(linear, 0.0, None)

    if brightness <= 0.0:
        return (0, 0, 0)

    peak = float(np.max(linear))
    if peak <= 0.0:
        return (0, 0, 0)

    if normalize_brightness:
        target_linear = float(colour.models.eotf_sRGB(brightness))
        linear_scaled = (linear / peak) * target_linear
    else:
        linear_scaled = linear * brightness

    srgb = colour.models.eotf_inverse_sRGB(linear_scaled)
    srgb = np.clip(srgb, 0.0, 1.0)
    return tuple(int(round(c * 255)) for c in srgb)


def map_spec_range(input_X_refs=(230, 400), output_X_refs=(380, 750)):
    """Return a linear mapping function from one wavelength range to another.

    This enables applying visible-light colour analysis to spectra outside
    the visible range by linearly stretching / shifting the wavelength axis.

    Parameters
    ----------
    input_X_refs : (float, float)
        Source wavelength range (lo, hi) in nm.
    output_X_refs : (float, float)
        Target wavelength range (lo, hi) in nm.  Default (380, 750) roughly
        covers the visible spectrum.

    Returns
    -------
    mapper : callable
        ``mapper(spec_X) → mapped_X`` (accepts and returns ndarray).

    Example
    -------
    >>> mapper = map_spec_range(input_X_refs=(230, 400), output_X_refs=(380, 750))
    >>> mapped_X = mapper(original_wavelengths)
    >>> X, Y, Z = Spec_to_CIE_1931_XYZ(mapped_X, spec_Y, mode="emission")
    """
    in_lo, in_hi = input_X_refs
    out_lo, out_hi = output_X_refs
    scale = (out_hi - out_lo) / (in_hi - in_lo)
    offset = out_lo - in_lo * scale

    def mapper(x):
        return np.asarray(x, dtype=float) * scale + offset

    return mapper


# ===========================================================================
#  Color mixing utilities
# ===========================================================================

def mix_two_colors(color1, color2, percentage, mixing_order=2):
    """
    :param color1: color in hex, e.g. "00FF00"
    :param color2: color in hex, e.g. "00FF00"
    :param mixing_order: numeric order, 'hsv', or 'lab'
        - numeric: power-mean mixing of each sRGB channel.
        - 'hsv': interpolate hue along the shorter arc, plus s/v linearly.
        - 'lab': linear interpolation in CIE L*a*b* under D65. When the two
          input colours share L*, the mixed colour also has that L* — useful
          for palettes built on top of ``unify_perceptual_lightness``.
    :param percentage: if =0, color1, if =1, color2, else, return an interpolation of the color
    :return: a mixed color in hex number
    """
    if mixing_order == 'hsv':
        import colorsys
    if percentage == 0:
        return color1
    if percentage == 1:
        return color2

    color1 = [eval('0x' + x) for x in [color1[0:2], color1[2:4], color1[4:6]]]
    color2 = [eval('0x' + x) for x in [color2[0:2], color2[2:4], color2[4:6]]]

    def mixing(value1, value2, mixing_percentage, order):
        return round((value1 ** order + (value2 ** order - value1 ** order) * mixing_percentage) ** (1 / order))

    def lab_mixing(rgb1, rgb2, lab_percentage):
        # Convert sRGB 0-255 → 0-1 → XYZ → L*a*b*; interpolate linearly;
        # convert back to sRGB. Linear in L*a*b* keeps perceived lightness
        # smooth — and *constant* when the two endpoints share L*.
        rgb_arr = np.array([[rgb1[0] / 255.0, rgb1[1] / 255.0, rgb1[2] / 255.0],
                            [rgb2[0] / 255.0, rgb2[1] / 255.0, rgb2[2] / 255.0]],
                           dtype=float)
        XYZ = colour.sRGB_to_XYZ(rgb_arr)
        Lab = colour.XYZ_to_Lab(XYZ)
        Lab_mid = Lab[0] + lab_percentage * (Lab[1] - Lab[0])
        XYZ_mid = colour.Lab_to_XYZ(Lab_mid)
        rgb_mid = np.clip(colour.XYZ_to_sRGB(XYZ_mid), 0.0, 1.0)
        return (round(float(rgb_mid[0]) * 255),
                round(float(rgb_mid[1]) * 255),
                round(float(rgb_mid[2]) * 255))

    def hsv_mixing(rgb1, rgb2, hsv_percentage):
        # Convert sRGB 0-255 -> 0-1, then to HSV.
        h1, s1, v1 = colorsys.rgb_to_hsv(*[x / 255 for x in rgb1])
        h2, s2, v2 = colorsys.rgb_to_hsv(*[x / 255 for x in rgb2])

        # Hue is undefined when saturation is 0 (white/black/gray). colorsys
        # returns 0 in that case, which would pull the interpolated hue toward
        # red (h=0) and produce a rainbow streak. Borrow the other colour's
        # hue when one side is achromatic.
        if s1 == 0 and s2 != 0:
            h1 = h2
        if s2 == 0 and s1 != 0:
            h2 = h1

        # Interpolate hue along the SHORTER arc on the [0, 1) hue circle.
        # Plain (1-t)*h1 + t*h2 always crosses the long way when |h2-h1| > 0.5,
        # producing an unwanted rainbow when wrapping ±180°.
        dh = h2 - h1
        if dh > 0.5:
            dh -= 1.0
        elif dh < -0.5:
            dh += 1.0
        h = (h1 + hsv_percentage * dh) % 1.0

        s = s1 + hsv_percentage * (s2 - s1)
        v = v1 + hsv_percentage * (v2 - v1)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return (round(r * 255), round(g * 255), round(b * 255))

    if mixing_order == 'hsv':
        ret = hsv_mixing(color1, color2, percentage)
    elif mixing_order == 'lab':
        ret = lab_mixing(color1, color2, percentage)
    else:
        ret = [mixing(color1[x], color2[x], percentage, mixing_order) for x in range(3)]

    ret = ''.join('{:02X}'.format(int(round(num))) for num in ret)
    return ret


def _hex_to_rgb01(hex_str):
    s = hex_str.lstrip('#')
    return (int(s[0:2], 16) / 255.0, int(s[2:4], 16) / 255.0, int(s[4:6], 16) / 255.0)


def _rgb01_to_hex(r, g, b):
    r = max(0.0, min(1.0, r))
    g = max(0.0, min(1.0, g))
    b = max(0.0, min(1.0, b))
    return '{:02X}{:02X}{:02X}'.format(int(round(r * 255)),
                                       int(round(g * 255)),
                                       int(round(b * 255)))


def unify_perceptual_lightness(colors, set_lightness=None, verbose=True):
    """Equalise perceived lightness across a sequence of colours.

    Each colour is converted to CIE L*a*b*; their L* values are replaced by
    the target L* (mean of the operated set, or ``set_lightness`` if given),
    while a* and b* (chromaticity) are kept untouched. The resulting L*a*b*
    triples are then converted back to sRGB hex strings.

    Parameters
    ----------
    colors : sequence of str
        Hex colours, with or without leading ``#``.
    set_lightness : float or None
        If given, use this value as the target CIE L* instead of the mean
        of the operated colours' L* values.  Useful to force a specific
        perceptual brightness level.
    verbose : bool
        If True (default), print each colour's before/after hex and
        L*, a*, b* values together with the target L*.

    Returns
    -------
    list of str
        Hex colours (no leading ``#``), same length as ``colors``.
    """
    if not colors:
        return []
    hex_clean = [c.lstrip('#').upper() for c in colors]

    operated_indices = list(range(len(hex_clean)))

    rgb_arr = np.array([_hex_to_rgb01(hex_clean[i]) for i in operated_indices],
                       dtype=float)
    # colour-science expects sRGB in [0, 1]; XYZ from sRGB uses D65 by default.
    XYZ = colour.sRGB_to_XYZ(rgb_arr)
    Lab = colour.XYZ_to_Lab(XYZ)            # shape (N, 3): L*, a*, b*

    if set_lightness is not None:
        target_L = float(set_lightness)
    else:
        target_L = float(np.mean(Lab[:, 0]))

    if verbose:
        print(f"unify_perceptual_lightness: target L* = {target_L:.2f}")

    Lab_new = Lab.copy()
    Lab_new[:, 0] = target_L
    XYZ_back = colour.Lab_to_XYZ(Lab_new)
    rgb_back = colour.XYZ_to_sRGB(XYZ_back)
    rgb_back = np.clip(rgb_back, 0.0, 1.0)

    out = list(hex_clean)
    for k, i in enumerate(operated_indices):
        r, g, b = rgb_back[k]
        new_hex = _rgb01_to_hex(r, g, b)
        if verbose:
            L0, a0, b0 = float(Lab[k, 0]), float(Lab[k, 1]), float(Lab[k, 2])
            # Recompute L* of the clipped output to reflect any gamut clipping.
            Lab_out = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.array([r, g, b], dtype=float)))
            L1, a1, b1 = float(Lab_out[0]), float(Lab_out[1]), float(Lab_out[2])
            print(f"  [{i}] #{hex_clean[i]} (L*={L0:.2f}, a*={a0:.2f}, b*={b0:.2f})"
                  f"  →  #{new_hex} (L*={L1:.2f}, a*={a1:.2f}, b*={b1:.2f})")
        out[i] = new_hex
    return out


def color_scale(colors, ref_points, value, mixing_order=2):
    """
    generate a color scale, extract color for a value
    :param colors: list of colors corresponds to list of ref_points, color in Hex like FFFFFF
    :param ref_points:
    :param value:
    :param mixing_order:
    :return:
    """

    assert sorted(ref_points) == ref_points or sorted(ref_points, reverse=True) == ref_points, 'Color Ref Points need to be in sequence.'
    if sorted(ref_points, reverse=True) == ref_points:
        ref_points = list(reversed(ref_points))
        colors = list(reversed(colors))

    if value < ref_points[0]:
        return colors[0]
    if value > ref_points[-1]:
        return colors[-1]

    for value_count in range(len(ref_points) - 1):
        value1, value2 = ref_points[value_count:value_count + 2]
        if value1 <= value <= value2:
            color1, color2 = colors[value_count:value_count + 2]
            return mix_two_colors(color1, color2, (value - value1) / (value2 - value1), mixing_order=mixing_order)


if __name__ == '__main__':
    pass
