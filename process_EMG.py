"""
EMG処理スクリプト (process_EMG.m のPython移植版)

処理フロー:
  1. バンドパスフィルタ (Butterworth 4次, 40–400 Hz)
  2. 基線合わせ (平均除去)
  3. 整流化 (絶対値)
  4. RMSスムージング (100 サンプル窓)

ファイルはアルファベットフォルダ内の OpenSignals .txt を
タイムスタンプ順に並べ、QWERTY 順でキーに対応付ける。
"""

from pathlib import Path
import json
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt

# ── 設定 ────────────────────────────────────────────────────────────────────
FS = 1000          # サンプリング周波数 [Hz]
MUS_NUM = 2        # 筋チャンネル数
T_RMS = 100        # RMS窓幅 [サンプル]
BP_LOW = 40        # バンドパス下限 [Hz]
BP_HIGH = 400      # バンドパス上限 [Hz]
BP_ORDER = 4       # バターワース次数

QWERTY_KEYS = list("qwertyuiopasdfghjklzxcvbnm")

ALPHABET_DIR = Path(__file__).parent / "alphabet"
FIGURES_DIR  = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
DEVICE_ID = "000780897FDB"   # 対象デバイスID


# ── データ読み込み ──────────────────────────────────────────────────────────
def read_opensignals(filepath: Path) -> np.ndarray:
    """OpenSignals txt を読み込み EMG 生データ (N x MUS_NUM) を返す"""
    with open(filepath) as f:
        lines = f.readlines()

    # "# EndOfHeader" の次行からデータ開始
    data_start = next(
        i + 1 for i, line in enumerate(lines) if "EndOfHeader" in line
    )

    data = np.array(
        [line.split() for line in lines[data_start:] if line.strip()],
        dtype=float,
    )
    # 列: nSeq, DI, CH1, CH2  → CH1, CH2 を返す
    return data[:, 2 : 2 + MUS_NUM]


# ── フィルタリング ───────────────────────────────────────────────────────────
def filter_emg(emg_raw: np.ndarray, fs: int = FS) -> np.ndarray:
    """MATLAB filter_EMG と同等の処理"""
    # 1. バンドパス (zero-phase)
    b, a = butter(BP_ORDER, [BP_LOW / (fs / 2), BP_HIGH / (fs / 2)], btype="bandpass")
    emg_band = filtfilt(b, a, emg_raw, axis=0)

    # 2. 基線合わせ
    emg_center = emg_band - emg_band.mean(axis=0)

    # 3. 整流化
    emg_rect = np.abs(emg_center)

    # 4. RMS スムージング (uniform_filter1d で sliding window の二乗平均 → sqrt)
    emg_low = np.sqrt(uniform_filter1d(emg_rect**2, size=T_RMS, axis=0, mode="nearest"))

    return emg_low


# ── ファイル一覧とキー対応 ──────────────────────────────────────────────────
def get_key_file_pairs() -> list[tuple[str, Path]]:
    """タイムスタンプ昇順でファイルをソートし QWERTY キーを割り当てる"""
    files = sorted(
        f for f in ALPHABET_DIR.glob(f"opensignals_{DEVICE_ID}_*.txt")
        if ":Zone.Identifier" not in f.name
    )
    pairs = [(QWERTY_KEYS[i], f) for i, f in enumerate(files) if i < len(QWERTY_KEYS)]
    return pairs


# ── 日本語打鍵頻度 ──────────────────────────────────────────────────────────
# 出典: Wikipedia「文字の出現頻度」ひらがな頻度表 (約2,071万文字の日本語テキスト)
# https://ja.wikipedia.org/wiki/%E6%96%87%E5%AD%97%E3%81%AE%E5%87%BA%E7%8F%BE%E9%A0%BB%E5%BA%A6
HIRAGANA_FREQ = {
    'の': 9.262, 'に': 5.354, 'た': 5.155, 'い': 5.119, 'は': 4.528,
    'を': 4.521, 'と': 4.480, 'る': 4.425, 'が': 4.156, 'し': 4.095,
    'で': 3.693, 'て': 3.661, 'な': 3.477, 'か': 2.594, 'っ': 2.257,
    'れ': 2.177, 'ら': 2.044, 'も': 1.913, 'う': 1.704, 'す': 1.645,
    'り': 1.613, 'こ': 1.508, 'だ': 1.356, 'ま': 1.345, 'さ': 1.250,
    'き': 1.127, 'め': 1.081, 'く': 1.072, 'あ': 0.986, 'け': 0.963,
    'ど': 0.949, 'ん': 0.918, 'え': 0.790, 'よ': 0.745, 'つ': 0.744,
    'や': 0.706, 'そ': 0.635, 'わ': 0.594, 'ち': 0.479, 'み': 0.431,
    'せ': 0.403, 'ろ': 0.355, 'ば': 0.349, 'お': 0.318, 'じ': 0.275,
    'べ': 0.270, 'ず': 0.257, 'げ': 0.237, 'ほ': 0.235, 'へ': 0.227,
    'び': 0.156, 'む': 0.151, 'ご': 0.130, 'ね': 0.113, 'ぶ': 0.112,
    'ぐ': 0.104, 'ぎ': 0.096, 'ひ': 0.092, 'ょ': 0.070, 'づ': 0.063,
    'ぼ': 0.060, 'ざ': 0.058, 'ふ': 0.056, 'ゃ': 0.056, 'ぞ': 0.049,
    'ゆ': 0.041, 'ぜ': 0.033, 'ぬ': 0.025, 'ぱ': 0.021, 'ゅ': 0.013,
    'ぴ': 0.008, 'ぽ': 0.006, 'ぷ': 0.005, 'ぺ': 0.002,
    'ぁ': 0.001, 'ぇ': 0.001, 'ぢ': 0.000, 'ぉ': 0.000, 'ぃ': 0.000,
}

# ローマ字変換テーブル (MS-IME / Google IME デフォルト準拠)
# っ は単独 xtu、小書き仮名は xa/xi/xu/xe/xo/xya/xyo/xyu
ROMAJI_MAP = {
    'あ': 'a',   'い': 'i',   'う': 'u',   'え': 'e',   'お': 'o',
    'か': 'ka',  'き': 'ki',  'く': 'ku',  'け': 'ke',  'こ': 'ko',
    'さ': 'sa',  'し': 'si',  'す': 'su',  'せ': 'se',  'そ': 'so',
    'た': 'ta',  'ち': 'ti',  'つ': 'tu',  'て': 'te',  'と': 'to',
    'な': 'na',  'に': 'ni',  'ぬ': 'nu',  'ね': 'ne',  'の': 'no',
    'は': 'ha',  'ひ': 'hi',  'ふ': 'hu',  'へ': 'he',  'ほ': 'ho',
    'ま': 'ma',  'み': 'mi',  'む': 'mu',  'め': 'me',  'も': 'mo',
    'や': 'ya',                'ゆ': 'yu',               'よ': 'yo',
    'ら': 'ra',  'り': 'ri',  'る': 'ru',  'れ': 're',  'ろ': 'ro',
    'わ': 'wa',  'を': 'wo',  'ん': 'nn',
    'が': 'ga',  'ぎ': 'gi',  'ぐ': 'gu',  'げ': 'ge',  'ご': 'go',
    'ざ': 'za',  'じ': 'zi',  'ず': 'zu',  'ぜ': 'ze',  'ぞ': 'zo',
    'だ': 'da',  'ぢ': 'di',  'づ': 'du',  'で': 'de',  'ど': 'do',
    'ば': 'ba',  'び': 'bi',  'ぶ': 'bu',  'べ': 'be',  'ぼ': 'bo',
    'ぱ': 'pa',  'ぴ': 'pi',  'ぷ': 'pu',  'ぺ': 'pe',  'ぽ': 'po',
    'っ': 'xtu', 'ょ': 'xyo', 'ゃ': 'xya', 'ゅ': 'xyu',
    'ぁ': 'xa',  'ぃ': 'xi',  'ぅ': 'xu',  'ぇ': 'xe',  'ぉ': 'xo',
}


def compute_key_frequency() -> dict[str, float]:
    """ひらがな出現頻度 × ローマ字打鍵数 → アルファベット別打鍵頻度 (%)"""
    key_counts: dict[str, float] = {c: 0.0 for c in 'abcdefghijklmnopqrstuvwxyz'}
    for kana, freq in HIRAGANA_FREQ.items():
        romaji = ROMAJI_MAP.get(kana, '')
        for letter in romaji:
            if letter in key_counts:
                key_counts[letter] += freq
    total = sum(key_counts.values())
    return {k: v / total * 100 for k, v in key_counts.items()}


def plot_emg_vs_frequency(features: dict, key_freq: dict[str, float]):
    """EMG積分値(CH1+CH2) と 打鍵頻度 の比較プロット"""
    keys     = np.array(features["keys"])
    integral = features["integral"].sum(axis=1)  # CH1+CH2
    freq     = np.array([key_freq.get(k, 0.0) for k in keys])

    # 正規化 (0–1)
    integral_norm = integral / integral.max()
    freq_norm     = freq / freq.max()

    # --- 棒グラフ: 頻度順に並べて EMG を重ねる ---
    order = np.argsort(freq)[::-1]
    x = np.arange(len(keys))
    w = 0.4

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # 上段: 打鍵頻度
    ax = axes[0]
    ax.bar(x, freq[order], width=0.8, color='steelblue', alpha=0.8, label='Keystroke freq (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(keys[order])
    ax.set_ylabel('Keystroke frequency (%)')
    ax.set_title('Japanese keystroke frequency (Wikipedia hiragana freq → romaji)')
    ax.legend()

    # 下段: 正規化して重ねて比較
    ax2 = axes[1]
    ax2.bar(x - w/2, freq_norm[order],     width=w, color='steelblue', alpha=0.8, label='Keystroke freq (norm)')
    ax2.bar(x + w/2, integral_norm[order], width=w, color='tomato',    alpha=0.8, label='EMG integral CH1+CH2 (norm)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(keys[order])
    ax2.set_ylabel('Normalized value')
    ax2.set_title('Keystroke frequency vs EMG integral (sorted by keystroke freq)')
    ax2.legend()

    plt.tight_layout()
    out_path = FIGURES_DIR / "emg_vs_frequency.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"保存: {out_path}")
    plt.show()

    # --- 散布図 + 相関 (全体 / w・q除外) ---
    _plot_correlation(keys, freq, integral, key_freq, exclude=None)
    _plot_correlation(keys, freq, integral, key_freq, exclude={'w', 'q'})

    # テキスト表示
    print("\n-- Keystroke frequency vs EMG integral (sorted by freq) --")
    print(f"{'key':>4}  {'freq(%)':>8}  {'EMG integral':>14}  {'freq rank':>10}  {'EMG rank':>9}")
    freq_rank = {k: r+1 for r, k in enumerate(keys[np.argsort(freq)[::-1]])}
    emg_rank  = {k: r+1 for r, k in enumerate(keys[np.argsort(integral)[::-1]])}
    for k in keys[order]:
        print(f"{k:>4}  {key_freq.get(k,0):>8.3f}  {integral[list(keys).index(k)]:>14.1f}"
              f"  {freq_rank[k]:>10}  {emg_rank[k]:>9}")


def _plot_correlation(keys, freq, integral, key_freq, exclude: set | None):
    from scipy.stats import pearsonr, spearmanr

    mask = np.array([k not in (exclude or set()) for k in keys])
    kf = freq[mask]
    ki = integral[mask]
    kl = keys[mask]

    r_p, p_p = pearsonr(kf, ki)
    r_s, p_s = spearmanr(kf, ki)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(kf, ki, color='steelblue', s=60, zorder=3)
    for k, x_, y_ in zip(kl, kf, ki):
        ax.annotate(k, (x_, y_), textcoords='offset points', xytext=(5, 3), fontsize=9)

    # 回帰直線
    if len(kf) > 1:
        coef = np.polyfit(kf, ki, 1)
        xline = np.linspace(kf.min(), kf.max(), 100)
        ax.plot(xline, np.polyval(coef, xline), 'tomato', lw=1.5, label='linear fit')

    title_suffix = " (excl. w, q)" if exclude else " (all keys)"
    ax.set_xlabel('Keystroke frequency (%)')
    ax.set_ylabel('EMG integral CH1+CH2')
    ax.set_title(f'Correlation{title_suffix}\nPearson r={r_p:.3f} (p={p_p:.3f})  Spearman ρ={r_s:.3f} (p={p_s:.3f})')
    ax.legend()
    plt.tight_layout()

    suffix = "_excl_wq" if exclude else "_all"
    out_path = FIGURES_DIR / f"emg_correlation{suffix}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"保存: {out_path}")
    plt.show()


# ── キーボードヒートマップ ───────────────────────────────────────────────────
# 標準QWERTY アルファベット部の座標 (x, row)  ※ 行ごとに0.25uずつ右にずれる
_KB_LAYOUT: dict[str, tuple[float, int]] = {
    'q':(0.00,2),'w':(1.00,2),'e':(2.00,2),'r':(3.00,2),'t':(4.00,2),
    'y':(5.00,2),'u':(6.00,2),'i':(7.00,2),'o':(8.00,2),'p':(9.00,2),
    'a':(0.25,1),'s':(1.25,1),'d':(2.25,1),'f':(3.25,1),'g':(4.25,1),
    'h':(5.25,1),'j':(6.25,1),'k':(7.25,1),'l':(8.25,1),
    'z':(0.75,0),'x':(1.75,0),'c':(2.75,0),'v':(3.75,0),'b':(4.75,0),
    'n':(5.75,0),'m':(6.75,0),
}
# 左手担当キー
_LEFT_KEYS  = set('qwertasdfgzxcvb')
_RIGHT_KEYS = set('yuiophjklnm')


def _draw_keyboard(ax, fig, dmap: dict, title: str, label: str, cmap_name='YlOrRd'):
    """キーボード1枚を描画するヘルパー"""
    import matplotlib.patches as mpatches
    from matplotlib.cm import get_cmap
    from matplotlib.colors import Normalize

    KEY_W = 0.88
    cmap  = get_cmap(cmap_name)
    vals  = np.array([v for v in dmap.values() if v is not None])
    norm  = Normalize(vmin=0, vmax=vals.max())

    for key, (x, row) in _KB_LAYOUT.items():
        val   = dmap.get(key)
        color = cmap(norm(val)) if val is not None else '#d0d0d0'
        txt_c = 'white' if (val is not None and norm(val) > 0.55) else '#333333'

        rect = mpatches.FancyBboxPatch(
            (x + 0.06, row + 0.06), KEY_W, KEY_W,
            boxstyle='round,pad=0.06',
            facecolor=color, edgecolor='white', linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(x + 0.5, row + 0.56, key.upper(),
                ha='center', va='center', fontsize=12, fontweight='bold', color=txt_c)
        if val is not None:
            ax.text(x + 0.5, row + 0.22, f'{val:.1f}' if val < 100 else f'{val/1000:.1f}k',
                    ha='center', va='center', fontsize=6.5, color=txt_c, alpha=0.85)

    ax.axvline(x=4.88, color='#4488cc', lw=1.2, ls='--', alpha=0.5)
    ax.text(4.88, 3.05, 'L / R', ha='center', va='bottom', fontsize=8, color='#4488cc')
    ax.set_xlim(-0.1, 10.3)
    ax.set_ylim(-0.1, 3.3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=12, pad=6)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, orientation='vertical', shrink=0.75, pad=0.01, label=label)


def plot_keyboard_heatmap(features: dict, key_freq: dict | None = None):
    """CH1 / CH2 / 合計 の積分値をキーボード上にヒートマップ表示"""
    keys     = list(features['keys'])
    integral = features['integral']

    data = {
        'CH1  (Left hand)' : dict(zip(keys, integral[:, 0])),
        'CH2  (Right hand)': dict(zip(keys, integral[:, 1])),
        'CH1+CH2  (Total)' : dict(zip(keys, integral.sum(axis=1))),
    }

    fig, axes = plt.subplots(3, 1, figsize=(11, 9))
    for ax, (title, dmap) in zip(axes, data.items()):
        _draw_keyboard(ax, fig, dmap, title, 'EMG integral [a.u.]')

    fig.suptitle('EMG Activity — Keyboard Heatmap', fontsize=13, y=1.01)
    plt.tight_layout()
    out_path = FIGURES_DIR / 'keyboard_heatmap.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"保存: {out_path}")
    plt.show()


def compare_layouts(features: dict, key_freq: dict[str, float]):
    """QWERTY / 大西配列 / 最適配列 の総筋活動コストを比較して棒グラフ表示"""
    from scipy.optimize import linear_sum_assignment

    positions = list(features['keys'])
    emg_by_pos = dict(zip(positions, features['integral'].sum(axis=1)))

    # ── 大西配列レイアウト (QWERTY位置 → 文字、None=非アルファベット) ──────────
    # 出典: o24.works/layout/
    # Row1(Q-P): q l u , . f w r y p
    # Row2(A-L): e i a o - k t n s   ← hは;位置(未計測)
    # Row3(Z-M): z x c v ; g d       ← j,bは,./位置(未計測)
    ONISHI: dict[str, str | None] = {
        'q':'q', 'w':'l', 'e':'u', 'r':None, 't':None,
        'y':'f', 'u':'w', 'i':'r', 'o':'y', 'p':'p',
        'a':'e', 's':'i', 'd':'a', 'f':'o', 'g':None,
        'h':'k', 'j':'t', 'k':'n', 'l':'s',
        'z':'z', 'x':'x', 'c':'c', 'v':'v', 'b':None, 'n':'g',
    }

    # 3配列のコスト計算
    def total_cost(layout: dict[str, str | None]) -> tuple[float, float]:
        """(covered cost, uncovered freq %) を返す"""
        cost = 0.0
        uncovered = 0.0
        for pos in positions:
            letter = layout.get(pos)
            if letter is None:
                continue
            cost += emg_by_pos[pos] * key_freq.get(letter, 0.0)
        # 未計測ポジションの文字の頻度合計（カバーできていない分）
        covered_letters = {v for v in layout.values() if v is not None}
        uncovered = sum(v for k, v in key_freq.items() if k not in covered_letters)
        return cost, uncovered

    qwerty_layout  = {p: p for p in positions}
    opt_positions  = list(features['keys'])
    emg_arr        = features['integral'].sum(axis=1)
    freqs_arr      = np.array([key_freq.get(l, 0.0) for l in opt_positions])
    C              = np.outer(emg_arr, freqs_arr)
    row_ind, col_ind = linear_sum_assignment(C)
    opt_layout     = {opt_positions[r]: opt_positions[c] for r, c in zip(row_ind, col_ind)}

    layouts = {
        'QWERTY'       : qwerty_layout,
        'Onishi\n(o24)': ONISHI,
        'Optimized\n(this study)': opt_layout,
    }

    results = {}
    print("\n── 総筋活動コスト比較 (Σ EMG × 打鍵頻度) ──────────────────────────")
    print(f"{'配列':>12}  {'コスト':>10}  {'カバー率':>10}")
    for name, layout in layouts.items():
        cost, uncov = total_cost(layout)
        coverage    = 100 - uncov
        results[name] = cost
        print(f"{name.replace(chr(10),' '):>12}  {cost:>10,.0f}  {coverage:>9.1f}%")

    # 棒グラフ
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = list(results.keys())
    values = list(results.values())
    colors = ['#4477aa', '#ee7733', '#228833']
    bars   = ax.bar(labels, values, color=colors, alpha=0.85, width=0.5)

    # QWERTYを100%として削減率ラベル
    base = values[0]
    for bar, val, label in zip(bars, values, labels):
        pct = (base - val) / base * 100
        pct_str = f"−{pct:.1f}%" if pct > 0 else ("QWERTY" if pct == 0 else f"+{-pct:.1f}%")
        ax.text(bar.get_x() + bar.get_width()/2, val + base*0.01,
                pct_str, ha='center', va='bottom', fontsize=10, fontweight='bold',
                color=bar.get_facecolor())

    ax.set_ylabel('Total muscle load  Σ EMG × freq  [a.u.]')
    ax.set_title('Keyboard layout comparison\n(muscle load for Japanese romaji input)')
    ax.set_ylim(0, max(values) * 1.18)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=10)

    fig.text(0.5, -0.04,
             'Onishi layout: o24.works/layout/  |  * some positions unmeasured (h, d, j, b, etc.) — slight underestimate for Onishi',
             ha='center', fontsize=7.5, color='gray')

    plt.tight_layout()
    out_path = FIGURES_DIR / 'layout_comparison.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"保存: {out_path}")
    plt.show()


def optimize_layout(features: dict, key_freq: dict[str, float]) -> dict:
    """
    Hungarian algorithm で打鍵頻度×筋活動コストを最小化するキー配置を求める。
    返り値: {ポジション識別キー: 新しく割り当てる文字}
    """
    from scipy.optimize import linear_sum_assignment

    positions = list(features['keys'])           # ポジション順 (= 現在のキーラベル)
    emg       = features['integral'].sum(axis=1) # 各ポジションのEMG積分
    letters   = positions[:]                     # 割り当て候補の文字（同じ25文字）
    freqs     = np.array([key_freq.get(l, 0.0) for l in letters])

    # コスト行列: C[i,j] = emg[ポジションi] × freq[文字j]
    C = np.outer(emg, freqs)

    row_ind, col_ind = linear_sum_assignment(C)

    new_layout   = {positions[r]: letters[c] for r, c in zip(row_ind, col_ind)}
    orig_cost    = float(np.dot(emg, freqs))          # QWERTYのまま同順に割り当てたコスト
    optimal_cost = float(C[row_ind, col_ind].sum())
    reduction    = (orig_cost - optimal_cost) / orig_cost * 100

    print(f"\n-- Layout Optimization --")
    print(f"  Original QWERTY cost : {orig_cost:,.0f}")
    print(f"  Optimized cost       : {optimal_cost:,.0f}")
    print(f"  Reduction            : {reduction:.1f}%")
    print(f"\n  Position → QWERTY / Optimized")
    for pos in positions:
        marker = '' if new_layout[pos] == pos else ' ←'
        print(f"    {pos.upper()} → {new_layout[pos].upper()}{marker}")

    return new_layout, orig_cost, optimal_cost


def plot_optimized_keyboard(features: dict, key_freq: dict[str, float]):
    """QWERTY と最適配列をEMGヒートマップで縦並び比較"""
    import matplotlib.patches as mpatches
    from matplotlib.cm import get_cmap
    from matplotlib.colors import Normalize

    positions  = list(features['keys'])
    emg        = features['integral'].sum(axis=1)
    emg_by_pos = dict(zip(positions, emg))

    new_layout, orig_cost, opt_cost = optimize_layout(features, key_freq)
    reduction = (orig_cost - opt_cost) / orig_cost * 100

    KEY_W = 0.88
    cmap  = get_cmap('YlOrRd')
    norm  = Normalize(vmin=0, vmax=emg.max())

    fig = plt.figure(figsize=(13, 5))
    ax_kb = fig.add_axes([0.02, 0.08, 0.88, 0.82])   # キーボード領域
    ax_cb = fig.add_axes([0.92, 0.15, 0.02, 0.65])   # カラーバー専用領域
    axes = [ax_kb]
    titles  = [f'Optimized  (total muscle load: {opt_cost/1e6:.2f} M  /  -{reduction:.1f}% vs QWERTY)']
    layouts = [new_layout]

    # 全ポジションの打鍵頻度（割り当て後の文字ベース）
    all_freqs = [key_freq.get(new_layout.get(pos, pos), 0.0) for pos in _KB_LAYOUT]
    max_freq_val = max(all_freqs) if max(all_freqs) > 0 else 1.0
    MAX_R = 0.30   # 最大円半径 (key unit)

    for ax, title, layout in zip(axes, titles, layouts):
        for pos, (x, row) in _KB_LAYOUT.items():
            emg_val  = emg_by_pos.get(pos)
            letter   = layout.get(pos, pos)
            freq_val = key_freq.get(letter, 0.0)
            moved    = (letter != pos)

            # --- 背景キー（EMGで色付け）---
            color = cmap(norm(emg_val)) if emg_val is not None else '#d0d0d0'
            txt_c = 'white' if (emg_val is not None and norm(emg_val) > 0.55) else '#333333'
            rect = mpatches.FancyBboxPatch(
                (x + 0.06, row + 0.06), KEY_W, KEY_W,
                boxstyle='round,pad=0.06',
                facecolor=color,
                edgecolor='#cc4422' if moved else 'white',
                linewidth=2.5 if moved else 1.5,
            )
            ax.add_patch(rect)

            # --- 打鍵頻度を表す円（半透明の青い円）---
            r = MAX_R * (freq_val / max_freq_val) ** 0.5   # 面積∝頻度になるよう√スケール
            if r > 0.01:
                circle = plt.Circle(
                    (x + 0.5, row + 0.5), r,
                    color='#2266cc', alpha=0.25, zorder=3,
                )
                ax.add_patch(circle)

            # --- キーラベル ---
            ax.text(x + 0.5, row + 0.52, letter.upper(),
                    ha='center', va='center', fontsize=11, fontweight='bold',
                    color=txt_c, zorder=4)
            ax.text(x + 0.5, row + 0.22, f'({pos.upper()})',
                    ha='center', va='center', fontsize=5.5,
                    color=txt_c, alpha=0.6, zorder=4)

        ax.axvline(x=4.88, color='#4488cc', lw=1.2, ls='--', alpha=0.5)
        ax.text(4.88, 3.05, 'L / R', ha='center', va='bottom', fontsize=8, color='#4488cc')
        ax.set_xlim(-0.3, 10.8)
        ax.set_ylim(-0.9, 3.3)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=11, pad=5)

    # --- 凡例（円サイズ）データ座標で描画 ---
    ax.text(0.0, -0.35, 'Keystroke frequency:', fontsize=7.5,
            color='#444444', va='center')
    for i, (pct, label) in enumerate([(1.0, f'{max_freq_val:.0f}%'), (0.5, f'{max_freq_val*0.25:.1f}%'), (0.15, '~0%')]):
        lx = 2.2 + i * 1.4
        ly = -0.55
        r  = MAX_R * pct ** 0.5
        ax.add_patch(plt.Circle((lx, ly), r, color='#2266cc', alpha=0.25, zorder=3))
        ax.add_patch(plt.Circle((lx, ly), r, color='#2266cc', alpha=0.5,
                                fill=False, linewidth=0.8, zorder=4))
        ax.text(lx, ly - r - 0.09, label, ha='center', va='top', fontsize=7, color='#444444')
    ax.text(10.8, -0.35, '(X) = QWERTY position', fontsize=7,
            color='#888888', va='center', ha='right')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=ax_cb, label='EMG integral [a.u.]')
    fig.text(0.02, 0.005,
             'Source: Wikipedia Japanese hiragana freq → romaji keystroke freq  |  Color = EMG integral (position cost)',
             fontsize=7, color='gray')
    fig.suptitle('Keyboard Layout Optimization  (minimize Σ EMG × keystroke frequency)',
                 fontsize=12, y=0.98)
    out_path = FIGURES_DIR / 'keyboard_optimized.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"保存: {out_path}")
    plt.show()


def plot_keyboard_heatmap_compare(features: dict, key_freq: dict[str, float]):
    """EMG合計 と 打鍵頻度 を横並びで比較"""
    keys     = list(features['keys'])
    integral = features['integral'].sum(axis=1)
    emg_map  = dict(zip(keys, integral))
    freq_map = {k: key_freq.get(k, 0.0) for k in _KB_LAYOUT}

    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    _draw_keyboard(axes[0], fig, emg_map,  'EMG integral  CH1+CH2', 'EMG integral [a.u.]',  cmap_name='YlOrRd')
    _draw_keyboard(axes[1], fig, freq_map, 'Keystroke frequency (Japanese romaji)', 'Frequency [%]', cmap_name='Blues')

    fig.suptitle('EMG Activity vs Keystroke Frequency', fontsize=13)
    plt.tight_layout()
    out_path = FIGURES_DIR / 'keyboard_compare.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"保存: {out_path}")
    plt.show()


# ── 特徴量抽出 ──────────────────────────────────────────────────────────────
def extract_features(results: dict[str, np.ndarray]) -> dict:
    """各キーの積分値・ピーク値を返す (チャンネルごと)"""
    keys = list(results.keys())
    integral = np.array([np.trapz(results[k], axis=0) / FS for k in keys])  # (n_keys, MUS_NUM)
    peak     = np.array([results[k].max(axis=0)          for k in keys])     # (n_keys, MUS_NUM)
    return {"keys": keys, "integral": integral, "peak": peak}


def plot_ranking(features: dict):
    """積分値・ピーク値の降順棒グラフ"""
    keys     = np.array(features["keys"])
    integral = features["integral"]
    peak     = features["peak"]
    integral_sum = integral.sum(axis=1, keepdims=True)   # (n_keys, 1)
    peak_sum     = peak.sum(axis=1, keepdims=True)

    # チャンネル別 + 合計の辞書
    per_ch_metrics = {"Integral [a.u.]": integral, "Peak [a.u.]": peak}
    sum_metrics    = {"Integral CH1+CH2": integral_sum, "Peak CH1+CH2": peak_sum}

    # --- チャンネル別グラフ (2列) ---
    fig1, axes1 = plt.subplots(len(per_ch_metrics), MUS_NUM, figsize=(12, 4 * len(per_ch_metrics)))
    for row, (metric_name, values) in enumerate(per_ch_metrics.items()):
        for ch in range(MUS_NUM):
            ax = axes1[row, ch]
            order = np.argsort(values[:, ch])[::-1]
            ax.bar(range(len(keys)), values[order, ch], color=f"C{ch}", alpha=0.8)
            ax.set_xticks(range(len(keys)))
            ax.set_xticklabels(keys[order])
            ax.set_title(f"{metric_name}  —  CH{ch+1}")
            ax.set_xlabel("key")
    fig1.tight_layout()
    out1 = FIGURES_DIR / "emg_ranking.png"
    fig1.savefig(out1, dpi=120, bbox_inches="tight")
    print(f"保存: {out1}")
    plt.show()

    # --- CH合計グラフ (1列) ---
    fig2, axes2 = plt.subplots(1, len(sum_metrics), figsize=(12, 4))
    for col, (metric_name, values) in enumerate(sum_metrics.items()):
        ax = axes2[col]
        order = np.argsort(values[:, 0])[::-1]
        ax.bar(range(len(keys)), values[order, 0], color="C2", alpha=0.8)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys[order])
        ax.set_title(metric_name)
        ax.set_xlabel("key")
    fig2.tight_layout()
    out2 = FIGURES_DIR / "emg_ranking_sum.png"
    fig2.savefig(out2, dpi=120, bbox_inches="tight")
    print(f"保存: {out2}")
    plt.show()

    # テキストでも表示
    for metric_name, values in per_ch_metrics.items():
        print(f"\n-- {metric_name} --")
        header = f"{'key':>4}" + "".join(f"  CH{ch+1:>10}" for ch in range(MUS_NUM)) + "     Sum"
        print(header)
        total = values.sum(axis=1)
        order = np.argsort(total)[::-1]
        for rank, idx in enumerate(order):
            row_str = f"{keys[idx]:>4}" + "".join(f"  {values[idx, ch]:>10.2f}" for ch in range(MUS_NUM))
            print(f"{rank+1:>2}. {row_str}  {total[idx]:>10.2f}")


# ── メイン ──────────────────────────────────────────────────────────────────
def main():
    pairs = get_key_file_pairs()
    print(f"対象ファイル数: {len(pairs)}  キー: {''.join(k for k, _ in pairs)}")

    # --- 処理 ---
    results: dict[str, np.ndarray] = {}
    for key, filepath in pairs:
        print(f"  {key}  ←  {filepath.name}")
        emg_raw = read_opensignals(filepath)
        results[key] = filter_emg(emg_raw)

    # --- 概要プロット (全キー × 全チャンネル) ---
    n_keys = len(results)
    fig, axes = plt.subplots(
        n_keys, MUS_NUM,
        figsize=(14, n_keys * 1.2),
        sharex=False,
    )
    if n_keys == 1:
        axes = axes[np.newaxis, :]

    ch_labels = [f"CH{ch+1}" for ch in range(MUS_NUM)]
    for row, (key, emg) in enumerate(results.items()):
        t = np.arange(len(emg)) / FS
        for ch in range(MUS_NUM):
            ax = axes[row, ch]
            ax.plot(t, emg[:, ch], lw=0.4, color=f"C{ch}")
            ax.set_ylabel(key, fontsize=8, rotation=0, labelpad=14)
            ax.tick_params(labelsize=6)
            if row == 0:
                ax.set_title(ch_labels[ch])
            if row == n_keys - 1:
                ax.set_xlabel("Time [s]")

    fig.suptitle("Filtered EMG — QWERTY order", y=1.01)
    plt.tight_layout()
    out_path = FIGURES_DIR / "emg_overview.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\n保存: {out_path}")
    plt.show()

    # --- 積分値・ピーク値ランキング ---
    features = extract_features(results)
    plot_ranking(features)

    # --- 日本語打鍵頻度との比較 ---
    key_freq = compute_key_frequency()
    plot_emg_vs_frequency(features, key_freq)

    # --- キーボードヒートマップ ---
    plot_keyboard_heatmap(features)
    plot_keyboard_heatmap_compare(features, key_freq)
    plot_optimized_keyboard(features, key_freq)


if __name__ == "__main__":
    main()
