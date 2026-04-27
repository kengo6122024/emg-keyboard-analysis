# EMG キーボード解析

QWERTYキーボードのタイピング時における筋電図（EMG）解析。日本語打鍵頻度との比較、およびキーボード配列の最適化を行います。

## 概要

2チャンネルの表面筋電（CH1: 左手、CH2: 右手）を計測しながら、QWERTYキーボードのアルファベットキーを順番に（q → w → e → … → n）タイピングしました。MATLABで作成した処理スクリプトをPythonに移植し、解析を行っています。

## データ

- **計測機器**: biosignalsplux（OpenSignalsフォーマット）
- **サンプリング周波数**: 1000 Hz
- **チャンネル数**: 2（EMG × 2筋）
- **ファイル**: `alphabet/` 内の25ファイル — タイムスタンプ順にQWERTY順のキーに対応（q, w, e, r, t, y, u, i, o, p, a, s, d, f, g, h, j, k, l, z, x, c, v, b, n）

## 処理フロー

```
生EMG信号
  → バンドパスフィルタ  （バターワース4次、40–400 Hz）
  → 基線合わせ          （平均除去）
  → 整流化              （絶対値）
  → RMSスムージング     （100サンプル窓）
```

## 解析内容

### 1. EMG波形確認（`emg_overview.png`）
全25キー × 2チャンネルのフィルタ後EMG波形。

### 2. 活動量ランキング（`emg_ranking.png`、`emg_ranking_sum.png`）
チャンネルごと、およびCH1+CH2の合計で積分値・ピーク値を降順にランキング。

### 3. 打鍵頻度との比較（`emg_vs_frequency.png`、`emg_correlation_*.png`）
日本語ローマ字入力の打鍵頻度（Wikipedia のひらがな出現頻度データ〜約2,071万文字〜をNihon-shikiローマ字に変換して集計）とEMG積分値を比較。

- ひらがな頻度データの出典: [Wikipedia — 文字の出現頻度](https://ja.wikipedia.org/wiki/%E6%96%87%E5%AD%97%E3%81%AE%E5%87%BA%E7%8F%BE%E9%A0%BB%E5%BA%A6)
- ローマ字変換: Nihon-shiki（MS-IME / Google IME デフォルト準拠）

### 4. キーボードヒートマップ（`keyboard_heatmap.png`、`keyboard_compare.png`）
EMG活動量と打鍵頻度をQWERTYキーボード配置上にヒートマップで可視化。

### 5. 配列最適化（`keyboard_optimized.png`）
ハンガリアン法（線形割り当て問題）で総筋活動コストを最小化：

```
最小化:  Σ  EMG(ポジションi) × 打鍵頻度(文字j)
```

結果: QWERTY比で**総筋活動コスト −44.1%** の配列を導出。

- 背景色 = 各ポジションのEMGコスト（赤いほど筋負担大）
- 円の大きさ = 割り当てた文字の打鍵頻度（大きいほどよく使う）

![最適化キーボード配列](figures/keyboard_optimized.png)

## 使い方

```bash
pip install numpy scipy matplotlib
python process_EMG.py
```

実行すると全図が同ディレクトリに保存されます。

## 必要環境

- Python 3.10以上
- numpy
- scipy
- matplotlib

## ファイル構成

```
EMG_analysis/
├── process_EMG.py              # メイン解析スクリプト（MATLABからPython移植）
├── process_EMG.m               # 元のMATLABスクリプト（.gitignore対象）
├── alphabet/                   # 生データ（OpenSignals .txt、25キー分）（.gitignore対象）
├── emg_overview.png
├── emg_ranking.png
├── emg_ranking_sum.png
├── emg_vs_frequency.png
├── emg_correlation_all.png
├── emg_correlation_excl_wq.png
├── keyboard_heatmap.png
├── keyboard_compare.png
└── keyboard_optimized.png
```
