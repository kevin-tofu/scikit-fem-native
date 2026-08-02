ここ、実は**問題によります**。

そして、このプロジェクトをやる価値があるかどうかも、ここで決まります。

## ケース1：線形静解析（小変形）

例えば

```text
assemble: 20%
solve:    80%
```

ということは普通にあります。

この場合、

**Assemblyを5倍速くしても**

```text
20 → 4

合計

100 → 84
```

なので、1.2倍しか速くならない。

これはインパクトが小さい。

---

## ケース2：材料非線形

ここから変わってきます。

例えば

```text
Material update 35%
Assembly        25%
Solve           40%
```

ここでAssembly側（材料更新込み）を5倍速くすると、

```text
60 → 12
```

になり、

```text
100 → 52
```

約2倍速くなる。

かなり効きます。

---

## ケース3：良い前処理がある場合

例えばPETSc+AMGで

```text
Solve 20%

Assembly 80%
```

ということもあります。

特に

* 線形弾性
* 良質なAMG
* そこまで巨大じゃない自由度

では普通にあり得ます。

---

# あなたの用途を見ると

ここが重要なんだけど、

あなたは

* 接触
* プリロード
* 幾何非線形
* CB-ROM
* 治具配置
* 何度も解析

ですよね。

ここでは

**Solver時間だけが支配的ではない。**

例えばROMを作るなら

```text
1000 snapshots
```

作る。

すると

```text
assemble

↓

solve
```

を1000回やる。

もし

```text
assemble 40%

solve 60%
```

なら、

Assembly高速化は十分効く。

---

## さらにROMでは

実は

```text
K

R
```

だけ欲しいケースがある。

例えば

```text
ΦᵀKΦ
```

を作る。

この場合

**solveしない。**

だから

```text
Assembly 100%
```

になる。

これはかなり大きい。

---

# 私が一番期待しているのはここ

あなたは以前、

> **CB-ROMで疎行列のまま扱えないか**

という話をしていました。

その場合、

```text
Native Assembly

↓

K

↓

Projection

↓

Reduced matrix
```

になります。

つまり

```text
Solve
```

がそもそも最後だけ。

だからAssemblyの価値が非常に大きい。

---

# ただ、一つ気になっている

実はこのOSS、

**Assemblyだけでは少し弱い**かもしれません。

というのも、

もし

```text
assemble()

↓

K
```

だけなら、

「PETScでいいじゃん」

となる可能性があります。

---

## だから私はもう一段広げたい

例えば

```text
Native Kernel
```

を売りにする。

つまり

```text
NeoHookean

J2

Damage

Contact
```

を全部

```cpp
evaluate(...)
```

として提供する。

すると

```text
Assembly Engine
```

ではなく

```text
Constitutive Engine
```

にもなる。

これは結構唯一性がある。

---

# 私なら一度ベンチマークする

実は仕様書を書くより、

**まず1週間で測ります。**

例えばscikit-femで

```text
Tet4

50万DOF

NeoHookean
```

くらいで

```text
assemble

solve
```

を分けて測る。

もし

```text
assemble 65%

solve 35%
```

なら、

GOです。

逆に

```text
assemble 10%

solve 90%
```

なら、

方向性を変えるべき。

---

## 私が一番知りたい数字

実は私はこの数字がすごく知りたいです。

**あなたの実際の研究対象で、**

```text
Total Time

=

Geometry

+

Material

+

Assembly

+

Solve
```

を計測したことはありますか？

もしまだなら、このOSSを作る前に**最優先でプロファイル**した方がいいです。

なぜなら、この数字が

* プロジェクトの価値
* どこまでC++化するか
* PETSc連携の優先度

を全部決めるからです。

---

# Form APIとfused nonlinear kernelは両立させる

Neo-Hookeanの比較では、native側で

```text
geometry
→ deformation gradient
→ constitutive update
→ residual + consistent tangent
→ CSR scatter
```

を一つのkernelに融合すると、scikit-femの汎用フォーム評価より非常に
大きな高速化が得られた。ただし、この結果からユーザーの自由なForm APIを
廃止して専用kernelだけに寄せてはいけない。

skfemntvは次の二層構成にする。

## 1. scikit-fem互換Form API

```python
@skfemntv.BilinearForm
def tangent(u, v, w):
    ...
```

任意の弱形式、表面積分、interface coupling、独自係数を記述するための
柔軟な経路である。`import skfemntv as skfem`で書き換えやすいことを重視する。

## 2. fused native kernel API

```python
result = assembler.assemble(u, state, num_threads=4)
```

材料更新、残差、接線、global scatterを融合する高性能経路である。
Neo-Hookean、J2塑性、damage、大量snapshot生成、ROMなどを対象にする。

組み込み材料だけに限定せず、将来は次のようなC++ kernel contractを公開する。

```text
evaluate_qp(deformation, previous_state, parameters)
    -> stress, consistent_tangent, trial_state
```

ユーザーは組み込みkernel、ユーザー作成C++ kernel、外部材料ライブラリの
adapter、または自由なForm APIを選択できる。積分点ごとのPython callbackは
性能を失うため、高速な独自kernelはC++として登録する。

# residual-only評価が必要な理由

非線形解析でも毎回接線行列を更新するとは限らない。

* line search中の残差評価
* Newton収束判定
* modified Newton法
* 接線を数反復ごとに更新する方法
* explicit法
* matrix-free法
* ROMで `Phi.T @ R` だけ必要な場合

したがって`R only`経路には実用上の意味がある。ただし性能グラフでは、
scikit-femの`R + K`とnativeの`R only`を同じ処理として比較してはならない。
主比較は同一処理の`R + K`同士にし、`R only`は補助表または別グラフにする。

# 性能比較で明示すること

native専用kernelのベンチマークでは次を必ず記録する。

* mesh topologyとelement order
* quadrature ruleとintegration order
* residual-onlyかresidual+tangentか
* thread数と実効thread数
* setupを含むか、warm-cache assemblyだけか
* 比較前の残差・接線の数値一致
* affine要素のconstant-gradientなど、利用した専用化

これにより「C++だから速い」という曖昧な主張ではなく、どの専用化と融合が
価値を生んだかを説明できる。
