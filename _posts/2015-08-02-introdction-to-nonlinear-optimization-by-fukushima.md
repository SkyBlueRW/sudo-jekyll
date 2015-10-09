---
layout: post
title:  "《非线性最优化基础》学习笔记"
date:   2015-08-02 14:33:54
categories:
---

《[非线性最优化基础](http://book.douban.com/subject/6510671/)》 作者 [福嶋雅夫](http://www.seto.nanzan-u.ac.jp/~fuku/index.html) (《非线性最优化基础》（豆瓣链接：[http://book.douban.com/subject/6510671/](http://book.douban.com/subject/6510671/)）。福嶋雅夫（Masao Fukushima），教授，日本南山大学理工学院系统与数学科学系，日本京都大学名誉教授，加拿大滑铁卢大学/比利时那慕尔大学/澳大利亚新南威尔士大学客座教授。主页：[http://www.seto.nanzan-u.ac.jp/~fuku/index.html](http://www.seto.nanzan-u.ac.jp/~fuku/index.html)。)

该文为[冯象初教授](http://web.xidian.edu.cn/xcfeng/)(冯象初，教授，西安电子科技大学数学系。主页：[http://web.xidian.edu.cn/xcfeng/](http://web.xidian.edu.cn/xcfeng/))有关非线性最优化的讲座的笔记。

## 主要内容 ##

**理论基础**

1. 凸函数、闭函数
2. 共轭函数
3. 鞍点问题
4. Lagrange 对偶问题
5. Lagrange 对偶性的推广
6. Fenchel 对偶性

**算法**

1. Proximal gradient methods
2. Dual proximal gradient methods
3. Fast proximal gradient methods
4. Fast dual proximal gradient methods

<!--more-->

## 理论基础 ##

### 凸函数、闭函数 ###

给定函数 \\(f : \Re^n \to [-\infty, +\infty] \\)，称 \\(\Re^{n+1}\\) 的子集

\\[ graph \; f = \left\{ ({\bf x}, \beta)^T \in \Re^{n+1} \mid \beta = f({\bf x}) \right\} , \\]

为 \\(f\\) 的**图像**（graph），而称位于 \\(f\\) 的图像上方的点的全体构成的集合

\\[epi \; f =\left\{ ({\bf x}, \beta)^T \in \Re^{n+1} \mid \beta \geqslant f({\bf x}) \right\} \\]

为 \\(f\\) 的**上图**（epigraph）。若上图 \\(epi \; f\\) 为凸集，则称 \\(f\\) 为**凸函数**(convex function)。

**定理 2.27** 设 \\( \mathcal{I} \\) 为任意非空指标集，而 \\(f_i : \Re^n \to [-\infty, +\infty] \; (i \in \mathcal{I})\\) 均为凸函数，则由

\\[ f({\bf x}) = \sup \left\{ f_i({\bf x}) \mid i \in \mathcal{I} \right\} \\]

定义的函数 \\(f : \Re^n \to [-\infty, +\infty] \\) 为凸函数。进一步，若 \\(\mathcal{I}\\) 为有限指标集，每个 \\(f_i\\) 均为正常的凸函数，并且 \\(\cap_{i \in \mathcal{I}} \; dom \; f_i \neq \varnothing \\)，则 \\(f\\) 为正常凸函数。

若对任意收敛于 \\({\bf x}\\) 的点列 \\(\{ {\bf x}^k\} \subseteq \Re^n\\) 均有

\\[ f({\bf x}) \geqslant \limsup_{k \to \infty}f({\bf x}^k) \\]

成立，则称函数 \\(f:\Re^n\to[-\infty,+\infty]\\) 在 \\({\bf x}\\) 处**上半连续**（upper semicontinuous）；反之，当

\\[ f({\bf x}) \leqslant \liminf_{k \to \infty}f({\bf x}^k) \\]

成立时，称 \\(f\\) 在 \\({\bf x}\\) 处**下半连续**（lower semicontinuous）。若 \\(f\\) 在 \\({\bf x}\\) 处既为上半连续又为下半连续，则称 \\(f\\) 在 \\({\bf x}\\) 处**连续**（continuous）。

### 共轭函数 ###

给定正常凸函数 \\(f:\Re^n \to (-\infty,+\infty]\\)，由

\\[f^\ast({\bf\xi}) = \sup \left\{ \lt{}{\bf x},{\bf\xi}\gt{}-f({\bf x}) \mid {\bf x}\in \Re^n \right\} \\]

定义的函数 \\(f^\ast:\Re^n \to [-\infty,+\infty]\\) 称为 \\(f\\) 的**共轭函数**（conjuagate function）。

**定理 2.36** 正常凸函数 \\(f:\Re^n \to (-\infty,+\infty]\\) 的共轭函数 \\(f^\ast\\) 必为闭正常凸函数。

### 鞍点问题 ###

设 \\(Y\\) 与 \\(Z\\) 分别为 \\(\Re^n\\) 与 \\(\Re^m\\) 的非空子集，给定以 \\(Y\times Z\\) 为定义域的函数 \\(K:Y\times Z\to[-\infty,+\infty]\\)，定义两个函数 \\(\eta:Y\to[-\infty,+\infty]\\) 与 \\(\zeta:Z\to[-\infty,+\infty]\\) 如下：

\\[\eta({\bf y})=\sup\left\{ K({\bf y},{\bf z}) \mid {\bf z} \in Z\right\} \\]

\\[\zeta({\bf z})=\inf\left\{ K({\bf y},{\bf z}) \mid {\bf y} \in Y\right\} \\]

\\[\min \; \; \eta({\bf y})\\s.t. \; \; \; {\bf y} \in Y\\]

\\[\max \; \; \zeta({\bf z})\\s.t. \; \; \; {\bf z} \in Z\\]

**引理 4.1** 对任意 \\({\bf y}\in Y\\) 与 \\({\bf z}\in Z\\) 均有 \\(\zeta({\bf z}) \leqslant \eta({\bf y})\\) 成立。进一步，还有
\\[\sup\left\{ \zeta({\bf z})\mid {\bf z}\in Z\right\} \leqslant \inf\left\{ \eta({\bf y})\mid {\bf y} \in Y\right\} \\]

**定理 4.1** 点 \\((\bar{\bf y},\bar{\bf z})\in Y\times Z\\) 为函数 \\(K:Y\times Z\to[-\infty,+\infty]\\) 的鞍点的充要条件是 \\(\bar{\bf y}\in Y\\) 与 \\(\bar{\bf z}\in Z\\) 满足

\\[\eta(\bar{\bf y})=\inf\left\{ \eta({\bf y})\mid {\bf y}\in Y\right\} =\sup\left\{ \zeta({\bf z})\mid {\bf z}\in Z\right\} =\zeta(\bar{\bf z})\\]

### Lagrange 对偶问题 ###

考虑如下非线性规划问题：

\\[ \min \; \; f({\bf x}) \\ s.t. \; \; g_i({\bf x}) \leqslant 0, \; \; i=1, \cdots, m\\]

其中 \\(f: \Re^n \to \Re\\), \\(g_i: \Re^n \to \Re (i=1, \cdots, m)\\)。

\\[ S = \left\{ x \in \Re^n \mid g_i({\bf x}) \leqslant 0 \text{, } \; \; i=1, \cdots, m\right \}\\]

\\[ L_0({\bf x}, {\bf \lambda}) = \begin{cases}
        f({\bf x}) + \sum^m_{i=1}\lambda_ig_i({\bf x})\;, & {\bf \lambda} \geqslant {\bf 0}\\
        -\infty \; , & {\bf \lambda} \ngeqslant {\bf 0}
    \end{cases}
\\]

\\[ \theta({\bf x}) = f({\bf x}) + \delta_S({\bf x})\\]

\\[ \theta({\bf x}) = \sup \left\{ L_0({\bf x}, {\bf \lambda}) \mid {\bf \lambda} \in \Re^m \right\}\\]

\\[ \omega_0({\bf \lambda}) = \inf \left\{ L_0({\bf x}, {\bf \lambda}) \mid {\bf x} \in \Re^n \right\} \\]

Constrains relax

\\[ F_0({\bf x}, {\bf u}) = \begin{cases}
        f({\bf x}),  & {\bf x} \in        S({\bf u}) & \min  & f({\bf x}) & & \\
        +\infty,      & {\bf x} \notin S({\bf u}) & s.t.      & g_i({\bf x}) & \leqslant u_i, & i = 1, \cdots, m 
    \end{cases}
\\]

\\[ S({\bf u}) = \left\{ {\bf x} \in \Re^n \mid g_i({\bf x}) \leqslant u_i, \; i=1, \cdots, m \right\} \\]

**引理 4.5** Lagrange 函数 \\(L_0: \Re^{n+m} \to [-\infty, +\infty) \\) 与函数 \\(F_0: \Re^{n+m} \to (-\infty,+\infty]\\) 之间有如下关系成立：

\\[L_0({\bf x}, {\bf \lambda}) = \inf \left\{ F_0({\bf x}, {\bf u}) + \lt{}{\bf \lambda}, {\bf u}\gt{} \mid {\bf u} \in \Re^m \right\}\\]

\\[F_0({\bf x}, {\bf u}) = \sup \left\{ L_0({\bf x}, {\bf \lambda}) - \lt{}{\bf \lambda}, {\bf u}\gt{} \mid {\bf \lambda} \in \Re^m \right\}\\]

### Lagrange 对偶性的推广 ###

对于原始问题 \\((P)\\)，考虑函数 \\(F: \Re^{n+M} \to (-\infty, +\infty]\\)，使得对任意固定的 \\({\bf x} \in \Re^n\\)，\\(F({\bf x}, \cdot): \Re^M \to (-\infty, +\infty]\\) 均为闭正常凸函数，并且满足

\\[ F({\bf x}, {\bf 0}) = \theta({\bf x}) \text{, } {\bf x} \in \Re^n \\]

**例 4.7** 设 \\(M = m\\)，考虑函数 \\(F_0: \Re^{n+m} \to (-\infty, +\infty]\\)，利用满足 \\(q({\bf 0}) = 0\\) 的闭正常凸函数 \\(q: \Re^m \to (-\infty, +\infty]\\) 定义函数 \\(F: \Re^{n+m} \to (-\infty, +\infty]\\) 如下：

\\[ F({\bf x}, {\bf u}) = F_0({\bf x}, {\bf u}) + q({\bf u}) \\]


\\[ \theta({\bf x}) = f({\bf x}) + \delta_S({\bf x}) \\]
\\[ \implies F({\bf x}, {\bf u}) \mid F({\bf x}, {\bf 0}) = \theta({\bf x}) \\]
\\[ \implies L({\bf x}, {\bf \lambda}) = \inf \left\{ F({\bf x}, {\bf u}) + \lt{}{\bf \lambda}, {\bf u}\gt{} \mid {\bf u} \in \Re^M \right\} \\]
\\[ \implies \omega({\bf \lambda}) = \inf \left\{ L({\bf x}, {\bf \lambda}) \mid {\bf x} \in \Re^n \right\} \\]


### Fenchel 对偶性 ###

\\[ \min_{\bf x} f({\bf x}) + g({\bf Ax}) \\]

\\[ \begin{cases} & F({\bf x}, {\bf 0}) = \theta({\bf x}), & x \in \Re^n \\
& \theta({\bf x}) = f({\bf x}) + g({\bf Ax}) & \end{cases} \\]

\\[ \implies F({\bf x}, {\bf u}) = f({\bf x}) + g({\bf Ax} + {\bf u}) \\]
\\[ \begin{eqnarray*} \implies L({\bf x}, {\bf \lambda}) & = & \inf \left\{ f({\bf x}) + g({\bf Ax} + {\bf u}) + \lt{}{\bf \lambda}, {\bf u}\gt{} \mid {\bf u} \in \Re^m \right\} \\
& = & f({\bf x}) - g^\ast(-{\bf \lambda}) - \lt{}{\bf \lambda}, {\bf Ax}\gt{} \end{eqnarray*}\\]
\\[ \begin{eqnarray*} \implies \omega({\bf \lambda}) & = & \inf \left\{ f({\bf x} - g^\ast(-{\bf \lambda}) - \lt{}{\bf \lambda}, {\bf Ax}\gt{} \mid {\bf x} \in \Re^n \right\} \\
& = & -f^\ast({\bf A}^T{\bf \lambda}) - g^\ast(-{\bf \lambda}) \end{eqnarray*}\\]

\\[ \min_{\bf \lambda} f^\ast\left( {\bf A}^T{\bf \lambda} \right) + g^\ast(-{\bf \lambda})\\]
\\[ \max_{\bf \lambda} -f^\ast\left({\bf A}^T{\bf \lambda} \right) - g^\ast\left(-{\bf\lambda}\right)\\]

## 算法 ##

### 1. Proximal Gradient Method ###

参考 [Algorithms for large-scale convex optimization - DTU 2010](http://www.eecs.berkeley.edu/~elghaoui/Teaching/EE227A/lecture18.pdf)(A Lecture note from "02930 Algorithms for Large-Scale Convex Optimization" taught by Per Christian Hansen (pch@imm.dtu.dk) and Professor Lieven Vandenberghe ([http://www.seas.ucla.edu/~vandenbe/](http://www.seas.ucla.edu/~vandenbe/)) at Danmarks Tekniske Universitet ([http://www.kurser.dtu.dk/2010-2011/02930.aspx?menulanguage=en-GB](http://www.kurser.dtu.dk/2010-2011/02930.aspx?menulanguage=en-GB)). The Download Link is found at the page of "EE227BT: Convex Optimization - Fall 2013" taught by Laurent El Ghaoui at Berkeley ([http://www.eecs.berkeley.edu/~elghaoui/Teaching/EE227A/lecture18.pdf](http://www.eecs.berkeley.edu/~elghaoui/Teaching/EE227A/lecture18.pdf)). And both of the lectures mentioned the book "Convex Optimization" by Stephen Boyd and Lieven Vandenberghe ([http://stanford.edu/~boyd/cvxbook/](http://stanford.edu/~boyd/cvxbook/)) and the software "CVX" - a MATLAB software for desciplined Convex Programming ([http://cvxr.com/cvx/](http://cvxr.com/cvx/)). A similar lecture note on Proximal Gradient Method from "EE236C - Optimization Methods for Large-Scale Systems (Spring 2013-14)" ([http://www.seas.ucla.edu/~vandenbe/ee236c.html](http://www.seas.ucla.edu/~vandenbe/ee236c.html)) at UCLA' can be found at [http://www.seas.ucla.edu/~vandenbe/236C/lectures/proxgrad.pdf](http://www.seas.ucla.edu/~vandenbe/236C/lectures/proxgrad.pdf).)

#### Proximal mapping ####

The **proximal mapping** (or proximal operator) of a convex function \\(h\\) is

\\[ {\bf prox}_h(x) = \mathop{argmin}_u \left( h(u) + \frac{1}{2} \|u - x\|^2_2 \right)\\]

**examples**

**1.** \\(h(x) = 0: {\bf prox}_h(x) = x\\)

**2.** \\(h(x) = I_C(x)\\) (indicator function of \\(C\\)): \\({\bf prox}_h\\) is projection on \\(C\\)

\\[ {\bf prox}_h(x) = P_C(x) = \mathop{argmin}_{u \in C} \|u - x\|^2_2 \\]

**3.** \\(h(x) = t \|x\|_1\\): \\({\bf prox}_h\\) is shinkage (soft threshold) operation

\\[ {\bf prox}_h = \begin{cases}
    x_i - t & x_i   \geqslant t \\
    0       & |x_i| \leqslant t \\
    x_i + t & x_i   \leqslant -t
\end{cases} \\]

#### Proximal gradient method ####

**unconstrained problem** with cost function split in two components

\\[ \mathop{argmin} f(x) = g(x) + h(x) \\]

\\(g\\) convex, differentiable, with **dom** \\(g=\Re^n\\)

\\(h\\) closed, convex, possibly nondifferentiable; \\({\bf prox}_h\\) is inexpensive

**proximal gradient algorithm**

\\[ x^{(k)} = {\bf prox}_{t_kh} \left( x^{(k-1)} - t_k \nabla g \left( x^{(k-1)} \right) \right) \\]

\\[ t_k \gt{} 0 \text{ is the step size,}\\]

constant or determined by line search

#### Interpretation ####

\\[ x^+ = {\bf prox}_{th} \left( x - t\nabla g(x) \right) \\]


from definition of proximal operator:

\\[ \begin{eqnarray*}
x^+ & = &  \mathop{argmin}_u \left( h(u) + \frac{1}{2t} \left\| u - x + t\nabla g(x) \right\|^2_2 \right) \\
    & = & \mathop{argmin}_u \left( h(u) + g(x) + \nabla g(x)^T(u-x) + \frac{1}{2t} \left\| u - x \right\|^2_2 \right)
\end{eqnarray*}\\]

\\(x^+\\) minimizes \\(h(u)\\) plus a simple quadratic local of \\(g(u)\\) around \\(x\\)

#### Examples ####

\\[ minimize \; \; g(x) + h(x) \\]

**gradient method**: \\(h(x) = 0\\), i.e., minimize g(x)

\\[ x^{(k)} = x^{(k-1)} - t_k\nabla g\left( x^{(k-1)} \right)\\]

**gradient projection method**: \\(h(x) = I_C(x)\\), i.e., minimize \\(g(x)\\) over \\(C\\)

\\[ x^{(k)} = P_C \left( x^{(k-1)} - t_k\nabla g \left(x^{(k-1)} \right) \right) \\]

**iterative soft-thresholding**: \\(h(x) = \|x\|_1\\), i.e., \\( minimize \; \; g(x)+ \| x \|_1\\)

\\[ x^{(k)} = {\bf prox}_{t_kh} \left( x^{(k-1)} - t_k\nabla g\left( x^{(k-1)} \right)  \right) \\]

and

\\[ {\bf prox}_{th}(u)_i = 
\begin{cases}
u_i - t & & u_i \geq t \\
0       & & -t \leq u_i \leq t \\
u_i + t & & u_i \geq t
\end{cases}\\]

 ![{{ "/assets/img/fukushima-softthresholding.jpg" | prepend: site.baseurl | prepend: site.url }}]()

### 2. Dual Proximal Gradient Methods ###

参考 L. Vandenberghe EE236C (Spring 2013-14)

#### Composite structure in the Dual ####

\\[ \begin{eqnarray*}
minimize & & f(x)+g(Ax) \\
maximize & & -f^\ast \left( -A^Tz \right) - g^\ast(z)
\end{eqnarray*}\\]

dual has the right structure for the proximal gradient method if

prox-operator of \\(g\\) (or \\(g^\ast\\)) is cheap (closed form or simple algorithm)

\\(f\\) is strongly convex (\\(f(x)-(\frac{\mu}{2})x^T\\) is convex) implies \\(f^\ast\left(-A^Tz\right)\\) has Lipschitz continuous gradient (\\(L=\frac{\|A\|^2_2}{\mu}\\)):

\\[ \left\| A\nabla f^\ast(-A^Tu)-A\nabla f^\ast(-A^Tv) \right\|_2 \leq \frac{\|A\|^2_2}{\mu}\|u-v\|_2 \\]

because \\(\nabla f^2\\) is Lipschitz continuous with constant \\(\frac{1}{\mu}\\)

#### Dual proximal gradient update ####

\\[ z^+ = prox_{tg\ast}\left( z+tA\nabla f^\ast\left( -A^Tz \right) \right) \\]

equivalent expression in term of \\(f\\):

\\[ z^+ = prox_{tg\ast}(z+tA\hat{x}) \text{  where } \hat{x} = \mathop{argmin}_x \left( f(x) + z^TAx \right)\\]

**1.**  if \\(f\\) is separable, calculation of \\(\hat{x}\\) decomposes into independent problems

**2.**  step size \\(t\\) constant or from backtracking line search

#### Alternating minimization interpretation ####

Moreau decomposition gives alternate expression for \\(z\\)-update

\\[ z^+ = z + t(A\hat{x} - \hat{y}) \\]

where

\\[ \begin{eqnarray*}
\hat{x} & = & \mathop{argmin}_x \left( f(x) + z^TAx \right) \\
\hat{y} & = & prox_{t^{-1}g} \left( \frac{z}{t} + A\hat{x} \right)        \\
        & = & \mathop{argmin}_y \left(g(y) + z^T(A\hat{x} - y) + \frac{t}{2} \|A\hat{x} - y\|^2_2  \right)
\end{eqnarray*}\\]

in each iteration, an alternating minimization of:

**1. Lagrangian** \\(f(x) + g(y) + z^T(Ax - y)\\) over \\(x\\) 

**2. augmented Lagrangian** \\(f(x) + g(y) + z^T(Ax - y) + \frac{t}{2} \|Ax - y\|^2_2\\) over \\(y\\)

#### Regularized norm approximation ####

\\[ minimize f(x) + \|Ax - b\| \text{   (with } f \text{ strongly convex)   } \\]

a special case with \\(g(y) = \|y - b\|\\)

\\[
g^\ast = \begin{cases}
b^Tz    & & \|z\|_\ast \leq 1 \\
+\infty & & otherwise 
\end{cases}
\\]

\\[
prox_{tg\ast}(z) = P_C(z - tb)
\\]

C is unit norm ball for dual norm \\(\|\cdot\|_\ast\\)

**dual gradient projection update**

\\[ \begin{eqnarray*}
\hat{x} & = & \mathop{argmin}_x \left( f(x) + z^TAx \right) \\
z^+     & = & P_C(z + t(A\hat{x} - b))
\end{eqnarray*}\\]

#### Example ####

\\[
minimize \; \; f(x) + \sum^p_{i=1}\|B_ix\|_2 \text{   (with } f \text{ strongly convex)   }
\\]

**dual gradient projection update**

\\[ \begin{eqnarray*}
\hat{x} & = & \mathop{argmin}_x \left( f(x) + \left(\sum^p_{i=1}B^T_iz_i\right)^Tx \right) \\
z^+_i   & = & P_{C_i}(z_i + tB_i\hat{x}) \text{, } \; \; i=1, \cdots, p
\end{eqnarray*}\\]

\\(C_i\\) is unit Euclidean norm ball in \\(\Re^{m_i}\\), if \\(B_i \in \Re^{m_i \times n}\\)

#### Minimization over intersection of convex sets ####

\\[ \begin{eqnarray*}
minimize   & & f(x) \\
subject to & & x \in C_i \cap \cdots \cap C_m
\end{eqnarray*}\\]

\\(f\\) strongly convex; e.g., \\(f(x) = \|x - a\|^2_2\\) for projecting \\(a\\) on intersection

sets \\(C_i\\) are closed, convex, and easy to project onto

**dual proximal gradient update**

\\[ \begin{eqnarray*}
\hat{x} & = & \mathop{argmin}_x \left( f(x) + (z_i + \cdots + z_m)^Tx \right) \\
z^+_i   & = & z_i + t\hat{x} - tP_{C_i}\left(\frac{z_i}{t} + \hat{x}\right) \text{, }\; \; i=1, \cdots, m
\end{eqnarray*}\\]

#### Decomposition of separable problems ####

\\[
minimize \; \; \sum^n_{j=1}f_j(x_j) + \sum^m_{i=1}g_i(A_{i1}x_1 + \cdots + A_{in}x_n )
\\]

each \\(f_i\\) is strongly convex; \\(g_i\\) has inexpensive prox-operator

**dual proximal gradient update**

\\[ \begin{eqnarray*}
\hat{x}_j & = & \mathop{argmin}_{x_j} \left( f_j(x_j) + \sum^m_{i=1}z^T_iA_{ij}x_j \right) \text{, } \; \; j=1, \cdots, n \\
z^+_i        & = & prox_{tg^\ast_i}\left(z_i + t\sum^n_{j=1}A_{ij}\hat{x}_j \right) \text{, } \; \; i=1, \cdots, m
\end{eqnarray*}\\]

### 3. Fast proximal gradient methods ###

参考 L. Vandenberghe EE236C (Spring 2013-14)

#### FISTA (basic version) ####

\\[
minimize \; \; f(x) = g(x) + h(x)
\\]

\\(g\\) convex, differentiable with \\(\mathop{dom} g=\Re^n\\)

\\(h\\) closed, convex, with inexpensive \\(prox_{th}\\) operator

**algorithm**: choose any \\(x^{(0)} = x^{(-1)}\\); for \\(k \geq 1\\), repeat the steps

\\[ \begin{eqnarray*}
y             & = & x^{(k-1)} + \frac{k-2}{k+1} \left( x^{(k-1)} - x^{(k-2)} \right) \\
x^{(k)} & = & prox_{t_kh} \left( y - t_k\nabla g(y) \right)
\end{eqnarray*}\\]

step size \\(t_k\\) fixed or determined by line search

acronym stands for 'Fast Iterative Shrinkage-Thresholding Algorithm'

#### Interpretation ####

first iteration (\\(k = 1\\)) is a proximal gradient step at \\(y = x^{(0)}\\)

next iterations are proximal gradient steps at extrapolated points \\(y\\)

![{{ "/assets/img/fukushima-interpretation.png" | prepend: site.baseurl | prepend: site.url }}]()

note: \\(x^{(k)}\\) is feasible (in \\(\mathop{dom} h\\)); \\(y\\) may be outside \\(\mathop{dom} h\\)

#### Reformulation of FISTA ####

define \\(\theta_k = \frac{2}{k+1}\\) and introduce an intermediate variable \\(v^{(k)}\\)

**algorithm**: choose \\(x^{(0)} = v^{(0)}\\); for \\(k \geq 1\\), repeat the steps

\\[ \begin{eqnarray*}
y             & = & (1 - \theta_k)x^{(k-1)} + \theta_kv^{(k-1)} \\
x^{(k)} & = & prox_{t_kh}(y-t_k\nabla g(y))\\
v^{(k)} & = & x^{(k - 1)} + \frac{1}{\theta_k}\left( x^{(k)} - x^{(k-1)} \right)
\end{eqnarray*}\\]

#### Nesterov's second method ####

**algorithm**: choose \\(x^{(0)} = v^{(0)}\\); for \\(k \geq 1\\), repeat the steps

\\[ \begin{eqnarray*}
y             & = & (1 - \theta_k)x^{(k-1)} + \theta_kv^{(k-1)} \\
v^{(k)} & = & prox_{\left(\frac{t_k}{\theta_k}\right)h} \left( v^{(k-1)} - \frac{t_k}{\theta_k}\nabla g(y) \right)\\
x^{(k)} & = & (1 - \theta_k)x^{(k-1)} + \theta_kv^{(k)}
\end{eqnarray*}\\]

User\\(\theta_k = \frac{2}{k+1}\\) and \\(t_k = \frac{1}{L}\\), or one of the line search methods

identical to FISTA if \\(h(x) = 0\\)

unlike in FISTA, \\(y\\) is feasible (in \\(\mathop{dom} h\\)) if we take \\(x^{(0)} \in \mathop{dom} h\\)

### 4. Fast dual proximal gradient methods ###

参考 A Fast Dual Proximal Gradient Algorithm for Convex Minimization and Applications by Amir Beck and Marc Teboulle at October 10, 2013

\\[ \begin{eqnarray*}
(D)   & = & \max_y\left\lbrace q(y) \equiv -f^\ast\left(A^Ty\right)-g^\ast(-y)\right\rbrace,\\
(D') & = & \min F(y) + G(y),\\
(P') & = & \min \left\lbrace f(x) + g(z): Ax - z = 0 \right\rbrace.
\end{eqnarray*}\\]

\\[
F(y) := f^\ast\left( A^Ty \right), \; \; G(y) :=g^\ast(-y)
\\]

Initialization: \\(L \geq \frac{\|A\|^2}{\sigma}\\), \\(w_1 = y_0 \in \mathbb{V}\\), \\(t_1 = 1\\).

General Step \\((k \geq 1)\\):

\\[ \begin{eqnarray*}
y_k           & = & prox_{\frac{1}{L}G}\left( w_k - \frac{1}{L} \nabla F(w_k) \right)\\
t_{k+1}   & = & \frac{1 + \sqrt{1 + 4t^2_k}}{2} \\
w_{k+1} & = & y_k + \left( \frac{t_k - 1}{t_{k+1}} \right) (y_k - y_{k-1}).
\end{eqnarray*}\\]

#### The Fast Dual-Based Proximal Gradient Method (FDPG) ####

Input: \\(L \geq \frac{\|A\|^2}{\sigma} - \text{ an upper bound on the Lipschitz constant of } \nabla F\\)

Step \\(0\\). Take \\(w_1 = y_0 \in \mathbb{V}\\), \\(t_1 = 1\\).

Step \\(k\\). (\\(k \geq 0\\)) Compute

\\[ \begin{eqnarray*}
u_k           & = & \mathop{argmax}_x \left\lbrace \lt{}x, A^Tw_k\gt{} - f(x) \right\rbrace\\
v_k           & = & prox_{Lg}(Au_k - Lw_k)\\
y_k           & = & w_k - \frac{1}{L}(au_k - v_k)\\
t_{k+1}   & = & \frac{1 + \sqrt{1 + 4t^2_k}}{2}\\
w_{k+1} & = & y_k + \left( \frac{t_k - 1}{t_{k+1}} \right) (y_k - y_{k-1}). \tag*{$\blacksquare$}
\end{eqnarray*}\\]
