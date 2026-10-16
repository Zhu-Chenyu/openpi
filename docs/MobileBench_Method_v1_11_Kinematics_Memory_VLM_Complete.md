# MobileBench Method v1.11
## Kinematics Encoder、Fast／Slow Memory 更新与 VLM 接入完整说明

**本版沿用 v1.10 已确认的架构，集中说明三个问题：Kinematics Encoder 编码什么、Memory 怎样更新、Memory 怎样进入 VLM。**

主线保持为：

> 旧 Fast／Slow Memory 和近期实际运动先编码成连续 tokens，与当前观察一起进入 VLM；VLM 输出 \(H_t\)。\(H_t\) 独立预测 Affordance，并通过外部 MLP Writers 更新记忆。随后，\(H_t\)、新 Fast／Slow 和外部 Kinematics 编码统一融合为 \(Z_t\)，供 Upper／Base 双动作头读取。

本文是已讨论方案的完整设计规格，不是原版 π0.5 已经实现的功能说明，也不代表新增模块已获得实验验证。以用户提供的 v1.10 为设计依据；文末单独列出官方代码接口的核对来源。[D1]

---

## 0. 先固定六个定义

| 符号／模块 | 本版含义 |
|---|---|
| \(H_t\) | **VLM 的输出**，是一组已经读取历史的任务特征 tokens，不是输入向量。 |
| \(M_{t-1}^F,M_{t-1}^S\) | 本轮开始前保存的旧记忆，投影后进入 VLM。 |
| \(M_t^F,M_t^S\) | 本轮由外部 Writers 更新后的新记忆，进入 Fusion，并保存给下一轮。 |
| Motion Encoder | 编码近期**实际发生的运动与反馈**，输出少量 VLM 输入 tokens。 |
| Kinematics Encoder | 编码 Workspace、精确状态、Gripper／TCP 和 Base 能力；详细条件留在 VLM 外部。 |
| \(Z_t\) | \(H_t\)、新快慢记忆和 Kinematics 特征**全部融合完成后**的动作条件。 |

监督保持为三类：**Action、Affordance、Slow Text。Fast 没有独立辅助 loss。**本版不恢复 Trace 重建、独立 EEF Goal Decoder、Joint Decoder 或额外可达性分类任务。

“没有独立 Fast loss”不等于冻结 Fast，也不等于禁止其他目标经共享计算路径间接训练它。

---

## 1. 图一：完整 Inference 与输入输出

```text
[Images + instruction/state] --> [Embeddings] -----+
[Old M_F(t-1), M_S(t-1)] -----> [Input proj.] -----+
[Recent actual motion] ------> [Motion MLP] -------+
                                                   |
                                                   v
                                           [Concat tokens]
                                                   |
                                                   v
                                           [VLM Transformer]
                                                   |
                                                   v
                                                 H_t

H_t ---------------------> [MLP_Aff] ---------------------> Affordance

[H_t + old M + state] ---> [MLP Writers + Update] --------> new M_F, M_S
                                                          |
                                                          +--> save for t+1

H_t -----------------------------------------------+
[New M_F(t), M_S(t)] ------------------------------+
[Kin inputs] --> [Kinematics MLPs] --> E_kin ------+
                                                   |
                                                   v
                                                [Fuse]
                                                   |
                                                   v
                                                 Z_t
                                                   |
                                                   v
                                 [Upper Head <-----> Base Head]
                                 [EEF + gripper    body commands]
                                  bidirectional cross-attention
                                                   |
                                                   v
                                [IK / controllers -> execution]
                                                   |
                      new observation + actual feedback -> next step
```

图中重复出现的 H_t 都是同一个张量；Writers 和 Affordance MLP 使用同一次 VLM 前向产生的 \(H_t\)。没有第二个 VLM；更新记忆后，也不在本轮再次把新记忆送入 VLM。

### 1.1 Inference 输入

| 输入 | 内容与来源 |
|---|---|
| 当前视觉 \(I_t\) | 当前可用相机图像，缺失视角用有效性 mask 标记。 |
| 指令 \(\ell\) | 整体任务指令；不附带真实阶段答案。 |
| 当前状态 \(s_t\) | 关节、EEF、Gripper、身体状态等在线可获得信息。 |
| 近期运动 \(x_t^{mot}\) | 自上次策略调用以来的实际身体／末端运动、时间间隔和可用反馈。 |
| 旧记忆 | 对应 episode 内此前计算得到的 Fast／Slow latent。 |
| 本体配置 | 已知机器人的可达位姿库、TCP／夹爪配置、Base 类型和控制限制。 |

### 1.2 Inference 输出

| 输出 | 用途 |
|---|---|
| NAV／Object Affordance 点及有效性 | 独立几何读出；本版不强制将预测点再次作为动作输入。 |
| Upper Action Chunk | EEF＋Gripper 指令，经对应机器人的 IK／执行接口处理。 |
| Base Action Chunk | 约定语义的身体运动指令，经对应移动控制器执行。 |
| 新 \(M_t^F,M_t^S\) | 保存并供下一次观察读取，不是要发给机器人的物理动作。 |

使用一个共享模型控制训练中出现过的机器人。不同机器人的动作语义、单位、坐标和有效维度必须通过数据／执行 adapter 对齐，不要求它们复现同一条物理轨迹。

---

## 2. Kinematics Encoder 是什么？

### 2.1 它是一组身体信息编码器，不是 IK，也不是任务规划器

Kinematics Encoder 回答：

> **当前这台机器人具有什么几何与运动能力，现在处于什么身体状态？**

它把以下四类输入分别经过 MLP，得到动作模型能够读取的 tokens：

$$
E_t^{kin}=[W_t;e_t^q;e^g;e_t^b].
$$

其中 \(W_t\) 可以是一组位姿 tokens，其他项可以各占一个或少量 tokens。这里的方括号表示沿 token 维拼接，不是逐元素求和。

| 分支 | 输入是什么 | 谁提供 | MLP 输出 |
|---|---|---|---|
| Workspace | 可达 TCP 的位置和朝向；可附加相对当前 EEF 的几何关系 | 已知机器人模型离线生成 | \(W_t\)：空间能力 tokens |
| State | 关节位置／速度、当前 EEF 位姿、Gripper 状态、身体状态及有效维度 mask | 在线本体测量和 FK | \(e_t^q\)：当前状态条件 |
| Gripper／TCP | 类型、开口范围、几何尺度、工具中心偏置和接近轴 | 固定机器人配置与标定 | \(e^g\)：末端执行器条件 |
| Base | 身体／底盘类型、允许指令维度、运动限制；可加入已有执行响应 | 机器人配置与在线反馈 | \(e_t^b\)：移动能力条件 |

“Base”是身体移动的统一接口，不限定为轮式底盘。固定结构约束和当下跟踪误差是不同信息；这里提供条件，不宣称一个 MLP 已经学出完整动力学模型。

### 2.2 Workspace 不是当前关节状态直接变出来的一张图

对每个已知机器人 \(r\)，先确定上身可控关节、固定安装关系和实际 TCP。离线采样合法关节配置 \(q_i\)，通过正运动学获得：

$$
\mathcal P_r=\{(q_i,p_i,R_i)\}_{i=1}^{N_W},
\qquad (p_i,R_i)=FK_r(q_i).
$$

采样时使用关节限制与约定的自碰撞检查。底盘位置固定，上身控制的升降、伸缩或躯干关节可包含在 \(q_i\) 中。

在线加载这台机器人的库，而不是让网络只看 \(q_t\) 就猜出整个范围。库中位置／朝向统一表达在身体参考系，并使用当前夹爪的 TCP 定义。

**需要同时保留位置和朝向：某个位置可以达到，不代表在那里任意末端朝向都能实现。**

有限采样没有覆盖某个位姿、或某次 IK 求解失败，不能直接当成不可达证明。这份库也不包含当前场景中的全部碰撞、路径或动态稳定性约束。

### 2.3 每一个可达位姿怎样过 MLP？

最小输入为：

$$
x_i^W=[p_i;\rho(R_i)],
\qquad w_{i,t}=\operatorname{MLP}_W(x_i^W),
$$

其中 \(\rho(R_i)\) 是统一的旋转表示，例如 6D 表示。每个位姿独立经过**同一个共享 MLP**，输出一个 token，不把整张工作空间先压成一个机器人 ID。

沿用此前讨论，允许加入状态相关几何；下面给出这一可选输入的明确形式：

$$
x_{i,t}^W=
[p_i;\rho(R_i);p_i-p_{E,t};\rho(R_{E,t}^{\top}R_i)].
$$

这样，库本身可以不变，但 token 同时表达该位姿相对当前末端的位置和朝向差。采用最小输入还是增加相对几何，需要在配置中写明，不默认为已验证的优选方案。

$$
W_t=\operatorname{Stack}_{i=1}^{N_W}\operatorname{MLP}_W(x_{i,t}^W).
$$

其他三类输入分别编码：

$$
e_t^q=\operatorname{MLP}_q(x_t^q),\quad
e^g=\operatorname{MLP}_g(x^g),\quad
e_t^b=\operatorname{MLP}_b(x_t^b).
$$

所有输出对齐到动作条件宽度 \(d_z\)。不同关节数量通过固定状态布局与 mask 对齐；旋转关节和移动关节分别按物理含义归一化。

**位置尺度需要保留真实身体差异。**若分别把每台机器人的工作空间缩放成同一个单位范围，却不保留尺度，模型可能失去“哪台手臂更长”的信息。归一化统计和坐标约定必须随 checkpoint 保存。

### 2.4 Kinematics 在哪里用？

详细 Workspace、Gripper／TCP 和 Base 编码**不整体进入 VLM**，而在 VLM 外参与最终融合：

$$
\boxed{
Z_t=\operatorname{Fuse}(H_t,M_t^F,M_t^S,E_t^{kin}).
}
$$

不是先形成 \(Z_t\)，再在旁边附加一份身体信息。\(Z_t\) 就是所有条件融合后的最终结果。

两个动作头都读取完整 \(Z_t\)：上身需要知道底盘怎样配合，底盘也需要知道上身能力。Kinematics 编码器主要由动作路径的损失训练，不增加独立机器人分类或可达性问答 loss。

本版没有独立的 EEF Goal Decoder／Goal–Workspace Decoder；以前关于工作空间的输入保留在本编码器中。任务 Affordance 点也不等同于经过本体几何验证的可执行位姿。

---

## 3. Memory 怎样更新？

### 3.1 区分“输入投影”“候选写入”“持久状态”

一次循环存在三种不同的张量：

| 张量 | 是什么 | 是否跨时刻保存 |
|---|---|---|
| 旧 \(M_{t-1}^F,M_{t-1}^S\) | 上一次存下来的 latent 状态 | 是 |
| \(T_{t-1}^F,T_{t-1}^S\) | 旧记忆为进入 VLM 而投影得到的输入 tokens | 不单独保存 |
| 新 \(M_t^F,M_t^S\) | 外部 MLP Writers 更新后的状态 | 是，替换旧状态 |

把旧记忆投影到 VLM 宽度，不等于已经更新了记忆。VLM 内记忆位置产生的隐藏激活，也不自动覆盖外部状态。本版明确由外部 MLP Writers 负责持久写入。

### 3.2 从 \(H_t\) 得到 Writer 输入

保持 \(H_t\) 为一组完整 tokens；只在 Writer 支路做带 mask 的池化与投影：

$$
h_t=P_h\left(\operatorname{MaskedPool}(H_t)\right).
$$

本版的最小实现可以采用 masked mean＋线性投影。它不改变 Action Fusion 仍直接读取完整 \(H_t\) 的约定。

### 3.3 Fast：每次新观察更新

对第 \(i\) 个 Fast slot，广播本轮 \(h_t\)，并与该 slot 的旧状态组合：

$$
\widetilde m_{t,i}^{F}
=\operatorname{MLP}_{F}
\left([h_t;e_t^{F,state};\operatorname{LN}(m_{t-1,i}^{F});s_i^{F} ]\right).
$$

\(s_i^F\) 是可学习槽位标识。\(e_t^{F,state}\) 是此前允许 Fast Writer 读取的本体状态／反馈摘要，可由一个轻量投影获得；它不包含隐藏任务答案。

同一组内各槽位共享 MLP 参数，候选输出宽度为外部记忆宽度 \(d_m\)。实现时可用两层 Linear＋非线性，不需要额外 Transformer 或文字生成器。

更新为：

$$
\boxed{
M_t^F=(1-\alpha_F)M_{t-1}^F+\alpha_F\widetilde M_t^F.
}
$$

\(\alpha_F\) 控制新内容的写入比例。固定混合系数不等于 Writer 固定不训练。

### 3.4 Slow：候选仍来自 \(H_t\)，但间隔写入

$$
\widetilde m_{t,i}^{S}
=\operatorname{MLP}_{S}
\left([h_t;\operatorname{LN}(m_{t-1,i}^{S});s_i^{S}]\right).
$$

每个环境按自身 episode 的观察计数定义写入标记：

$$
\chi_t=1\quad\text{当 }t=0\text{ 或 }t\bmod K=0.
$$

$$
\boxed{
M_t^S=
\begin{cases}
(1-\alpha_S)M_{t-1}^S+\alpha_S\widetilde M_t^S,&\chi_t=1,\\
M_{t-1}^S,&\chi_t=0.
\end{cases}
}
$$

Slow 没写入的时刻仍可被 VLM 和动作头读取；只是它的持久状态不变。无须在这些时刻计算一份不会使用的写入候选。

当前版本不增加基于真实 phase 的写入触发器，也不让 Slow Writer 直接读取本轮新的 Fast。旧 Fast 已经通过 VLM 影响 \(H_t\)，两组仍存在间接交互。

### 3.5 初始化、存储与时间顺序

新 episode 开始时，使用可学习初始槽位或固定初始化加槽位标识。它们是共享初始化参数，不是每条轨迹提供的正确历史。

顺序始终为：

```text
old memory
    -> input projection
    -> VLM with the current observation
    -> H_t
    -> external MLP writers
    -> new memory
    -> current Fusion AND storage for the next observation
```

同一轮动作采样期间，记忆固定。执行完成后，下一次观察才带入新的实际反馈。**“计划抓取”不会因为动作被输出，就立即变成“已确认抓住”。**

两组分别按环境维护，episode 结束时重置；NAV／MANIP 切换不重置。训练和推理使用相同的实际观察间隔。

快慢只是更新尺度和训练约束的区别，不保证固定容量状态可以无限期无损保存信息。Slow 也可能漏掉短暂事件；这一限制需要实验测量，不能靠命名消除。

### 3.6 哪组受到什么监督？

**Fast：没有独立辅助 loss。**不做 Fast 文本解码，也不增加 EEF Trace 重建。它通过动作和其他允许的间接梯度学习保存有用信息。

**Slow：使用阶段短文本监督。**只从 \(M_t^S\) 读出语义特征，与“阶段—对象—持物状态—目的地”短句的编码对齐。沿用前面确定的语义对齐，不默认改成逐字生成。

文本只出现在训练标签侧，不作为 Writer 输入；Slow 文本 loss 主要在写入后计算。运行时可以移除语言监督支路，长期保存的仍是 latent。

---

## 4. Memory 怎样进入 VLM？

### 4.1 接入点：输入 embeddings 之后，第一层多模态 Transformer 之前

不是输入图片编码器，不是变成 prompt 文字，也不是把整个记忆转成 token IDs。

原有视觉／文字先变成向量；旧 Memory 投影成相同宽度的连续 tokens，然后拼接到同一个输入序列。[R1]

$$
X_t^{in}=\operatorname{Concat}_{seq}
[T_t^{obs};T_t^{mot};T_{t-1}^{F};T_{t-1}^{S}].
$$

四部分的含义是：当前图像和指令／状态、近期实际运动、旧 Fast、旧 Slow。

### 4.2 对齐宽度，不改变记忆槽位数量

设外部记忆宽度是 \(d_m\)，实际 VLM 的隐藏宽度是 \(D\)：

$$
M_{t-1}^{F}\in\mathbb R^{B\times N_F\times d_m},\qquad
M_{t-1}^{S}\in\mathbb R^{B\times N_S\times d_m}.
$$

分别做：

$$
T_{t-1}^{F}
=P_F^{in}(\operatorname{LN}(M_{t-1}^{F}))
+e_{type}^{F}+e_{slot}^{F},
$$

$$
T_{t-1}^{S}
=P_S^{in}(\operatorname{LN}(M_{t-1}^{S}))
+e_{type}^{S}+e_{slot}^{S}.
$$

输出形状分别是 \([B,N_F,D]\) 与 \([B,N_S,D]\)。Input projection 可先用 Linear；Type 标识区分快慢组，Slot 标识区分槽位。

例如 \([B,32,256]\rightarrow[B,32,D]\)：只是每个向量变宽，仍然是 32 个 tokens。\(D\) 从所用 checkpoint 读取，不写死为某个模型的固定大小。[R2]

这些输入投影与后面的动作侧投影不共享为同一接口：

- \(P^{in}\)：\(d_m\rightarrow D\)，负责进入 VLM。
- \(P^{act}\)：\(d_m\rightarrow d_z\)，负责进入最终动作条件。

### 4.3 Motion Encoder 为什么也进入？

Motion 描述“刚才实际发生什么”，Kinematics 描述“这个身体允许怎么做”。两者不混用。

本版将新增实际运动整理成紧凑输入：

$$
x_t^{mot}=[\Delta T_t^{base};\Delta T_t^{EEF};g_t;\Delta t;f_t].
$$

这里的各项以固定数值表示展开；\(f_t\) 只包含真实可获得的反馈，并携带缺失性标记。运动变化来自测量／估计，不能直接把下发命令当成实现的位移。

$$
T_t^{mot}
=\operatorname{Reshape}\left(\operatorname{MLP}_{mot}(x_t^{mot})\right)
\in\mathbb R^{B\times N_{mot}\times D}.
$$

\(N_{mot}\) 为少量 tokens。第一版不把整条高频历史轨迹全部输入；只编码上一段新增运动，由记忆递归保存历史。

本体状态继续沿用所选 π0.5 实现的原有输入预处理；例如本次核对的官方 tokenizer 支持把离散状态加入 prompt。动作侧 State MLP 仍保留精确连续状态。[R3]

### 4.4 沿序列维拼接，并同步扩展 mask 和 positions

$$
[B,N_{obs},D]+[B,N_{mot},D]+[B,N_F,D]+[B,N_S,D]
\rightarrow[B,N_{total},D].
$$

这里加号只表示拼接。不是逐元素相加，也不是把最后一维扩成 \(4D\)。

额外维护 \([B,N_{total}]\) 的有效性 mask、位置索引和注意力规则。按实际有效输入处理 padding，池化和读出不计入 padding；训练／推理的顺序与位置处理一致。

### 4.5 Attention 必须让 \(H_t\) 真正读到 Memory

当前条件区采用以下规则：

| Query 读取哪一组 Key／Value | Observation／text | Motion | Old Fast | Old Slow | 本轮带噪动作／未来答案 |
|---|---:|---:|---:|---:|---:|
| Observation／text | 允许 | 允许 | 允许 | 允许 | 禁止 |
| Motion | 允许 | 允许 | 允许 | 允许 | 禁止 |
| Old Fast | 允许 | 允许 | 允许 | 允许 | 禁止 |
| Old Slow | 允许 | 允许 | 允许 | 允许 | 禁止 |

**条件区内部双向读取，不代表跨环境时间读取未来。**进入该区的只有本轮已有观察和历史状态。

如果只把 Memory 追加到末尾，却保留“前面不能看后面”的 causal mask，观察位置的输出可能仍读不到记忆。必须同时修改实际 attention 规则。[R1]

慢组监督文本不放进这个双向条件区；否则真实阶段答案可能泄漏给策略。

### 4.6 VLM 输出后，明确哪部分叫 \(H_t\)

$$
Y_t=F_{VLM}(X_t^{in};\mathcal A_t),
\qquad
H_t=\operatorname{Select}_{obs}(Y_t).
$$

\(\operatorname{Select}_{obs}\) 取原有观察／指令位置的最终 hidden states。它们可以读取前述 Memory，因此 \(H_t\) 已结合历史，不能再称为“纯当前观察特征”。

VLM 内 Memory 位置的输出不作为另一套持久记忆写入规则。仍然按照第 3 节，从 \(H_t\) 经过外部 MLP Writers 更新 \(M_t^F,M_t^S\)。

**投影解决尺寸兼容，训练才使这些连续 tokens 对模型有用；不能声称加进序列后便天然拥有稳定记忆。**

---

## 5. 图二：三个机制的细节框图

以下 A、B、C 都是图一的局部展开，不是三个串联的新系统。图二中的 VLM 与图一是同一个模块。

```text
A. KINEMATICS: physical inputs -> MLPs -> encoded tokens -> final Fusion

Workspace pose samples --> [shared MLP_W per pose] --> W_t -----+
Continuous robot state --> [MLP_q] ------------------> e_q -----+
Gripper / TCP config ----> [MLP_g] ------------------> e_g -----+
Base type / limits ------> [MLP_b] ------------------> e_b -----+
                                                               |
                       H_t + new M_F + new M_S -----------------+
                                                               v
                                                          [Fuse -> Z_t]
                                                               |
                                                        [Upper / Base]

B. MEMORY WRITE: use the H_t produced by this observation

                         [VLM output H_t]
                                  |
                         [Masked pool + P_h]
                                  |
                 +----------------+------------------+
                 v                                   v
    [h_t, state, old Fast slot]          [h_t, old Slow slot]
                 |                                   |
             [MLP_F]                              [MLP_S]
                 |                                   |
        [Fast candidate]                     [Slow candidate]
                 |                                   |
      [Mix with old Fast]                  [Mix only on slow ticks]
      every observation                    otherwise keep old Slow
                 |                                   |
             [new M_F]                            [new M_S]
                 +----------------+------------------+
                                  |
                  current Fusion + store for next observation

C. MEMORY READ: old states become continuous VLM input tokens

Images -------------> [Vision encoder] --------------> image tokens --+
Instruction / state -> [Text embedding] -------------> text tokens ---+
Actual recent motion -> [Motion MLP] ----------------> motion tokens -+
Old M_F ------------> [Norm + Linear_F + type/slot] ---> Fast tokens ---+
Old M_S ------------> [Norm + Linear_S + type/slot] ---> Slow tokens ---+
                                                                     |
                                                      [Concat seq axis]
                                                                     |
                                      [Valid mask + bidirectional prefix mask]
                                                                     |
                                                      [VLM Transformer]
                                                                     |
                                                      [Select obs output]
                                                                     |
                                                                    H_t
```

### 5.1 Fusion 的最小实现

先将各路对齐到 \(d_z\)，然后沿 token 维拼接：

$$
\boxed{
Z_t=\operatorname{Concat}_{seq}
[P_H^{act}H_t;P_F^{act}M_t^F;P_S^{act}M_t^S;W_t;e_t^q;e^g;e_t^b].
}
$$

这个最小实现不额外增加一层复杂 Fusion Transformer。实际的跨条件信息读取在 Action Heads 的 attention 内发生。

两个动作头都将 \(Z_t\) 投影为自己每一层的 K／V，动作 tokens 作为 Query 读取。Upper 和 Base 还保留中间动作特征的双向 cross-attention。

\(Z_t\) 是一组条件 tokens，不是单个强制压缩向量，也不是原版 VLM 的逐层 KV cache。增加该条件接口和双动作专家是本方案的动作侧改造，不能声称只修改输入拼接就已完成全部模型。

### 5.2 独立 Affordance 输出

$$
\hat A_t=\operatorname{MLP}_{Aff}(\operatorname{Read}(H_t)).
$$

Read 沿用既定任务读出或轻量池化，不强制增加 Joint Decoder。MLP 可用 NAV／Object 两个输出槽位，每个输出一个点与有效性。

点的维度、坐标与归一化必须在数据中固定。本文可用当前身体系三维点说明接口，但不把此前尚未固定的 2D／3D 标注选择当成已经完成的实现。

本版 Affordance 是任务相关目标点，不强制等于该机器人已验证可执行的点。详细 Kinematics 没进入该预测路径，不能却要求它在没有相应身体条件时完成身体特定的可行性判断。

---

## 6. 图三：完整 Training 与监督

```text
Forward inputs: observation, actual motion, OLD memories, robot capability
                                 |
                [Embed obs / motion / old memories]
                                 |
                               [VLM]
                                 |
                               [H_t]
                                 |
        +------------------------+-----------------------------+
        |                        |                             |
    [MLP_Aff]          [Fast Writer + update]        [Slow Writer + update]
        |                        |                             |
      A_hat                    new M_F                       new M_S
        |                        |                             |
     [L_aff]                     |             +---------------+
        ^                        |             |               |
        |                        |             |       [Slow semantic readout]
        A*                       |             |               |
                                 |             |         [L_slow-text]
                                 |             |               ^
                  H_t -----------+-------------+               |
                                 |                      [TextEnc(y*)]
                                 v
Robot capability -> [Kin MLPs] -> [Fuse -> Z_t]
                                 |
Noisy action + noise time ------> [Upper <----> Base Heads]
                                 |
                           predicted flows
                                 |
                              [L_act] <--- flow targets from expert a* + noise

Labels: A*, y*, expert a*       --> losses / action-noise construction only
Fast: no independent text / trace / reconstruction loss
Total: L_act_Upper + L_act_Base + lambda_A * L_aff + lambda_S * L_slow-text
```

### 6.1 训练输入、输出与标签分开

| 类别 | 内容 |
|---|---|
| 模型前向条件 | 与 inference 相同的观察、实际运动、机器人条件和模型自己递归产生的旧记忆。 |
| 动作生成训练输入 | 专家动作加噪后的 Upper／Base 动作，以及去噪时间；只进入动作模块。 |
| 前向预测 | Affordance、新快慢记忆、Slow 语义读出、两个动作分支的生成向量场。 |
| 监督答案 | Affordance 点与有效性、阶段短文本、由专家动作和噪声构造的 flow target。 |

训练时没有外部提供的“正确 Fast／Slow latent”。标签也不能作为答案输入 VLM 或 Memory Writer。

### 6.2 总损失

$$
\boxed{
\mathcal L=
\mathcal L_{act}^{Upper}
+\mathcal L_{act}^{Base}
+\lambda_A\mathcal L_{aff}
+\lambda_S\mathcal L_{slow\text{-}text}.
}
$$

| 分支 | 监督与梯度含义 |
|---|---|
| Affordance MLP | 点坐标和有效性监督；使用真实有效标签控制回归 mask。 |
| Slow Readout | 只读新 Slow，与阶段文本语义对齐；主要在写入后计算。 |
| Fast | 没有独立辅助目标，但参与动作、VLM 历史读取和共享训练路径。 |
| Upper／Base | 对应机器人自己的专家动作监督，不要求不同身体使用相同动作标签。 |
| Kinematics／Fusion | 通过动作条件路径训练；共享参数存在时按实际计算图确定其他间接梯度。 |

Affordance 的标签必须是当前或历史能够支持的点；从未见过、没有线索或已经失效的位置不能被当成应精确恢复的记忆。无效点可以用 NULL 语义，不能让占位零坐标进入物理执行。

Slow 监督示例：

> 已确认抓住红杯子；当前是携物导航；目的地为先前指定的放置区域；放置尚未完成。

采用冻结文本编码器提取标签特征，与 Slow 语义读出做对比／语义对齐。语义等价短句不作负例。只约束一个读出，不把所有 Slow tokens 强制等同于一段文本 embedding。

### 6.3 Fast 没有独立监督，为何仍然能被训练？

直接路径是：

$$
\mathcal L_{act}\rightarrow Z_t\rightarrow M_t^F\rightarrow\operatorname{MLP}_F.
$$

跨时刻还有：

$$
M_{t-1}^F\rightarrow VLM\rightarrow H_t
\rightarrow\hat A_t\rightarrow\mathcal L_{aff}.
$$

经 \(H_t\) 和 Slow Writer 的间接语言梯度也可能存在。因此本版只声称**没有 Fast 专属辅助 loss**，不声称它严格只接受动作梯度。

### 6.4 时序训练与实现边界

按 episode 顺序采样，每个 batch slot 独立维护记忆。使用连续窗口内的截断时间反向传播；到窗口边界 detach，但继续传递数值，不逐帧清空。

训练窗口应覆盖有效的历史依赖，并考虑 Slow 写入间隔。窗口以外没有直接的完整信用分配，不能把循环运行当成已经解决长期保留。

冻结 VLM 参数与对整段 forward 使用 no_grad 不同。需要梯度经过 VLM 返回新增 memory 投影和前一时刻 Writer 时，不能切断这条输入梯度。采用原有梯度隔离配方时，应明确给新增路径设定规则。

动作去噪期间记忆只更新一次。新观察下重新计算 VLM 条件；完整双向条件区变化后，不能无条件复用上一观察的全部 KV cache。[R1]

---

## 7. 一次前向的实现顺序与张量接口

### 7.1 张量表

这里使用统一的 \(d_m\) 作为两组外部存储宽度；分开设置宽度也是接口兼容的扩展，不是本版必需。

| 张量 | 形状 | 所在位置 |
|---|---|---|
| \(M_{t-1}^F\) | \([B,N_F,d_m]\) | 输入前的 Fast 状态 |
| \(M_{t-1}^S\) | \([B,N_S,d_m]\) | 输入前的 Slow 状态 |
| \(T_{t-1}^{F/S}\) | \([B,N_{F/S},D]\) | 投影后的 VLM 输入 |
| \(T_t^{mot}\) | \([B,N_{mot},D]\) | 近期实际运动输入 |
| \(H_t\) | \([B,N_{obs},D]\) | VLM 输出的观察／指令位置 |
| \(h_t\) | \([B,d_h]\) | Writer 用的池化投影 |
| \(M_t^{F/S}\) | \([B,N_{F/S},d_m]\) | 新的持久记忆 |
| \(W_t\) | \([B,N_W,d_z]\) | 可达位姿 tokens |
| \(e_t^q,e^g,e_t^b\) | 每类可为 \([B,1,d_z]\) | 其他具身条件 |
| \(Z_t\) | \([B,N_Z,d_z]\) | 全部融合后的动作条件 |

最小拼接下 \(N_Z=N_{obs}+N_F+N_S+N_W+3\)。若某类状态占多个 tokens，按实际数量调整。所有位置都需携带相应 padding／有效性 mask。

\(N_F,N_S,N_W,K,\alpha_F,\alpha_S,d_m,d_z\) 是待记录和验证的配置，不是论文已经给定的最优数字。

### 7.2 算法级伪代码

以下函数表达接口与顺序，不是可直接套到任意 openpi 版本的完整补丁。Writer 的具体 slot 运算按第 3 节实现。

```python
# State belongs to each environment / episode, not to the shared model weights.
old_fast, old_slow = initialize_episode_memories()
step = 0

while episode_is_active():
    obs, actual_motion = read_online_observation_and_feedback()

    # 1. Build clean VLM inputs. Targets never enter this context.
    obs_tokens, obs_valid = embed_images_text_and_state(obs)
    motion_tokens, motion_valid = encode_recent_motion(actual_motion)
    fast_tokens = project_fast_for_vlm(old_fast)
    slow_tokens = project_slow_for_vlm(old_slow)
    tokens, valid, positions, prefix_mask = pack_clean_prefix(
        obs_tokens, obs_valid,
        motion_tokens, motion_valid,
        fast_tokens, slow_tokens,
    )

    # 2. One clean VLM pass. Observation positions can read both memories.
    hidden = vlm_transformer(tokens, valid, positions, prefix_mask)
    H = select_observation_positions(hidden, obs_tokens.shape[1])
    affordance = affordance_mlp(read_features(H, obs_valid))

    # 3. Persistent memory write occurs outside the VLM.
    h = pool_and_project_for_writers(H, obs_valid)
    fast_new = update_fast_with_mlp(h, obs.state, old_fast)
    write_slow = (step == 0) or (step % slow_interval == 0)
    slow_new = (
        update_slow_with_mlp(h, old_slow) if write_slow else old_slow
    )

    # 4. Detailed body capabilities stay outside the VLM.
    kin_tokens, kin_valid = kinematics_encoder(robot_config, obs.state)
    Z, Z_valid = fuse_projected_tokens(
        H, obs_valid, fast_new, slow_new, kin_tokens, kin_valid
    )

    # 5. Sample both action streams with fixed Z and cross-attention.
    upper_chunk, base_chunk = sample_coupled_action_heads(Z, Z_valid)
    execute_action_prefix(upper_chunk, base_chunk, robot_config)

    # These states summarize evidence available before the executed chunk.
    # Its actual result is incorporated on the next observation.
    old_fast, old_slow = fast_new, slow_new
    step += 1
```

训练时复用相同的前向顺序，将最后的采样替换成 flow-matching 训练步骤，并从 `affordance`、`slow_new` 和动作预测计算对应损失。`write_slow` 同时控制阶段文本监督时机。

### 7.3 接入官方代码时需要改哪些接口？

官方 `embed_prefix` 已负责图像和语言 embeddings 的拼接；其 prefix 内部 attention 与动作 suffix 隔离。我们新增的工作是扩展 tokens、mask、positions，取回 \(H_t\)，维护外部状态，并实现既定的 Fusion／双动作接口。[R1]

| 接口位置 | 本方案的修改 |
|---|---|
| 输入 embeddings | 增加 Fast／Slow 输入投影和 Motion MLP。 |
| Mask／positions | 使新增条件与观察双向读取，并屏蔽 padding。 |
| VLM 输出 | 取原有观察／指令位置形成 \(H_t\)。 |
| Policy state | 按环境保存两组外部记忆与 Slow 更新计数。 |
| 动作条件 | 用完整 \(Z_t\) 构造动作侧 K／V，训练新增投影与双头交互。 |
| Training data | 保留时间连续性、动作标签、点标签及慢组文本；不混入在线答案。 |

上述接口位置是核对结果；Fast／Slow Writers、Kinematics Encoder 和新的双头条件结构仍是本项目设计，不是现成库已经提供的函数。

---

## 8. 实现前必须保持一致的边界

**Motion 不等于 Kinematics。**前者的紧凑实际运动 tokens 进入 VLM；后者的详细能力特征留在外部，参加 \(Z_t\) 融合。精确本体状态可在两条路径以不同表示出现，但不能偷偷加入场景真值。

**读旧记忆，写新记忆。**同一次 VLM 输入不包含尚未产生的新记忆；Slow 没写入不影响其本轮被读取。

**独立 Affordance，不恢复旧的多级 Goal 架构。**本版动作直接读取 \(Z_t\)，不要求所有任务信息经过预测坐标。

**Fast 无独立 loss，Slow 有语言监督。**不重新加入 Trace、Fast text、独立 Goal 或几何分类 loss。任何新增目标都应另作实验版本。

**身体能力不是执行保证。**工作空间输入帮助共享模型适配已见身体；执行仍需约定的坐标转换、IK、移动控制和错误反馈。Upper IK 不得暗中覆盖 Base 指令。

**借用接口不等于已经验证效果。**需要检查投影与 Writer 的梯度、记忆消融、不同身体的任务表现，以及策略实际延迟。

---

## 9. 来源与本版设计依据

**[D1] 设计依据。**用户提供的 `MobileBench_Method_v1_10_MemoryTokens_Into_VLM.md`，尤其 §1–3：外部 Kinematics、Memory 输入投影、MLP 更新、统一 \(Z_t\) 融合和三类监督。本版将其中分散内容按三个问题展开，没有恢复先前删除的模块。具体 MLP 展开、张量表与伪代码是该方案的实现说明，不是引用论文的原始架构。

**[R1] 接口核对。**Physical Intelligence 官方 `openpi`，`src/openpi/models/pi0.py`；核对 `embed_prefix`、`make_attn_mask`、`compute_loss`、`sample_actions` 中的条件输入与动作区隔离。地址：`https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/models/pi0.py`。

**[R2] 宽度与连续向量接口。**同仓库 `src/openpi/models/gemma.py`；实际隐藏宽度从配置读取。地址：`https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/models/gemma.py`。

**[R3] 本体状态输入。**同仓库 `src/openpi/models/tokenizer.py`；状态与指令的处理沿用实际 checkpoint 对应预处理。地址：`https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/models/tokenizer.py`。

外部代码接口核对日期：2026-10-15。正式实现应锁定 commit；以上链接指向会变化的主分支，不是本文新增模型有效性的实验依据。

---

## 最终三句话

**Kinematics Encoder：将工作空间、当前状态、夹爪／TCP 和移动能力分别过 MLP，在 VLM 外形成身体条件。**

**Memory 更新：VLM 输出 \(H_t\) 后，外部 MLP 按槽位产生候选；Fast 每轮写，Slow 间隔写，与旧状态混合后保存。**

**进入 VLM：旧 Memory 经 Norm＋Linear 投影成连续 tokens，在第一层多模态 Transformer 前与图像、文字／状态和近期实际运动拼接；得到的 \(H_t\) 因而能读取历史。新快慢记忆和 Kinematics 再与 \(H_t\) 统一融合为 \(Z_t\)，条件化两个动作头。**
