# 強化學習 (RL) 實作作業：自訂連續動作空間之「智能溫控系統」環境與控制律設計

## 🎯 作業目標與情境描述

在現代綠色建築與工業控制中，如何兼顧**「系統穩定度」**、**「能源效率」**以及**「設備壽命」**是極為關鍵的課題。

本作業要求你**從頭撰寫一個 Python 環境類別 (`ThermostatEnv`)**，模擬一個房間的熱傳導系統。你必須**自主定義** Agent 所觀察到的狀態空間 (State Space) 與獎勵函數 (Reward Function)，並使用支援**連續動作空間 (Continuous Action Space)**的強化學習演算法（如 DDPG, SAC, PPO 等）訓練一個 Agent 來達成多目標系統優化。

---

## 1. 物理模擬系統規則 (System Dynamics)

環境在呼叫 `step(action)` 時，必須嚴格遵守以下物理更新法則：

* **目標設定與初始條件**：
  * 目標維持溫度 (`T_target`)：`25.0°C`
  * 平均環境外溫 (`T_mean`)：`15.0°C`
  * 初始房溫 (`T_init`)：`18.0°C`
  * 模擬總步數 (`N_max_steps`)：`200` 步
* **連續動作空間 (Continuous Action Space)**：
  * `Power` $\in [0.0, 5.0]$ (kW)，Agent 輸出的動作是一個**連續數值**，代表加熱器的即時輸出功率。
* **物理更新公式 (含日夜週期外溫與跳躍擴散隨機熱干擾)**：

  在每個時間步 $t$，房溫更新考慮動態室外溫度 $T_{\mathrm{outside}}(t)$ 與跳躍擴散干擾 $\epsilon_t$：

$$T_{t+1} = T_t - 0.05 \cdot (T_t - T_{\mathrm{outside}}(t)) + 0.8 \cdot P_t + \epsilon_t$$

### 1.1 週期性日夜動態外溫 ($T_{\mathrm{outside}}(t)$)
室外溫度隨時間步 $t$ 呈現正弦週期起伏（模擬日夜溫差）：

$$T_{\mathrm{outside}}(t) = T_{\mathrm{mean}} + A \cdot \sin\left( \frac{2\pi \cdot t}{\mathrm{Period}} \right)$$

* 平均室外溫度 ($T_{\mathrm{mean}}$)：`15.0°C`
* 日夜振幅 ($A$)：`3.0°C`（室外外溫在 `12.0°C` ~ `18.0°C` 之間正弦循環）
* 週期步數 ($\mathrm{Period}$)：`200` 時間步

### 1.2 跳躍擴散噪訊過程 (Jump-Diffusion Process, $\epsilon_t$)
比照強對流氣候或突發開窗衝擊，隨機干擾結合了背景高斯微風與突發事件：

$$\epsilon_t = \phi \cdot \epsilon_{t-1} + \eta_t + J_t \cdot B_t$$

* **背景小風噪訊 ($\eta_t$)**： $\eta_t \sim \mathcal{N}(0, \sigma^2)$，$\sigma = 0.05^\circ\text{C}$
* **突發開關 ($B_t$)**： $B_t \sim \mathrm{Bernoulli}(p_{\mathrm{jump}})$，機率 $p_{\mathrm{jump}} = 0.03$（每步約 3% 機率發生開窗/氣流衝擊）
* **突發溫度衝擊 ($J_t$)**： $J_t \sim \mathcal{N}(\mu_{\mathrm{jump}}, \sigma_{\mathrm{jump}}^2)$，平均值 $\mu_{\mathrm{jump}} = -1.5^\circ\text{C}$，標準差 $\sigma_{\mathrm{jump}} = 0.3^\circ\text{C}$
* **自回歸記憶係數 ($\phi$)**： `0.6`
* **初始干擾 ($\epsilon_0$)**： `0.0`

---

## 2. 控制優化目標與量化指標 (KPIs)

你設計的 RL Agent 在運行 200 個時間步的過程中，必須同時滿足以下三項控制目標，並在成果報告中計算以下**三大關鍵效能指標 (KPIs)**：

| 控制目標 | KPI 名稱 | 量化計算公式 | 工程意義與目標 |
| :--- | :--- | :--- | :--- |
| **1. 精準控溫** | **平均絕對溫差 (`MAE_temp`)** | $\mathrm{MAE} = \frac{1}{N}\sum_{t=1}^{N} \lvert T_t - 25.0 \rvert$ | 使房間溫度盡快到達並穩定維持在 `25.0°C`（目標：愈小愈好， $\mathrm{MAE} < 0.8^\circ\text{C}$）。 |
| **2. 極致省電** | **總能耗 (`Total_Energy`)** | $\mathrm{Total\ Energy} = \sum_{t=1}^{N} P_t \quad (\mathrm{kWh})$ | 在維繫溫度的前提下，盡可能降低總電能消耗。 |
| **3. 延長設備壽命** | **控制抖動度 (`Jerkiness`)** | $\mathrm{Jerkiness} = \mathrm{Var}(\Delta P) = \frac{1}{N-1}\sum_{t=1}^{N-1} (\Delta P_t - \bar{\Delta P})^2$ | 避免功率劇烈跳動（其中 $\Delta P_t = P_t - P_{t-1}$）。（目標：越接近 `0` 代表控制越平滑）。 |

> 💡 **提示**：當房溫達到 `25.0°C` 時，由於外溫正弦變化，所需維穩功率也會隨時間微幅動態調整。好的人工智慧控制器應該能自我調適並平滑抗禦開窗衝擊。

---

## 3. 核心實作任務 (Student Tasks)

### 任務 A：定義 State Space (`_get_state`)
* **思考點**：Agent 必須觀察到哪些物理量或歷史資訊（例如與目標溫差 $T_t - 25.0$ 等...），才能在動態外溫與開窗衝擊下做出精準且平順的功率輸出？
* **實作要求**：填寫 `ThermostatEnv._get_state()` 並回傳一個一維 NumPy Array。強烈建議對特徵進行適度歸一化 (Normalization)。

### 任務 B：設計 Reward Function (`_compute_reward`)
* **思考點**：如何將「控溫精準度」、「總能耗」與「設備抖動度」這三個相互衝突的目標組合為純量 Reward？
* **避坑指南**：適當調配權重 (Weights)，避免 Agent 出現「為了省電與避免抖動而直接關機擺爛」或「暴耗電狂震盪只求控溫」的鑽漏洞行為 (Reward Hacking)。
* **實作要求**：填寫 `ThermostatEnv._compute_reward(action_power)`。

### 任務 C：指定演算法實作 (Specified Algorithms)
本作業要求實作並對比以下**三種控制方法 / RL 演算法**：

1. **Bang-Bang 控制器 (規則基準)**：傳統雙位溫控器邏輯。當 $T_t \le 25.0^\circ\text{C}$ 時全功率輸出 (`5.0 kW`)； $T_t > 25.0^\circ\text{C}$ 時關閉 (`0.0 kW`)。
2. **DQN Agent (離散動作空間 RL 基準)**：採用 Deep Q-Network，將連續加熱功率 $[0.0, 5.0]$ kW 離散化為離散檔位（例如 6 個動作檔位 $\{0.0, 1.0, 2.0, 3.0, 4.0, 5.0\}$ kW）。
3. **PPO Agent (連續動作空間 RL 主力)**：採用 Proximal Policy Optimization，直接在連續動作空間 $A \in [0.0, 5.0]\text{ kW}$ 輸出精準功率。

### 任務 D：離散 vs. 連續控制動態分析
* **訓練與測試**：訓練 DQN 與 PPO 至收斂後，在相同的 200 步測試環境中進行評估。
* **對比分析點**：觀察離散動作 (DQN) 與連續動作 (PPO) 在連續熱物理系統中的差異。特別比較檔位頻繁切換對控制抖動度 ($\mathrm{Jerkiness}$) 與設備壽命的衝擊。

---

## 4. 專案檔案結構範例

建議將你的作業儲存為以下檔案架構：

```text
RL_test/
├── README.md               # 本作業說明文件
├── thermostat_env.py       # 環境類別檔 (含物理模擬與 Task A, Task B 實作)
├── ppo_agent.py            # 連續動作空間 PPO Agent 類別
├── dqn_agent.py            # 離散動作空間 DQN Agent 類別
├── baselines.py            # Bang-Bang 基準控制器
├── train.py                # 訓練腳本 (訓練 PPO 與 DQN Agent)
└── evaluate.py             # 評估與繪圖腳本 (計算 3 大 KPI 並繪製對比圖)
```

---

## 5. 成果報告與交付物 (Deliverables & Submission Checklist)

請於作業繳交時提供包含以下內容的**成果報告 (Report)**：

### 1. 設計原理說明
- [ ] **State Space 設計邏輯**：說明你選擇了哪些觀察特徵？為什麼？
- [ ] **Reward Function 設計邏輯**：列出你的 Reward 公式，並說明各權重項的選取考量與如何防範 Reward Hacking。

### 2. 效能指標 (KPI) 比較表
請繪製或列出你的 PPO Agent、DQN Agent 與 Bang-Bang 控制器的 KPI 對比表格：

| 控制器 / 演算法方法 | 平均絕對溫差 `MAE_temp` (°C) | 總消耗電能 `Total_Energy` (kWh) | 控制抖動度 `Jerkiness` ($\mathrm{kW}^2$) |
| :--- | :---: | :---: | :---: |
| **Bang-Bang 控制器 (規則基準)** | *(學生計算)* | *(學生計算)* | *(學生計算)* |
| **DQN Agent (離散動作 RL 基準)** | *(學生計算)* | *(學生計算)* | *(學生計算)* |
| **PPO Agent (連續動作 RL 主力)** | **< 0.8** *(目標)* | *(最佳化數值)* | **接近 0** *(目標)* |

### 3. 動態控制曲線圖 (200 Steps Plot)
請畫出包含以下兩張子圖的對比圖表（例如 `control_results.png`）：
1. **房間溫度變化曲線 $T_t$** ：需包含 `25.0°C` 目標虛線與室外動態溫度 $T_{\mathrm{outside}}(t)$ 。
2. **加熱功率輸出曲線 $P_t$** ：比較 Bang-Bang、DQN 與 PPO 的功率輸出平滑度。

---

## 6. 基礎 Starter Code (`thermostat_env.py`)

請直接參考並完成 `thermostat_env.py` 中的 `TODO` 區塊。

祝學習愉快！如有任何問題，請洽我大哥 Alston。
