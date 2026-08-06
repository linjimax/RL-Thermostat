import numpy as np

class ThermostatEnv:
    """
    連續動作空間之智能溫控環境 (Thermostat Environment)
    
    物理熱傳導動態公式 (含週期性日夜外溫與跳躍擴散隨機干擾):
    T_{t+1} = T_t - 0.05 * (T_t - T_outside(t)) + 0.8 * Power_t + epsilon_t
    
    外溫正弦週期變化:
    T_outside(t) = T_mean + A * sin(2 * pi * t / Period)
    
    跳躍擴散干擾過程 (Jump-Diffusion Process):
    epsilon_t = phi * epsilon_{t-1} + eta_t + J_t * B_t
    eta_t ~ N(0, sigma^2)          # 背景微風/小噪音
    B_t ~ Bernoulli(p_jump)        # 突發開窗/衝擊開關
    J_t ~ N(mu_jump, sigma_jump^2) # 突發大幅溫度衝擊
    """
    def __init__(self, target_temp=25.0, init_temp=18.0, seed=None):
        self.target_temp = target_temp
        self.init_temp = init_temp
        self.k = 0.05       # 散熱係數
        self.alpha = 0.8    # 加熱效率
        self.max_steps = 200
        
        # 1. 週期性日夜外溫參數 (Sine Curve)
        self.outside_mean = 15.0        # 平均外溫 (°C)
        self.outside_amplitude = 3.0   # 日夜溫差振幅 (°C)
        self.period = 200               # 週期步數
        
        # 2. 跳躍擴散隨機干擾參數 (Jump-Diffusion Noise)
        self.phi = 0.6                  # 自回歸記憶係數
        self.sigma = 0.05               # 背景高斯雜訊標準差 (°C)
        self.p_jump = 0.03              # 突發開窗/強氣流發生機率 (Bernoulli p)
        self.mu_jump = -1.5             # 突發衝擊平均降溫 (°C)
        self.sigma_jump = 0.3           # 突發衝擊標準差 (°C)
        
        # 定義連續動作空間邊界 [0.0, 5.0] kW
        self.action_min = 0.0
        self.action_max = 5.0
        
        if seed is not None:
            np.random.seed(seed)
        
        self.reset()

    def get_outside_temp(self, step=None):
        """
        計算當前時間步的動態室外溫度 T_outside(t)
        """
        t = self.current_step if step is None else step
        return self.outside_mean + self.outside_amplitude * np.sin(2.0 * np.pi * t / self.period)

    def reset(self):
        """
        重置環境至初始狀態
        """
        self.current_temp = self.init_temp
        self.current_power = 0.0
        self.prev_power = 0.0
        self.prev_temp = self.init_temp
        self.noise = 0.0        # 初始干擾
        self.current_step = 0
        self.outside_temp = self.get_outside_temp(0)
        return self._get_state()

    def _get_state(self):
        """
        ====================================================================
        TODO [任務 A]: 請在此自主設計並回傳你的 State (狀態空間)
        ====================================================================
        思考提示：
        1. Agent 應該觀察哪些特徵？(例如：與目標溫差 T_t - T_target等...)
        2. 強烈建議對狀態數據進行適當數值歸一化 (Normalization)，如縮放至 [-1.0, 1.0] 或 [0.0, 1.0]。
        3. 回傳格式必須為 NumPy Array (dtype=np.float32)。
        """
        raise NotImplementedError("請學生在此填寫 _get_state 實作內容")

    def step(self, action):
        """
        執行單一時間步物理更新
        action: float 或 np.ndarray, 範圍應在 [0.0, 5.0] 之間的連續功率值 (kW)
        """
        self.current_step += 1
        
        # 限制 action 在物理極限 [0.0, 5.0] kW 內
        if isinstance(action, (list, np.ndarray)):
            action = float(action[0])
        clipped_power = float(np.clip(action, self.action_min, self.action_max))
        
        # 保存上一步歷史資料
        self.prev_power = self.current_power
        self.prev_temp = self.current_temp
        self.current_power = clipped_power
        
        # 1. 計算當前時間步的日夜動態外溫 T_outside(t)
        self.outside_temp = self.get_outside_temp(self.current_step)
        
        # 2. 物理熱傳導計算
        heat_loss = self.k * (self.current_temp - self.outside_temp)
        heat_gain = self.alpha * self.current_power
        
        # 3. 跳躍擴散噪訊過程 (Jump-Diffusion Process)
        # epsilon_t = phi * epsilon_{t-1} + eta_t + J_t * B_t
        white_noise = np.random.normal(0, self.sigma)
        is_jump = np.random.binomial(1, self.p_jump)
        jump_impact = np.random.normal(self.mu_jump, self.sigma_jump) if is_jump else 0.0
        
        self.noise = self.phi * self.noise + white_noise + jump_impact
        
        # 4. 更新房溫
        self.current_temp = self.current_temp - heat_loss + heat_gain + self.noise
        
        # 5. 結束條件 (達到最大時間步長 200 步)
        done = self.current_step >= self.max_steps
        
        # 6. 計算獎勵純量
        reward = self._compute_reward(clipped_power)
        
        info = {
            "temp": self.current_temp,
            "power": self.current_power,
            "outside_temp": self.outside_temp,
            "is_jump": bool(is_jump),
            "step": self.current_step
        }
        
        return self._get_state(), reward, done, info

    def _compute_reward(self, action_power):
        """
        ====================================================================
        TODO [任務 B]: 請在此自主設計你的 Reward Function (獎勵函數)
        ====================================================================
        必須同時考量以下三大控制目標：
        1. 控溫精準度：懲罰房溫與目標溫度 25.0°C 的偏差 |T_t - 25.0|
        2. 極致省電：懲罰輸出功率 action_power (kW)
        3. 設備壽命 (防抖動)：懲罰功率變動量 (action_power - prev_power)^2
        
        思考提示：
        - 權重 (Weights) 該如何搭配，才能預防 Reward Hacking？
        - 回傳必須是一個 float 數值 (單一純量 Reward)。
        """
        raise NotImplementedError("請學生在此填寫 _compute_reward 實作內容")


def calculate_kpis(temp_history, power_history, target_temp=25.0):
    """
    計算三大效能指標 (KPIs) 之輔助函數
    
    :param temp_history: list 或 np.ndarray, 200 個時間步的房溫紀錄 (T_1 ~ T_N)
    :param power_history: list 或 np.ndarray, 200 個時間步的功率紀錄 (P_1 ~ P_N)
    :param target_temp: float, 目標設定溫度 (預設 25.0°C)
    :return: dict, 包含 MAE_temp, Total_Energy, Jerkiness 三大指標
    """
    temps = np.array(temp_history)
    powers = np.array(power_history)
    
    # 1. 精準控溫 MAE
    mae_temp = float(np.mean(np.abs(temps - target_temp)))
    
    # 2. 總消耗電能 (kWh)
    total_energy = float(np.sum(powers))
    
    # 3. 控制抖動度 (Var of Delta P)
    delta_p = np.diff(powers)
    jerkiness = float(np.var(delta_p)) if len(delta_p) > 0 else 0.0
    
    return {
        "MAE_temp": mae_temp,
        "Total_Energy": total_energy,
        "Jerkiness": jerkiness
    }
