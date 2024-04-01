# IDT-LFS-initial 使用最大温升率判断点火,验证了bertolino机理（成功）
In this repository, I' ll provide the code how to calculate IDT and LFS in March, 2024.

import numpy as np
import matplotlib.pyplot as plt
import time
import cantera as ct
import pandas as pd

def detect_ignition_by_max_derivative(times, temperatures):
    """
    通过检测温度变化率的最大值来判断是否发生点火。

    参数:
    - times: 时间点列表。
    - temperatures: 对应时间点的温度列表。

    返回:
    - ignition_detected: 布尔值，表示是否检测到点火。
    - max_rate_of_change: 最大温度变化率。
    - ignition_time: 点火时间，如果未检测到点火，则返回None。
    """
    max_rate_of_change = 0
    ignition_time = None

    # 计算温度变化率并找到最大值
    for i in range(1, len(times)):
        delta_T = temperatures[i] - temperatures[i - 1]
        delta_time = (times[i] - times[i - 1]) if times[i - 1] != 0 else times[i]
        rate_of_change = delta_T / delta_time

        if rate_of_change > max_rate_of_change:
            max_rate_of_change = rate_of_change
            ignition_time = times[i]

    ignition_detected = max_rate_of_change > 0  # 如果最大温度变化率大于0，则认为发生了点火
    return ignition_detected, max_rate_of_change, ignition_time * 1000  # 转换为毫秒

# 设置反应器初始条件
reactorTemperature = 1000  # 初始温度（K）没有用
reactorPressure = 20.0*101325.0  # 初始压力（Pa）
# 创建气体模型并设置初始状态
gas = ct.Solution('E:/Desktop/Postgraduate/mech/simplify/Bertolino/Bertolino_mech.yaml')
gas.TP = reactorTemperature, reactorPressure
gas.set_equivalence_ratio(phi=0.5, fuel='NH3', oxidizer={'o2': 1.0, 'n2': 3.76})
# 初始化反应器和反应网
r = ct.Reactor(contents=gas)
reactorNetwork = ct.ReactorNet([r])

# 批量计算不同温度下的点火延迟时间
T = np.array([1525, 1400, 1375, 1350, 1337.5, 1325, 1312.5,
              1300, 1287.5, 1275, 1262.5, 1250, 1237.5, 1225])# 温度数组

estimatedIgnitionDelayTimes = np.ones(len(T))  # 预估点火延迟时间数组
estimatedIgnitionDelayTimes[:] = 0.005
ignition_delays_ms = np.zeros(len(T))  # 理想气体点火延迟时间数组
# 对每个温度进行模拟，计算理想气体的点火延迟时间
for i, temperature in enumerate(T):
    # Setup the gas and reactor
    reactorTemperature = temperature
    gas.TP = reactorTemperature, reactorPressure
    gas.set_equivalence_ratio(phi=0.5, fuel='NH3',
                              oxidizer={'o2': 1.0, 'n2': 3.76})
    r = ct.Reactor(contents=gas)
    reactorNetwork = ct.ReactorNet([r])

    timeHistory = ct.SolutionArray(gas, extra=['t'])

    t0 = time.time()

    t = 0
    counter = 1
    while t < estimatedIgnitionDelayTimes[i]:
        t = reactorNetwork.step()
        if counter % 20 == 0:
            timeHistory.append(r.thermo.state, t=t)
        counter += 1

    # 提取时间和温度数据
    times = timeHistory.t
    temperatures = timeHistory.T
    #  检测点火时间
    ignition_detected, max_derivative, ignition_time_ms = detect_ignition_by_max_derivative(times, temperatures)
    if ignition_detected:
        ignition_delays_ms[i] = ignition_time_ms  # 存储点火延迟时间
        print(f"Ignition detected at {temperature} K:"
              f" {ignition_time_ms:.2f} ms with max derivative {max_derivative:.2f} K/s")
    else:
        print(f"No ignition detected at {temperature} K.")

    t1 = time.time()

    print(f"IDT：{ignition_delays_ms[i]:.3f} ms    T={temperature}K    Took {t1-t0:.2f}s to compute.")


# 显示实验点
excel_file = 'E:/Desktop/Postgraduate/mech/simplify/Shu_exp_used_for_Bertolino.xlsx'
df = pd.read_excel(excel_file)
# 从Excel文件中读取并突出显示特定的点
temperatures_inv = df['1000/T (1/K)'].values  # (1000/T)
log_ignition_delays_exp = df['Ignition delay time (ms)'].values  # 对数点火延迟时间

# 绘制图形
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.plot(1000 / T, ignition_delays_ms, '-.', linewidth=2.0, color='r', label='Bertolino_mech')
ax.scatter(temperatures_inv, log_ignition_delays_exp, color='red', zorder=5, label='Exp') #  zorder参数确保点在最上层

ax.set_xlabel(r'1000/T (K$^\mathdefault{-1}$)', fontsize=14)
ax.set_ylabel(r'Log10(IDT) (ms)', fontsize=14)
ax.set_yscale('log')  # 设置y轴为对数坐标
ax.set_ylim([0.01, 10])
ax.set_xlim([0.6, 0.85])  # 设置x轴的范围

ax2 = ax.twiny()
ticks = ax.get_xticks()
ax2.set_xticks(ticks)
ax2.set_xticklabels((1000/ticks).round(1))
ax2.set_xlim(ax.get_xlim())
ax2.set_xlabel('Temperature (K)', fontsize=14)

ax.legend(['bertolino_mech'], frameon=False, loc='upper left')
plt.savefig('D:/Users/XuXT/PycharmProjects/pythonProject/Fig/ignition_process.png',
            dpi=300, bbox_inches='tight')
plt.show()
