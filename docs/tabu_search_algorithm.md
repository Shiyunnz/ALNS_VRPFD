# Tabu Search 算法详细流程

本文档详细描述仓库中 Tabu Search 算法的实现流程，该算法融合了 ALNS 风格的算子和多客户无人机任务优化。

---

## 1. 算法概述

该实现是增强版的禁忌搜索算法，主要特点：

- **ALNS 风格算子**：包含破坏-重建 (Ruin-and-Recreate) 邻域
- **多客户无人机任务**：支持单次无人机飞行服务多个客户
- **自适应机制**：禁忌期、邻域大小根据问题规模自动调整
- **可行性优先**：即使初始解不可行，算法仍能搜索可行邻域

---

## 2. 核心组件

| 组件 | 文件位置 | 作用 |
|------|----------|------|
| `TabuSearch` | `tabu_search.py` | 主算法类，实现禁忌搜索流程 |
| `DroneTaskOptimizer` | `tabu_search.py` | ALNS 风格的无人机任务优化器 |
| `TabuMove` | `optimized_tabu_search.py` | 禁忌移动的数据结构 |
| `OptimizedTabuSearch` | `optimized_tabu_search.py` | 性能优化版本 |

---

## 3. 算法主流程

### 3.1 伪代码

```
Algorithm: Enhanced Tabu Search for VRPFD
Input: initial_solution, time_limit, max_iterations
Output: best_solution

1. INITIALIZE
   current ← initial_solution
   best ← current
   tabu_list ← empty deque(maxlen=tabu_tenure)
   stagnation_counter ← 0

2. MAIN LOOP (for iteration = 1 to max_iterations)
   2.1 IF time_limit exceeded OR feasible_stagnation ≥ 200:
       BREAK

   2.2 GENERATE NEIGHBORS using 6 generators:
       - Truck Relocate
       - Truck Swap  
       - Drone Moves (×2 weight)
       - Ruin-and-Recreate (ALNS-style)
       - Or-opt Chain Moves

   2.3 FOR each neighbor in shuffled(neighbors):
       a) Check if move is in tabu_list
       b) Evaluate neighbor cost and feasibility
       c) IF infeasible: try quick_repair()
       d) IF still infeasible: skip
       e) ASPIRATION: if tabu but cost < best_cost: accept
       f) Track best_neighbor if not tabu and best cost

   2.4 UPDATE STATE
       current ← best_neighbor
       tabu_list.append(move)
       
       IF current improves best:
           Apply 2-opt local search
           Apply drone optimization
           best ← current
           stagnation_counter ← 0
       ELSE:
           stagnation_counter += 1

   2.5 DIVERSIFICATION (if stagnation > threshold)
       current ← perturb(best)
       Apply 2-opt
       Clear tabu_list

3. RETURN best
```

### 3.2 流程图

```
┌─────────────────────────────────────────────────────────────┐
│                      初始化阶段                               │
├─────────────────────────────────────────────────────────────┤
│ • 评估初始解（处理不可行解情况）                                │
│ • 初始化禁忌表 (deque, maxlen=tabu_tenure)                   │
│ • 设置 best = current = initial_solution                     │
│ • 初始化停滞计数器                                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    迭代循环开始                               │
├─────────────────────────────────────────────────────────────┤
│ for iteration in range(max_iterations):                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  终止条件检查    │
                    │ • 时间限制       │
                    │ • 停滞次数过多   │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │ 否                          │ 是
              ▼                              ▼
┌─────────────────────────────────┐    ┌──────────┐
│      邻域生成 (6种生成器)         │    │ 返回best │
├─────────────────────────────────┤    └──────────┘
│ 1. _gen_truck_relocate          │
│ 2. _gen_truck_swap              │
│ 3. _gen_drone_moves (×2)        │
│ 4. _gen_ruin_recreate           │
│ 5. _gen_or_opt                  │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    邻域评估                                   │
├─────────────────────────────────────────────────────────────┤
│ for each (neighbor, move) in shuffled(potential_moves):     │
│   ├── 检查移动是否在禁忌表中                                  │
│   ├── 评估邻域解的成本和可行性                                │
│   ├── 若不可行 → 尝试快速修复 (_quick_repair)                 │
│   ├── 仍不可行 → 跳过                                        │
│   ├── 禁忌破例: 若 cost < best_cost 则接受禁忌移动            │
│   └── 记录最佳邻域解                                         │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    状态更新                                   │
├─────────────────────────────────────────────────────────────┤
│ IF best_neighbor exists:                                    │
│   current ← best_neighbor                                   │
│   tabu_list.append(move)                                    │
│                                                             │
│   IF current improves best:                                 │
│     ├── 应用 2-opt 局部搜索                                  │
│     ├── 应用无人机优化 (_apply_drone_optimization)           │
│     ├── best ← current                                      │
│     └── stagnation_counter ← 0                              │
│   ELSE:                                                     │
│     └── stagnation_counter += 1                             │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                  多样化/重启机制                              │
├─────────────────────────────────────────────────────────────┤
│ IF stagnation_counter > restart_threshold:                  │
│   ├── current ← perturb(best)  # 扰动最优解                  │
│   ├── 应用 2-opt 改进                                        │
│   ├── 清空禁忌表                                             │
│   └── stagnation_counter ← 0                                │
└────────────────┬────────────────────────────────────────────┘
                 │
                 └──────────────► 返回迭代循环开始
```

---

## 4. 邻域生成策略

### 4.1 卡车路线移动

#### Relocate (重定位)
将一个客户从源路线移动到目标路线的某个位置。

```python
def _gen_truck_relocate(solution):
    for each route_src in routes:
        for each customer in route_src:
            for each route_dst in routes (dst ≠ src):
                for each position in route_dst:
                    yield (new_solution, ("relocate", customer, src, dst))
```

#### Swap (交换)
交换两条不同路线中的客户。

```python
def _gen_truck_swap(solution):
    for route1, route2 in random_route_pairs:
        c1 = random_customer(route1)
        c2 = random_customer(route2)
        swap(c1, c2)
        yield (new_solution, ("swap", c1, c2))
```

#### Or-opt (链式移动)
移动 2-3 个连续客户组成的链到另一个位置。

```python
def _gen_or_opt(solution):
    for chain_len in [2, 3]:
        for each route:
            for each chain starting position:
                for each target route and position:
                    move chain to new position
                    yield (new_solution, ("or_opt", chain_len, chain))
```

### 4.2 无人机移动（ALNS 风格）

#### 多客户卡车→无人机
识别卡车路线上相邻的无人机可服务客户，创建多客户无人机任务。

```python
def _gen_drone_moves(solution):
    # 策略1: 多客户任务
    for launch_pos in route:
        for retrieve_pos in range(launch_pos+2, launch_pos+5):
            candidates = customers_between(launch_pos, retrieve_pos)
            task = create_multi_customer_task(candidates, max=2)
            if task:
                yield (new_solution, ("multi_truck_to_drone", customers))
    
    # 策略2: 单客户任务 (回退)
    for customer in drone_eligible:
        create_single_drone_task(customer)
        yield (new_solution, ("truck_to_drone", customer))
    
    # 策略3: 无人机→卡车
    for task in drone_tasks:
        move task.customers back to truck
        yield (new_solution, ("drone_to_truck", customers))
```

### 4.3 破坏-重建 (Ruin-and-Recreate)

ALNS 风格的大邻域搜索，移除 15-25% 的客户后贪婪重插。

| 策略 | 描述 | 移除规则 |
|------|------|----------|
| **随机移除** | 随机选择客户移除 | 均匀随机 |
| **最差绕行** | 移除造成最大绕行成本的客户 | 按绕行成本排序 |
| **Shaw 移除** | 移除空间上相近的客户 | 以种子客户为中心的最近邻 |

```python
def _gen_ruin_recreate(solution):
    remove_count = max(3, 20% of customers)
    
    # 策略1: 随机移除
    for _ in range(7):
        removed = random_sample(customers, remove_count)
        greedy_reinsert(removed)
        yield solution
    
    # 策略2: 最差绕行移除
    detour_scores = calculate_detours()
    worst = top_k(detour_scores, remove_count)
    greedy_reinsert(worst)
    yield solution
    
    # 策略3: Shaw 移除
    seed = random_customer()
    shaw_removed = seed + nearest_neighbors(seed, remove_count-1)
    greedy_reinsert(shaw_removed)
    yield solution
```

---

## 5. 关键参数

### 5.1 自动调整参数

```python
# 根据客户数 n 自动计算
tabu_tenure = max(10, int(sqrt(n) * 2))      # 禁忌期
max_stagnation = min(200, max(50, int(sqrt(n) * 5)))  # 最大停滞次数
max_neighbors = 100 if n < 50 else 150       # 最大邻域评估数
moves_per_type = 5 if n < 30 else 10         # 每类邻域的移动数
```

### 5.2 固定参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `max_iterations` | 1000 | 最大迭代次数 |
| `feasible_stagnation` | 200 | 可行解停滞早期终止阈值 |
| `restart_threshold` | `max(30, max_stagnation//2)` | 多样化重启阈值 |
| `remove_ratio` | 20% | 破坏-重建移除比例 |

---

## 6. 特殊机制

### 6.1 禁忌破例准则 (Aspiration Criterion)

即使移动在禁忌表中，若其产生的解优于当前全局最优，则解除禁忌。

```python
if is_tabu and cost < best_cost:
    is_tabu = False  # 解除禁忌
```

### 6.2 自适应禁忌期

每次迭代在基础禁忌期 ±5 范围内随机变化，避免循环。

```python
current_tenure = base_tabu_tenure + random.randint(-5, 5)
```

### 6.3 快速修复 (Quick Repair)

当邻域解不可行时，移除所有无人机任务，将客户转回卡车路线。

```python
def _quick_repair(solution):
    for task in drone_tasks:
        for customer in task.customers():
            insert_to_best_truck_position(customer)
    drone_tasks.clear()
    return solution
```

### 6.4 多样化重启

长期停滞时扰动最优解：

```python
def _perturb(solution):
    n_moves = random(n//10, n//5)  # 扰动强度随问题规模增加
    for _ in range(n_moves):
        move_type = random(['swap', 'relocate', 'drone_shuffle'])
        apply_random_move(move_type)
    return solution
```

### 6.5 无人机优化 (Drone Optimization)

找到新最优解后，积极尝试创建多客户无人机任务：

```python
def _apply_drone_optimization(solution):
    for _ in range(max_rounds=10):
        tasks_created = 0
        for route in truck_routes:
            for candidate_customers in route:
                task = try_create_drone_task(candidates)
                if task:
                    tasks_created += 1
        if tasks_created == 0:
            break
    return solution
```

---

## 7. 无人机任务评分 (ALNS 风格)

`DroneTaskOptimizer` 使用能量利用率评分：

```python
def _score_drone_task(customers, launch, retrieve):
    # 计算能耗
    distance = calculate_drone_distance(launch, customers, retrieve)
    energy_used = distance * energy_rate
    
    # 能量利用率奖励
    utilization = energy_used / drone_battery_capacity
    score = utilization * 10.0
    
    # 多客户奖励
    if len(customers) >= 2:
        score += 5.0 * len(customers)
    
    # 仓库发射奖励
    if launch == depot:
        score += 1.5
    
    return score
```

---

## 8. 文件结构

```
heuristics/tabu_search/
├── tabu_search.py              # 主实现 (TabuSearch, DroneTaskOptimizer)
├── optimized_tabu_search.py    # 性能优化版本 (OptimizedTabuSearch)
├── enhanced_tabu_search.py     # 增强版本 (EnhancedTabuSearch)
├── simple_tabu_search.py       # 简化版本
├── truck_focused_tabu_search.py # 卡车优先版本
├── run_tabu.py                 # 运行脚本
└── test_enhanced_tabu.py       # 测试文件
```

---

## 9. 使用示例

```python
from heuristics.tabu_search.tabu_search import TabuSearch
from alns_vrpfd.evaluation import Evaluator

# 初始化
evaluator = Evaluator(instance)
tabu = TabuSearch(
    evaluator=evaluator,
    tabu_tenure=20,
    max_iterations=1000,
    max_stagnation=100
)

# 运行
best_solution = tabu.run(initial_solution, time_limit=300)
```

---

## 10. 与 ALNS 的主要区别

| 特性 | Tabu Search | ALNS |
|------|-------------|------|
| 接受准则 | 禁忌表 + 禁忌破例 | 模拟退火概率接受 |
| 邻域结构 | 固定6种生成器 | 自适应算子选择 |
| 算子权重 | 均匀/固定 | 根据历史表现自适应 |
| 多样化 | 扰动 + 清空禁忌表 | 重热机制 |
| 强化搜索 | 2-opt + 无人机优化 | 局部搜索模块 |
