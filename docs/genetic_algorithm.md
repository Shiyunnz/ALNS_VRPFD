# Genetic Algorithm (GA) 详细流程

本文档详细描述仓库中遗传算法的实现流程，该算法融合了 ALNS 风格的无人机任务优化和自适应参数调整机制。

---

## 1. 算法概述

该实现是增强版的遗传算法，主要特点：

- **ALNS 风格的无人机优化**：多轮次积极创建无人机任务
- **完善的可行性修复**：自动修复容量、时间窗、无人机冲突等违规
- **自适应参数调整**：根据种群多样性动态调整交叉/变异率
- **多种交叉策略**：Order Crossover (OX) 和 Route Crossover
- **丰富的变异算子**：包含 ALNS 风格的破坏-重建变异

---

## 2. 核心组件

| 组件 | 作用 |
|------|------|
| `GeneticAlgorithm` | 主算法类，实现 GA 主循环 |
| `GAConfig` | 算法参数配置类 |
| `Individual` | 个体表示类（解+适应度） |
| `FeasibilityRepair` | 可行性修复器 |
| `DroneChainBuilder` | ALNS 风格的无人机任务构建器 |

---

## 3. 算法主流程

### 3.1 伪代码

```
Algorithm: Enhanced Genetic Algorithm for VRPFD
Input: instance, config, initial_solution (optional)
Output: best_individual

1. INITIALIZE
   population ← initialize_population()
   IF initial_solution provided:
       Replace worst individual with initial_solution
   population.sort()
   best_individual ← population[0]
   
2. MAIN LOOP (for generation = 1 to max_generations)
   2.1 CHECK TIME LIMITS
   
   2.2 CREATE NEW POPULATION
       new_population ← elite individuals  # Elitism
       
       WHILE |new_population| < population_size:
           parent1 ← tournament_selection()
           parent2 ← tournament_selection()
           
           # Crossover
           IF random() < crossover_rate:
               child1, child2 ← crossover(parent1, parent2)
               repair(child1), repair(child2)
           ELSE:
               child1, child2 ← parent1.clone(), parent2.clone()
           
           # Mutation
           IF random() < mutation_rate * 1.5:
               child1 ← mutate(child1)
               repair(child1)
           IF random() < mutation_rate * 1.5:
               child2 ← mutate(child2)
               repair(child2)
           
           # Local Search (2-opt, 80% probability)
           apply_2opt(child1), apply_2opt(child2)
           
           # Evaluate and add to population
           evaluate(child1), evaluate(child2)
           new_population.add(child1, child2)
       
       population ← ensure_diversity(new_population)
   
   2.3 UPDATE BEST INDIVIDUAL
       IF population[0].fitness < best_individual.fitness:
           best_individual ← population[0]
           stagnation_counter ← 0
       ELSE:
           stagnation_counter += 1
   
   2.4 LOCAL SEARCH (every 1-2 generations)
       apply_local_search(elite individuals)
   
   2.5 AGGRESSIVE DRONE OPTIMIZATION (every generation)
       FOR each individual in population:
           IF drone_tasks < 5:
               individual ← optimize_drone_tasks(individual)
   
   2.6 ADAPTIVE PARAMETER ADJUSTMENT
       IF generation % adaptation_interval == 0:
           adapt_parameters()
   
   2.7 CHECK EARLY STOPPING
       IF stagnation_counter >= max_stagnation:
           BREAK

3. FINAL REPAIR AND OPTIMIZATION
   repair(best_individual)
   optimize_drone_tasks(best_individual)
   
4. RETURN best_individual
```

### 3.2 流程图

```
┌─────────────────────────────────────────────────────────────┐
│                      初始化阶段                               │
├─────────────────────────────────────────────────────────────┤
│ • 生成初始种群 (population_size 个个体)                        │
│   - 使用两阶段启发式生成基础解                                  │
│   - 30% 个体应用无人机优化                                     │
│   - 50% 个体应用变异增加多样性                                  │
│ • 评估所有个体适应度                                           │
│ • 按适应度排序，记录最优个体                                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    迭代循环开始                               │
├─────────────────────────────────────────────────────────────┤
│ for generation in range(generations):                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  终止条件检查    │
                    │ • 时间限制       │
                    │ • 停滞代数过多   │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │ 否                          │ 是
              ▼                              ▼
┌─────────────────────────────────┐    ┌──────────┐
│       创建新种群                 │    │ 返回best │
├─────────────────────────────────┤    └──────────┘
│ 1. 精英保留 (elite_size个)       │
│ 2. 循环直到种群填满:             │
│    ├── 锦标赛选择 × 2           │
│    ├── 交叉 (概率 crossover_rate)│
│    │   ├── Order Crossover      │
│    │   └── Route Crossover      │
│    ├── 修复 (repair)            │
│    ├── 变异 (概率 mutation_rate) │
│    ├── 修复 (repair)            │
│    ├── 2-opt 局部搜索 (80%)      │
│    └── 评估适应度                │
│ 3. 确保种群多样性                │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    更新最优个体                               │
├─────────────────────────────────────────────────────────────┤
│ IF population[0] 优于 best_individual:                      │
│   best_individual ← population[0]                           │
│   stagnation_counter ← 0                                    │
│ ELSE:                                                       │
│   stagnation_counter += 1                                   │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                 局部搜索 (每1-2代)                            │
├─────────────────────────────────────────────────────────────┤
│ 对精英个体应用:                                               │
│ ├── 2-opt 改进                                              │
│ ├── 路线内重定位                                             │
│ ├── 路线间重定位                                             │
│ ├── 无人机优化                                               │
│ └── 无人机链优化 (ALNS风格)                                   │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│              积极无人机优化 (每代)                             │
├─────────────────────────────────────────────────────────────┤
│ FOR each individual in population:                          │
│   IF drone_tasks < 5:                                       │
│     ├── 应用多轮无人机任务创建                                 │
│     ├── 修复并评估                                           │
│     └── 接受改进或轻微代价增加的解                             │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│               自适应参数调整                                  │
├─────────────────────────────────────────────────────────────┤
│ IF generation % adaptation_interval == 0:                   │
│   IF diversity < threshold:                                 │
│     ├── mutation_rate × 1.1 (增加探索)                       │
│     └── crossover_rate × 0.95                               │
│   ELIF diversity > threshold × 2:                           │
│     ├── mutation_rate × 0.9 (减少探索)                       │
│     └── crossover_rate × 1.05                               │
└────────────────┬────────────────────────────────────────────┘
                 │
                 └──────────────► 返回迭代循环开始
```

---

## 4. 种群初始化

```python
def initialize_population():
    population = []
    for i in range(population_size):
        # 使用两阶段启发式生成初始解
        solution = build_two_phase_initial_solution(instance)
        solution = repair(solution)
        
        # 30% 个体应用无人机优化
        if i < population_size * 0.3:
            solution = drone_builder.optimize_drone_tasks(solution)
            solution = repair(solution)
        
        # 50% 个体应用变异增加多样性
        if i > population_size // 2:
            solution = mutate(solution)
            solution = repair(solution)
        
        individual = Individual(solution)
        evaluate(individual)
        population.append(individual)
    
    population.sort()
    return population
```

---

## 5. 选择算子

### 锦标赛选择 (Tournament Selection)

```python
def _tournament_selection():
    # 随机选择 tournament_size 个个体
    candidates = random.sample(population, tournament_size)
    # 返回适应度最好的个体
    return min(candidates, key=lambda x: x.fitness).solution
```

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `tournament_size` | 5~7 | 锦标赛规模，越大选择压力越大 |

---

## 6. 交叉算子

### 6.1 Order Crossover (OX)

用于卡车路线的客户顺序交叉。

```python
def _order_crossover(route1, route2):
    customers1 = get_customers(route1)  # 不含仓库
    customers2 = get_customers(route2)
    
    # 随机选择交叉段
    start, end = sorted(random.sample(range(len(customers1)), 2))
    
    # 子代1: 继承父代1的交叉段，其余按父代2顺序填充
    offspring1 = [None] * len(customers1)
    offspring1[start:end+1] = customers1[start:end+1]
    
    fill_pos = (end + 1) % len(customers1)
    for cust in customers2:
        if cust not in offspring1:
            offspring1[fill_pos] = cust
            fill_pos = (fill_pos + 1) % len(customers1)
    
    # 子代2 同理
    ...
    
    return TruckRoute(offspring1), TruckRoute(offspring2)
```

### 6.2 Route Crossover

继承父代1的完整路线，用父代2的客户填充剩余需求。

```python
def _route_crossover(parent1, parent2):
    child = parent1.clone()
    
    # 收集所有客户
    all_customers = set(all customer IDs)
    served_in_child = customers_in(child)
    
    # 移除父代2中与子代重复的客户
    missing = all_customers - served_in_child
    
    # 将缺失客户插入最佳位置
    for customer in missing:
        insert_cheapest(child, customer)
    
    return child
```

### 6.3 无人机任务交叉

```python
def drone_crossover(parent1, parent2):
    # 随机从两个父代中选择无人机任务
    # 避免无人机 ID 冲突
    used_drones = set()
    for task in shuffled(parent1.drone_tasks + parent2.drone_tasks):
        if task.drone_id not in used_drones:
            child.drone_tasks.append(task)
            used_drones.add(task.drone_id)
```

---

## 7. 变异算子

### 7.1 变异类型分布

| 概率范围 | 变异类型 | 描述 |
|----------|----------|------|
| 0 ~ 0.3 | Swap 变异 | 随机交换同路线两客户位置 |
| 0.3 ~ 0.6 | 破坏-重建 | ALNS 风格，移除 15% 客户后重插 |
| 0.6 ~ 0.8 | 卡车→无人机 | 将客户从卡车转移到无人机 |
| 0.8 ~ 1.0 | 无人机变异 | 移除或重新分配无人机任务 |

### 7.2 破坏-重建变异 (ALNS 风格)

```python
def ruin_recreate_mutation(solution):
    # 收集所有卡车客户
    all_customers = collect_truck_customers(solution)
    
    # 移除 10-20% 客户
    num_remove = max(2, int(len(all_customers) * 0.15))
    removed = random.sample(all_customers, num_remove)
    
    # 从所有路线中移除
    for route in solution.truck_routes:
        route.nodes = [n for n in route.nodes if n not in removed]
    
    # 使用最便宜插入重新插入
    for customer in removed:
        insert_cheapest(solution, customer)
    
    return solution
```

### 7.3 卡车→无人机变异

```python
def truck_to_drone_mutation(solution):
    route = random.choice(truck_routes)
    
    # 找到无人机可服务的客户
    eligible = [c for c in route if demand[c] <= drone_capacity]
    
    if eligible:
        customer = random.choice(eligible)
        route.remove(customer)
        
        # 找到可用的无人机
        available_drone = find_available_drone()
        
        if available_drone:
            task = DroneTask(
                drone_id=available_drone,
                launch_node=depot_start,
                retrieve_node=depot_end,
                customers=[customer]
            )
            solution.drone_tasks.append(task)
```

---

## 8. 可行性修复 (FeasibilityRepair)

修复顺序：

```
1. 移除重复客户 (_remove_duplicates)
       ↓
2. 修复容量违规 (_repair_capacity_violations)
       ↓  
3. 修复时间窗违规 (_repair_time_window_violations)
       ↓
4. 修复无人机分配冲突 (_repair_drone_assignment_conflicts)
       ↓
5. 修复无人机续航违规 (_repair_drone_endurance_violations)
       ↓
6. 确保所有客户被服务 (_repair_missing_customers)
```

### 8.1 容量违规修复

```python
def _repair_capacity_violations(solution):
    for route in truck_routes:
        total_demand = sum(demands[c] for c in route.customers)
        
        while total_demand > truck_capacity:
            # 移除需求最大的客户
            worst = max(route.customers, key=lambda c: demands[c])
            route.remove(worst)
            total_demand -= demands[worst]
            
            # 将客户插入其他路线或创建新路线
            insert_to_other_route(worst)
```

### 8.2 无人机续航修复

```python
def _repair_drone_endurance_violations(solution):
    for task in drone_tasks:
        distance = calculate_drone_distance(task)
        time = distance / drone_speed
        
        if time > drone_endurance:
            if len(task.customers) > 1:
                # 移除最远的客户
                farthest = max(task.customers, key=distance_from_depot)
                task.customers.remove(farthest)
                insert_to_truck(farthest)
            else:
                # 整个任务无效，转回卡车
                solution.drone_tasks.remove(task)
                insert_to_truck(task.customers[0])
```

---

## 9. 无人机链构建器 (DroneChainBuilder)

ALNS 风格的多轮次积极无人机任务创建。

### 9.1 优化流程

```python
def optimize_drone_tasks(solution):
    optimized = solution.clone()
    
    for round in range(max_rounds=5):
        # 获取当前无人机可服务但由卡车服务的客户
        eligible = get_drone_eligible_customers(optimized)
        
        if not eligible:
            break
        
        tasks_created = 0
        
        for route in truck_routes:
            # 尝试不同的发射/回收点组合
            best_task = None
            best_score = -inf
            
            for launch_pos in route:
                for retrieve_pos in route[launch_pos+2:]:
                    candidates = customers_between(launch_pos, retrieve_pos)
                    candidates = [c for c in candidates if c in eligible]
                    
                    if candidates:
                        task = build_drone_task(launch, retrieve, candidates, max=3)
                        score = len(task.customers)
                        
                        if score > best_score:
                            best_task = task
                            best_score = score
            
            if best_task:
                # 从卡车路线移除客户
                for cust in best_task.customers:
                    route.remove(cust)
                    eligible.discard(cust)
                
                optimized.drone_tasks.append(best_task)
                tasks_created += 1
        
        if tasks_created == 0:
            break
    
    return optimized
```

### 9.2 任务评分 (ALNS 风格)

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

## 10. 局部搜索

### 10.1 应用于精英个体的局部搜索

| 算子 | 描述 |
|------|------|
| `_local_search_2opt` | 路线内 2-opt 改进 |
| `_local_search_relocate` | 路线内客户重定位 |
| `_local_search_relocate_inter_route` | 路线间客户重定位 |
| `_local_search_drone_optimization` | 尝试添加无人机任务 |
| `_local_search_drone_chain` | ALNS 风格的无人机链优化 |

### 10.2 积极无人机优化

每代对**所有个体**应用：

```python
def _apply_aggressive_drone_optimization():
    for individual in population:
        if len(individual.drone_tasks) < 5:
            optimized = drone_builder.optimize_drone_tasks(individual.solution)
            optimized = repair(optimized)
            
            result = evaluate(optimized)
            
            # 接受改进 或 接受增加无人机但代价增加不超过10%
            if result.cost < individual.fitness or \
               (added_drones and result.cost < individual.fitness * 1.10):
                individual.solution = optimized
                individual.fitness = result.cost
```

---

## 11. 自适应参数调整

```python
def _adapt_parameters():
    current_diversity = calculate_diversity()
    avg_diversity = mean(recent_diversities)
    
    if avg_diversity < diversity_threshold:
        # 多样性过低：增加探索
        mutation_rate *= 1.1  # 上限 0.3
        crossover_rate *= 0.95  # 下限 0.5
    
    elif avg_diversity > diversity_threshold * 2:
        # 多样性过高：增加开发
        mutation_rate *= 0.9  # 下限 0.01
        crossover_rate *= 1.05  # 上限 0.95
```

---

## 12. 配置参数

### 12.1 GAConfig 默认值

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `population_size` | 100 | 种群大小 |
| `generations` | 100 | 最大代数 |
| `tournament_size` | 5 | 锦标赛规模 |
| `crossover_rate` | 0.8 | 交叉概率 |
| `mutation_rate` | 0.1 | 变异概率 |
| `elite_size` | 5 | 精英个体数量 |
| `max_stagnation` | 20 | 最大停滞代数 |
| `truck_route_crossover_rate` | 0.7 | 卡车路线交叉概率 |
| `drone_task_mutation_rate` | 0.3 | 无人机任务变异概率 |
| `adaptive_enabled` | True | 是否启用自适应调整 |
| `adaptation_interval` | 5 | 自适应调整间隔（代数） |
| `diversity_threshold` | 2.0 | 多样性阈值 |

### 12.2 配置文件 (config_ga.json)

```json
{
  "population_size": 150,
  "generations": 1500,
  "tournament_size": 7,
  "crossover_rate": 0.85,
  "mutation_rate": 0.15,
  "elite_size": 8,
  "max_stagnation": 100,
  "truck_route_crossover_rate": 0.8,
  "drone_task_mutation_rate": 0.25,
  "route_segment_swap_rate": 0.5,
  "time_limit": 600.0,
  "generation_time_limit": 10.0,
  "adaptive_enabled": true,
  "adaptation_interval": 50,
  "diversity_threshold": 10.0
}
```

---

## 13. Individual 数据结构

```python
@dataclass
class Individual:
    solution: Solution          # 解
    fitness: float = inf        # 适应度（总成本）
    feasible: bool = False      # 是否可行
    evaluation_time: float = 0  # 评估耗时
    truck_distance: float = 0   # 卡车行驶距离成本
    drone_distance: float = 0   # 无人机行驶距离成本
    delay_penalty: float = 0    # 延迟惩罚
```

---

## 14. 文件结构

```
heuristics/ga/
├── ga.py                # 主实现
│   ├── FeasibilityRepair    # 可行性修复器
│   ├── DroneChainBuilder    # 无人机链构建器
│   ├── GAConfig             # 参数配置
│   ├── Individual           # 个体表示
│   └── GeneticAlgorithm     # 主算法类
├── config_ga.json       # 配置文件
└── run_ga.py            # 运行脚本
```

---

## 15. 使用示例

```python
from heuristics.ga.ga import GeneticAlgorithm, GAConfig
from alns_vrpfd.evaluation import Evaluator

# 初始化配置
config = GAConfig(
    population_size=100,
    generations=200,
    crossover_rate=0.85,
    mutation_rate=0.15,
    elite_size=5
)

# 初始化算法
evaluator = Evaluator(instance)
ga = GeneticAlgorithm(
    instance=instance,
    config=config,
    evaluator=evaluator
)

# 运行
best_individual = ga.run(initial_solution=None)
print(f"Best cost: {best_individual.fitness}")
print(f"Feasible: {best_individual.feasible}")
```

---

## 16. 与其他算法的对比

| 特性 | GA | ALNS | Tabu Search |
|------|-----|------|-------------|
| 搜索策略 | 种群进化 | 单解迭代 + 自适应算子 | 单解迭代 + 禁忌表 |
| 接受准则 | 适应度排序 | 模拟退火概率 | 禁忌表 + 禁忌破例 |
| 多样性维护 | 种群 + 自适应参数 | 重热机制 | 多样化重启 |
| 无人机优化 | 每代积极优化 | 修复算子内嵌 | 局部搜索内嵌 |
| 并行潜力 | 高（种群可并行评估） | 低 | 低 |
| 收敛速度 | 较慢 | 快 | 中等 |
