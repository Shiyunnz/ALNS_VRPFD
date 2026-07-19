# Matheuristic LNS 更新总结与论文写作材料

本文档整理本轮 ALNS 改进中可以写入论文的方法、实验和讨论内容。核心贡献是引入一种 matheuristic large neighborhood search (MLNS) final-polishing 机制，用小规模穷举重构无人机 sortie，以处理普通增量式 ALNS 邻域难以发现的跨车、多客户无人机任务结构。

## 1. 本轮更新的核心动机

在 `R_40_10_1` 实例中，原 ALNS 最优解与 MILP 最优解之间存在一个很小但稳定的结构性差距。ALNS 可达到的代表性最优成本为 97.60，而 MILP 给出的最优成本为 97.18，差距约为 0.43%。进一步分析发现，这一差距并不是由卡车路径的简单排序错误造成，也不是单个无人机任务的局部锚点调整能够消除，而是来自一个耦合的三维组合分配问题：客户组合、无人机任务分配以及发射/回收卡车锚点必须同时改变。

MILP 最优结构包含一个跨车、多客户无人机 sortie，即 `T1@5 -> [8,10] -> T0@1`，同时保留一个单客户任务 `T1@4 -> [6] -> T1@5`。相比之下，ALNS 的局部最优结构为 `T1@4 -> [6,8] -> T1@5`，并将客户 10 单独安排为返回车场的无人机任务。这个差异说明，若只依赖拆分、合并、重新锚定或 2-opt 类局部搜索，搜索过程很容易停留在局部结构中，而难以同时重构客户配对、sortie 分组和跨车同步关系。

## 2. 方法层面的主要改动

本轮新增了一个 matheuristic LNS repair/polish 算子，文件为 `alns_vrpfd/core/operators/matheuristic_lns.py`。该算子不是在现有无人机任务上做单点微调，而是先选取一个小规模关键邻域，删除其中的客户服务关系，再对这些客户进行精确或近似精确的局部重构。其搜索空间包括客户集合划分、每个集合内的访问顺序、发射锚点、回收锚点和无人机编号分配。

具体而言，`MatheuristicLNSRepair` 随机选择 2 至 4 个无人机服务客户作为邻域，优先考虑多客户任务中的客户，并用单客户任务中的客户补足邻域规模。随后，算子从当前解中删除这些客户对应的无人机任务和卡车访问，再枚举该邻域客户的所有集合划分。对每个客户块，算子枚举客户访问排列、可行发射锚点、可行回收锚点以及无人机分配。每个候选 sortie 都先经过需求容量和鲁棒能耗的快速过滤，再用完整 `Evaluator` 检查同步、时间窗、锚点冲突、客户覆盖和鲁棒能耗约束。只有完整评估可行且成本更低的候选解才会被接受。

该方法属于 matheuristic 的原因在于，它嵌入在启发式 ALNS 框架中，但局部 repair 阶段使用了精确枚举来求解一个小型组合子问题。这样既保留了 ALNS 在全局探索中的灵活性，又允许算法在收敛后针对困难的跨车无人机结构进行高强度局部优化。

## 3. 与已有改进步骤的关系

本轮同时整理并验证了多个候选局部增强模块。Step6 的无人机任务 split/merge/re-anchor local search 是最有效的低成本增强，它将解的中位数和稳定性推向 97.60 附近。Step7 的复合 re-anchor、Step8 的多客户 sortie constructor、Step9 的同步卡车路径优化以及 mini-MILP truck polish 都未能稳定关闭 97.60 与 97.18 之间的差距。原因在于，这些方法要么仍然依赖已有无人机任务结构，要么只固定无人机任务后优化卡车路径，无法同时改变客户组合、无人机任务分配和跨车锚点。

MLNS 的定位因此不是替代普通局部搜索，而是作为 final-polishing 阶段处理少数结构性瓶颈。由于该算子需要枚举大量候选结构，默认配置中 `matheuristic_lns_enabled` 被设为 `false`，普通 ALNS 迭代中不启用。正式对比实验中仅在 C 组使用 `ALNS + Step6 + MLNS final polish`。

## 4. Oracle 验证结果

为了验证 97.18 是否能够从 97.60 的 ALNS 解通过局部无人机 sortie 重构达到，新增了离线 oracle 脚本 `scripts/oracle_cross_truck_sortie.py`。该脚本固定 ALNS 的卡车路线，并只对关键客户集合 `{6,8,10}` 进行穷举重构。

Oracle 共枚举 319,440 个局部组合，其中 717 个通过完整可行性检查。结果表明，在不改变卡车路线的情况下，MILP 最优结构可以从 ALNS 的 97.60 解中恢复出来。最佳局部重构为 `D0: [4 -> [6] -> 5]` 与 `D0: [5 -> [8,10] -> 1]`，成本为 97.18，与 MILP 最优成本一致。

| 指标 | 数值 |
|---|---:|
| ALNS 基准局部最优成本 | 97.60 |
| MILP 最优成本 | 97.18 |
| Oracle 最优重构成本 | 97.18 |
| 穷举候选数 | 319,440 |
| 可行候选数 | 717 |
| 相对 MILP gap | 0.00% |

这一 oracle 结果证明，原 ALNS 与 MILP 的差距不是由于成本函数或评估器不一致导致的，而是由于原有 ALNS 邻域无法有效覆盖该类跨车、多客户 sortie 结构。

## 5. A/B/C/D 正式对照实验

正式实验在 `R_40_10_1` 上进行，采用 5 个随机种子 `42-46`，每个 ALNS 运行 4000 次迭代。对照组定义如下：A 为 ALNS baseline，不启用 Step6；B 为 ALNS + Step6；C 为 ALNS + Step6 + MLNS final polish；D 为 MILP 对照。完整结果保存在 `results/final_comparison.json`。

| 方法 | Mean | Median | Min | Std | 命中 97.18 | 命中 97.60 | < 98 | Best gap vs. MILP | 平均运行时间(s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A: ALNS baseline | 98.88 | 97.76 | 97.60 | 2.02 | 0/5 | 1/5 | 3/5 | +0.43% | 6.03 |
| B: ALNS + Step6 | 97.92 | 97.76 | 97.60 | 0.31 | 0/5 | 1/5 | 3/5 | +0.43% | 5.84 |
| C: ALNS + Step6 + MLNS polish | 99.02 | 97.86 | 97.18 | 2.50 | 1/5 | 0/5 | 3/5 | +0.00% | 36.17 |
| D: MILP | 97.18 | - | 97.18 | - | - | - | - | 0.00% | 120.11 |

实验表明，Step6 能够提高 ALNS 的稳定性，使成本分布更加集中，但仍无法突破 97.60 的结构性瓶颈。MLNS final polish 在 seed 44 中将 ALNS 解从 97.60 改善至 97.18，完全关闭了相对于 MILP 的 best gap。该结果说明，MLNS 具有发现跨车、多客户无人机 sortie 的能力，但其平均运行时间明显高于普通 ALNS，因此更适合作为 final-polishing 或小规模实例的强化搜索模块，而不适合作为每轮迭代中的常规算子。

## 6. 可写入论文的方法描述

可在方法部分加入如下段落：

> To address structurally coupled drone-route improvements that are difficult to reach through incremental ALNS neighborhoods, we introduce a matheuristic large-neighborhood polishing procedure. After the ALNS search terminates, the procedure selects a small neighborhood of drone-served customers and temporarily removes their truck and drone service assignments. It then exhaustively reconstructs the local drone sortie structure by enumerating customer set partitions, within-sortie customer permutations, launch anchors, recovery anchors, and drone assignments. Candidate sorties are filtered by payload capacity and robust energy feasibility before being evaluated by the full cost and feasibility evaluator. The best feasible reconstruction is accepted only if it improves the incumbent solution. This final-polishing step targets cross-truck multi-customer sorties, where the launch and recovery trucks may differ, a structure that is rarely produced by single-move re-anchoring or route-ordering local search.

如果需要中文论文表述，可以写为：

> 为了解决普通 ALNS 邻域难以覆盖的跨车、多客户无人机任务结构，本文在 ALNS 搜索结束后引入一种 matheuristic large neighborhood search 精修机制。该机制首先从当前解中选取一个小规模无人机服务客户邻域，并暂时删除这些客户的卡车或无人机服务关系；随后对该邻域内客户的集合划分、访问顺序、发射锚点、回收锚点以及无人机编号进行穷举重构。每个候选无人机任务首先通过载重容量和鲁棒能耗约束的快速筛选，然后由完整评估器检查同步关系、时间窗、锚点冲突、客户覆盖和鲁棒能耗可行性。若重构后的解可行且总成本低于当前最优解，则接受该局部重构。该方法将启发式全局搜索与小规模精确枚举相结合，专门用于发现传统局部搜索难以产生的跨车无人机 sortie。

## 7. 可写入论文的结果描述

可在结果部分加入如下段落：

> The matheuristic polishing procedure was evaluated on the `R_40_10_1` instance, where the best ALNS solution remained 0.43% above the MILP benchmark. An offline oracle first confirmed that the MILP solution structure was reachable from the ALNS incumbent without modifying the truck routes: exhaustive reconstruction of the customer set `{6,8,10}` enumerated 319,440 local candidates, of which 717 were feasible, and recovered the 97.18 MILP cost by forming a cross-truck two-customer sortie. In the multi-seed comparison, the baseline ALNS and ALNS with drone split/merge/re-anchor local search both achieved a best cost of 97.60, whereas the MLNS-polished variant reached 97.18 in one of five seeds, eliminating the best-case gap to the MILP benchmark.

中文结果表述可写为：

> 在 `R_40_10_1` 实例中，普通 ALNS 的最优结果为 97.60，相比 MILP 最优值 97.18 仍存在 0.43% 的差距。离线 oracle 首先验证了该差距可以通过局部无人机任务重构消除：在固定卡车路线的情况下，对客户集合 `{6,8,10}` 穷举 319,440 个局部组合，其中 717 个组合满足完整可行性约束，最优重构成本为 97.18，与 MILP 最优值一致。在 5 个随机种子的正式对照实验中，ALNS baseline 与加入 Step6 的 ALNS 均只能达到 97.60 的最好结果，而加入 MLNS final polish 后，C 组在 seed 44 中达到 97.18，从而将相对于 MILP 的 best gap 从 0.43% 降至 0.00%。

## 8. 可写入论文的讨论内容

本轮结果可以在讨论部分强调两个方面。第一，普通 ALNS 的 0.43% gap 不是随机误差，而是由解结构差异造成的。跨车、多客户无人机 sortie 同时涉及客户组合、无人机分配和卡车同步，普通局部搜索若只改变其中一个维度，很难跨越该结构性障碍。第二，MLNS final polish 能够关闭 best-case gap，说明针对小规模关键邻域的精确重构是有效的；但由于枚举成本较高，该方法更适合作为最终精修策略，而不是高频迭代算子。

论文中应避免把 MLNS 描述为全面提升平均性能的通用算子。当前 5-seed 结果显示，C 组的最优值达到 MILP，但均值和方差并未优于 Step6，因为 MLNS 只在某些种子生成了合适的邻域并成功重构。因此更准确的表述是：MLNS 提供了一种补充性强化机制，用于识别和修复特定结构性瓶颈；它提升的是 best-known solution 的可达性，而不是在当前参数下稳定改善所有随机种子的平均表现。

## 9. 建议放入论文的贡献点

可以将本轮更新凝练为以下贡献表述：

1. 识别出 ALNS 与 MILP 之间的小 gap 可能源于跨车、多客户无人机 sortie 的结构性差异，而非单纯的路径排序误差。
2. 设计了一种嵌入 ALNS 框架的 matheuristic LNS final-polishing 方法，通过小规模精确枚举联合优化客户集合划分、无人机任务顺序、发射锚点、回收锚点和无人机分配。
3. 通过离线 oracle 证明，在固定卡车路线的情况下，MILP 最优结构可由 ALNS 局部解通过无人机 sortie 重构恢复。
4. 通过 A/B/C/D 对照实验表明，MLNS final polish 在 `R_40_10_1` 上将 best-case gap 从 0.43% 降至 0.00%，达到 MILP 最优成本 97.18。

## 10. 后续写作注意事项

如果将这部分写入正式论文，建议把它定位为“post-optimization intensification”或“matheuristic final polishing”，而不是主 ALNS 算子的常规组成部分。实验表格中应同时报告 best、median、mean 和运行时间，以避免只报告最优值造成方法稳定性被高估。若篇幅有限，可以将 oracle 放入附录，将 A/B/C/D 对照实验放入正文。

## 11. 下一步实验计划

下一阶段实验应先统一口径，再评估算法边界。所有主实验脚本应使用 `config/alns_config.yaml` 中的 `time_window_strategy: class_based`，避免将 demand-based 旧时窗、class-based 新时窗和 PWL/MIP 近似结果混在同一个比较表中。对于专门用于对比旧/新 deadline 的脚本，可以保留 demand-based 分支，但必须在输出表中标明其基准来源。

第一步是重跑 `R_30_10_1` 至 `R_30_10_5` 的 class-based Instance10 对照实验。建议采用统一协议：A 为 ALNS baseline，B 为 ALNS + Step6，C 为 A/B 全部 seed 中 best solution 的 MLNS final polish，而不是只对 B 组 best 做 polish；D 为 MILP 或 exact-feasible MIP 重构结果。每个实例至少使用 10 个随机种子，主表报告 mean、median、min、std、best gap、hit rate 和运行时间。若某个 MIP/PWL 解在 ALNS evaluator 下不可行，应将其标记为 non-comparable reference，而不是作为 exact gap 计算的分母。

第二步是针对 `R_30_10_1` 做注入实验。由于 MIP 解在 ALNS evaluator 下可行且注入 MIP 初始解可立即匹配 90.04，说明该实例的主要问题是搜索可达性，而不是模型不可比。建议设置三组实验：普通两阶段初始解、MIP 解注入初始解、以及从 ALNS best 出发的 larger-neighborhood repair。若注入组稳定保持 90.04，而普通组无法达到，则论文中可将其作为 ALNS 邻域覆盖不足的证据。

第三步是区分 MLNS 的适用边界。当前 MLNS 是 sortie-only reconstruction，适合卡车路线基本正确、主要差异来自无人机分组和跨车锚点的场景。对于 `R_30_10_1` 这种需要同时改变卡车路线、客户服务方式和无人机任务结构的实例，应设计 joint truck-drone LNS 作为后续扩展：删除 3 至 6 个关键客户，允许这些客户在 truck service 和 drone service 之间重新分配，同时枚举或启发式重排局部 truck route 与 drone sortie。该实验可以作为 future work 或补充实验，不必直接并入当前 MLNS 结果。

第四步是重跑 MLNS 触发策略实验。MLNS 不应作为每轮迭代中的必选步骤，而应作为按需 final-polishing 模块。建议记录触发条件，包括搜索停滞、存在多个相邻单客户无人机任务、存在潜在跨车锚点、以及当前 best 与 feasible benchmark 之间仍有 gap。若触发器判定无结构性瓶颈，则跳过 MLNS，以控制运行时间。

建议最终形成两张论文表格。第一张表报告 class-based Instance10 的 A/B/C/D 对照结果，并区分 exact-feasible benchmark 与 non-comparable PWL reference。第二张表报告 MLNS 的触发与成功情况，包括触发次数、改善次数、平均候选数、平均耗时和 best improvement。这样可以清楚说明 MLNS 是一种按需强化机制，而不是对所有实例稳定提升均值的通用算子。
