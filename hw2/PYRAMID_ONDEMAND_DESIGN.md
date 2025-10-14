━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 按需构建 Pyramid - 设计方案
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 方案设计

### 核心思想
每个 rank 只构建到它需要处理的**最大 octave**，而不是构建所有 8 个 octaves。

### 具体实现

修改 `generate_gaussian_pyramid` 函数，添加一个参数来指定构建多少个 octaves：

```cpp
ScaleSpacePyramid generate_gaussian_pyramid(const Image& img, 
                                            float sigma_min,
                                            int num_octaves,  // 要构建的数量
                                            int scales_per_octave);
```

在 hw2.cpp 中：
```cpp
// 每个 rank 只构建到它需要的最大 octave
int octaves_to_build = my_start_octave + my_num_octaves;

ScaleSpacePyramid gaussian_pyramid = generate_gaussian_pyramid(
    img, SIGMA_MIN, octaves_to_build, N_SPO);  // 只构建需要的
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 示例分析（n=4）

当前分配：
- Rank 0: 处理 octave 0
- Rank 1: 处理 octaves 1-3  
- Rank 2: 处理 octaves 4-5
- Rank 3: 处理 octaves 6-7

优化后构建：
- Rank 0: 构建 octaves 0     (1个) → 节省 7 个
- Rank 1: 构建 octaves 0-3   (4个) → 节省 4 个
- Rank 2: 构建 octaves 0-5   (6个) → 节省 2 个
- Rank 3: 构建 octaves 0-7   (8个) → 节省 0 个

总节省：(7 + 4 + 2 + 0) / (8 * 4) = 13/32 = 40.6%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 符合规定性检查

✅ 不修改 SIFT 参数
   - N_OCT, N_SPO 等参数值不变
   - 只是每个 rank 构建不同数量的 octaves

✅ 不跳过 scales
   - 每个构建的 octave 仍然有所有 scales
   - 没有改变算法逻辑

✅ 不改变计算复杂度
   - 所有需要的 octaves 仍然被某个 rank 构建
   - 总的计算量不变（分布式）

✅ 满足依赖关系
   - 因为从 octave 0 开始顺序构建
   - Octave i 依赖 octave i-1，这个关系得到满足

✅ 结果完全一致
   - 每个 rank 处理的 octaves 的结果与之前完全相同
   - 只是不浪费时间构建不用的 octaves

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 预期收益

### Test 06 (n=3)
当前分配：
- Rank 0: octave 0 (需要构建 1 个)
- Rank 1: octaves 1-4 (需要构建 4 个)  
- Rank 2: octaves 5-7 (需要构建 8 个)

当前：每个 rank 构建 8 个 → 总 24 个
优化：构建 1 + 4 + 8 = 13 个
节省：11/24 = 45.8%

### Test 08 (n=4)
节省：13/32 = 40.6%

### 时间估算
如果 pyramid 构建占 60% 总时间：
- n=3: 节省 45.8% * 60% = 27.5% 总时间
- n=4: 节省 40.6% * 60% = 24.4% 总时间

Test 06: 39.8s → 预计 28.8s (-28%)
Test 08: 35.2s → 预计 26.6s (-24%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 实施步骤

1. ✅ 确认方案符合作业规定
2. 修改 generate_gaussian_pyramid (不改变签名，只是使用 num_octaves 参数)
3. 修改 hw2.cpp 传递正确的 octaves 数量
4. 测试 n=1 (应该不变)
5. 测试 n=2, 3, 4 (应该显著提升)
6. 如果有问题，立即回退

风险：低（因为不改变算法，只是提前停止循环）
收益：高（20-30% 总时间）

