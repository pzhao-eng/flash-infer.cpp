## 背景
项目为flash-attention 2 forward处理，和原来的forward相比主要差异为：
1. 剥离了backward需要保存的中间变量
2. 没有考虑kv cache影响
3. headdim加载和shared memory尺寸对齐