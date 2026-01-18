# Engram: Scaling Language Models Through Conditional Memory

## Executive Summary

DeepSeek-AI introduces **Engram**, a novel architectural primitive that augments large language models with conditional memory via constant-time lookup operations. This work challenges the prevailing paradigm where Mixture-of-Experts (MoE) serves as the sole scaling mechanism, revealing that **static knowledge retrieval and dynamic computation represent complementary—not competing—forms of sparsity**. The research demonstrates that a 27B-parameter model with optimally allocated memory capacity outperforms iso-parameter, iso-FLOPs MoE baselines across diverse domains, with particularly striking gains in reasoning tasks (+5.0 on BBH) that exceed improvements on knowledge-intensive benchmarks (+3.4 on MMLU).

**Key Innovation**: Rather than forcing Transformers to simulate memory through expensive multi-layer computation, Engram provides a native lookup primitive for static patterns (named entities, formulaic phrases, N-grams), effectively "deepening" the network by freeing early layers from reconstruction tasks.

---

## Technical Architecture

### Core Design: Hash-Based N-gram Memory

Engram implements a modernized N-gram embedding system with four critical components:

**1. Tokenizer Compression (23% vocabulary reduction)**
- Maps semantically equivalent tokens to canonical IDs (e.g., "Apple" → "apple")
- Uses NFKC normalization and lowercasing
- Creates suffix N-grams: g_{t,n} = (x'_{t-n+1}, ..., x'_t)

**2. Multi-Head Hashing for Collision Mitigation**
- K distinct hash heads per N-gram order
- Deterministic function φ_{n,k} maps contexts to embedding indices
- Final memory vector: e_t = concat(e_{t,2,1}, ..., e_{t,N,K})
- Scales to 27B parameters with negligible O(1) retrieval overhead

**3. Context-Aware Gating (Attention-Inspired)**
- Hidden state h_t serves as dynamic Query
- Retrieved memory e_t provides Key/Value projections
- Gate α_t ∈ (0,1) computed via: α_t = σ(RMSNorm(h_t)^T · RMSNorm(k_t) / √d)
- Suppresses noisy retrievals when context contradicts static pattern

**4. Multi-Branch Integration**
- Shared embedding table and Value projection (W_V)
- Branch-specific Key projections {W^(m)_K} for M=4 branches
- Enables fused FP8 matrix multiplication on modern GPUs

### System-Level Efficiency: Decoupling Storage from Compute

**Critical Insight**: Unlike MoE's dynamic routing, Engram uses deterministic addressing based solely on input tokens.

**Training**: All-to-All sharding across GPUs, linear capacity scaling
**Inference**: Prefetch-and-overlap strategy
- Embeddings reside in host DRAM (100B parameters tested)
- Asynchronous PCIe transfer during earlier layer computation
- Multi-level cache hierarchy exploits Zipfian N-gram distribution
- **Result**: <3% throughput penalty despite 100B offloaded parameters

---

## The Sparsity Allocation Problem

### U-Shaped Scaling Law Discovery

**Research Question**: Given fixed total parameters P_tot and activated parameters P_act, how should sparse capacity be split between MoE experts and Engram memory?

**Allocation Ratio Definition**:
- ρ = fraction of inactive parameters assigned to MoE
- P^(sparse)_MoE = ρ · P_sparse
- P_Engram = (1 - ρ) · P_sparse

**Experimental Protocol** (constant sparsity ratio P_tot/P_act ≈ 10):
- 2×10^20 FLOPs regime: 5.7B total, 568M activated
- 6×10^20 FLOPs regime: 9.9B total, 993M activated

**Key Findings**:
1. **Pure MoE (ρ=100%) is suboptimal** across both regimes
2. **Optimal allocation: ρ ≈ 75-80%** (consistent across scales)
3. At 10B scale: Δ loss = 0.0139 improvement at ρ=80% vs ρ=100%
4. **Even ρ=40%** (reallocating 60% to memory) matches pure MoE

**Interpretation**: The U-shape reveals structural complementarity:
- **MoE-dominated (ρ→100%)**: Wastes depth reconstructing static patterns
- **Engram-dominated (ρ→0%)**: Lacks computation for context-dependent reasoning

### Infinite Memory Regime

Fixed backbone (3B total, 568M activated), swept memory slots from 2.58×10^5 to 1.0×10^7:
- **Strict power law**: log-linear improvement in validation loss
- Engram outperforms OverEncoding baseline at all scales
- Memory provides predictable scaling without additional FLOPs

---

## Large-Scale Pre-Training Results

### Model Configurations

| Model | Total Params | Activated | Experts | Engram Memory |
|-------|--------------|-----------|---------|---------------|
| Dense-4B | 4.1B | 3.8B | - | - |
| MoE-27B | 26.7B | 3.8B | 72 routed (top-6) | - |
| **Engram-27B** | **26.7B** | **3.8B** | **55 routed (top-6)** | **5.7B** |
| Engram-40B | 39.5B | 3.8B | 55 routed (top-6) | 18.5B |

All trained on 262B tokens with identical curriculum.

### Performance Highlights (Engram-27B vs MoE-27B)

**Knowledge-Intensive Tasks** (expected gains):
- MMLU: 60.4 vs 57.4 (+3.0)
- CMMLU: 61.9 vs 57.9 (+4.0)
- C-Eval: 62.7 vs 58.0 (+4.7)

**General Reasoning** (surprising larger gains):
- **BBH: 55.9 vs 50.9 (+5.0)**
- **ARC-Challenge: 73.8 vs 70.1 (+3.7)**
- **DROP: 59.0 vs 55.7 (+3.3)**

**Code & Math** (unexpected improvements):
- **HumanEval: 40.8 vs 37.8 (+3.0)**
- **GSM8K: 60.6 vs 58.4 (+2.2)**
- **MATH: 30.7 vs 28.3 (+2.4)**

**Critical Observation**: Gains on reasoning/code exceed knowledge tasks, contradicting the naive hypothesis that memory only aids factual retrieval.

---

## Mechanistic Analysis: Why Engram Works

### 1. Effective Depth Increase

**LogitLens Analysis** (KL divergence to final output):
- Engram models show **consistently lower KL in early layers**
- Steeper convergence indicates faster "prediction-ready" states
- Early knowledge lookup bypasses multi-layer feature composition

**Example (Table 3)**: Resolving "Diana, Princess of Wales"
- Standard LLM: Layers 1-2 → "Wales" (country)
- Layer 3 → "Wales" (region)
- Layers 4-5 → "Princess of Wales" (unspecific)
- Layer 6 → Full entity resolution

Engram retrieves this in **O(1)** instead of 6 layers of computation.

**CKA Similarity Analysis**:
- Soft alignment index: a_j = Σ(S_{i,j} · i) / Σ(S_{i,j}) for top-k MoE layers
- **Distinct upward shift from diagonal**: Engram layer 5 ≈ MoE layer 12
- Confirms: Early Engram layers achieve deeper representations

### 2. Attention Capacity Liberation

By delegating local dependencies to lookups, attention focuses on global context:

**Long-Context Extension Results** (32k tokens, same training budget):

| Metric | MoE-27B (50k) | Engram-27B (46k, iso-loss) |
|--------|---------------|---------------------------|
| Multi-Query NIAH | 84.2 | **97.0** (+12.8) |
| Variable Tracking | 77.0 | **87.2** (+10.2) |
| LongPPL (Book) | 4.38 | **4.19** |

**Extreme Efficiency**: Engram-27B at 41k steps (82% FLOPs) matches baseline LongPPL while surpassing RULER accuracy.

### 3. Functional Specialization

**Ablation Study** (suppressing Engram at inference):

| Task Category | Retained Performance |
|---------------|----------------------|
| **Factual Knowledge** | **29-44%** (TriviaQA: 29%) |
| Reading Comprehension | **81-93%** (C3: 93%) |
| Reasoning | 62-76% |
| Code | 58-76% |

**Interpretation**: 
- Engram serves as **primary parametric knowledge store**
- Backbone handles context-grounded reasoning independently
- Clean separation of static memory and dynamic computation

---

## Architecture Ablations

### Layer Placement Trade-off

Single 1.6B Engram module sweep (Layers 1-12):
- **Optimal: Layer 2** (Val Loss = 1.770)
- Layer 1 slightly worse (insufficient context for gating)
- Performance degrades monotonically after Layer 2

**Trade-off Explanation**:
- **Early injection**: Offloads local patterns before backbone wastes depth
- **Late injection**: Stronger contextual queries for gating precision

**Best Practice**: Divide memory across Layers 2 and 6 (Val Loss = 1.768)

### Component Importance Ranking

Ablating from reference configuration (1.6B memory):

1. **Multi-branch fusion**: Largest regression when removed
2. **Context-aware gating**: Critical for noise suppression
3. **Tokenizer compression**: 23% vocabulary reduction essential
4. **Depthwise convolution**: Marginal impact
5. **4-grams**: Slightly suboptimal under fixed budget (dilutes frequent 2/3-grams)

---

## Qualitative Validation: Gating Visualization

Analysis of gating scalars α_t reveals interpretable selectivity:

**English Patterns** (strong activation in red):
- Named entities: "Alexander **the Great**", "the Milky **Way**"
- Formulaic phrases: "By the **way**", "Princess of **Wales**"

**Chinese Patterns**:
- Idiomatic expressions: "四大**发明**" (Four Great Inventions)
- Historical entities: "张仲**景**" (Zhang Zhongjing)

**Validation**: Gating mechanism successfully identifies stereotyped dependencies without explicit supervision.

---

## Key Takeaways

### 1. Conditional Memory as Modeling Primitive
- **Not just an optimization trick**: Memory lookup is architecturally orthogonal to MoE computation
- Optimal allocation law generalizes across scales (ρ ≈ 75-80%)
- Enables predictable scaling via memory expansion without FLOPs increase

### 2. Reasoning Gains Exceed Knowledge Gains
- **Counterintuitive finding**: +5.0 BBH vs +3.4 MMLU
- Mechanism: Freeing early layers from reconstruction increases effective depth for complex reasoning
- Attention capacity liberated for global context modeling

### 3. Infrastructure-Aware Design Matters
- Deterministic addressing enables prefetching (unlike MoE's dynamic routing)
- 100B parameters offloaded with <3% overhead
- Multi-level caching exploits Zipfian distribution

### 4. System-Algorithm Co-Design
- Layer placement balances modeling (early injection) and latency (compute overlap)
- Multi-branch integration enables fused FP8 operations
- Training (All-to-All) and inference (prefetch) require different strategies

---

## Implications for Future Work

### Immediate Extensions
1. **Hierarchical N-grams**: Current work uses N≤3; explore 4-6 grams at larger memory scales
2. **Dynamic memory editing**: Leverage ROME/MEMIT insights for knowledge updates without retraining
3. **Cross-lingual memory sharing**: Tokenizer compression hints at universal pattern spaces

### Architectural Questions
1. **Optimal sparsity ratio**: Does P_tot/P_act ≈ 10 generalize beyond tested scales?
2. **Memory-attention fusion**: Can gating mechanism directly modulate attention weights?
3. **Retrieval granularity**: Beyond N-grams—phrase embeddings, syntactic patterns?

### System Challenges
1. **Cache hierarchy optimization**: Quantify benefit of SSD-backed long-tail storage
2. **Distributed training**: All-to-All scaling limits at massive GPU counts
3. **Dynamic batching**: Prefetch complexity with variable sequence lengths

---

## Conclusion

Engram establishes **conditional memory as an indispensable complement to conditional computation** in sparse language models. The work's significance extends beyond architectural novelty:

1. **Empirical**: U-shaped allocation law reveals structural complementarity, validated across 2-6×10^20 FLOPs regimes
2. **Mechanistic**: Demonstrates memory enables depth increase, not just knowledge storage
3. **Practical**: Infrastructure-aware design achieves 100B-parameter scaling with negligible overhead

The surprising finding that reasoning gains exceed knowledge gains suggests language modeling may be fundamentally bottlenecked by **inefficient static pattern reconstruction** rather than compute capacity alone. By providing a native lookup primitive, Engram allows Transformers to allocate their sequential depth where it matters most: compositional reasoning over global context.

**Final Assessment**: This work redefines the pareto frontier for sparse models, establishing memory capacity as a first-class scaling dimension alongside computation and parameters.

---

## References & Resources

- **Code**: https://github.com/deepseek-ai/Engram
- **Model Family**: DeepSeek-V3 (Liu et al., 2024a)
- **Related Work**: SCONE (Yu et al., 2025), OverEncoding (Huang et al., 2025a), BLT (Pagnoni et al., 2025)
- **Theoretical Foundations**: N-gram models (Shannon, 1948; Brants et al., 2007), PKM (Lample et al., 2019)
