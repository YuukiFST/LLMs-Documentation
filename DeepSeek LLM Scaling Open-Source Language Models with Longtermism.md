# DeepSeek LLM: Scaling Open-Source Language Models with Longtermism

## Executive Summary

DeepSeek LLM represents a significant contribution to open-source language model research, introducing a family of models (7B and 67B parameters) trained on 2 trillion tokens of bilingual (Chinese/English) data. The paper's core innovation lies in its rigorous re-examination of scaling laws, challenging previous work by Kaplan et al. (2020) and Hoffmann et al. (2022) while introducing methodological improvements that yield more accurate performance predictions.

**Key Achievement**: DeepSeek LLM 67B surpasses LLaMA-2 70B across multiple benchmarks, particularly excelling in code (HumanEval: 42.7% vs 28.7%), mathematics (MATH: 18.7% vs 13.5%), and reasoning tasks (BBH: 68.7% vs 62.9%). The chat variant outperforms GPT-3.5 in both Chinese and English open-ended evaluations.

**Research Impact**: The paper provides actionable insights for scaling future models through empirically-validated hyperparameter selection formulae and optimal compute allocation strategies, representing a "longtermist" approach to open-source LLM development.

---

## Technical Analysis

### 1. Scaling Laws: A Paradigm Refinement

#### 1.1 Hyperparameter Scaling Laws

**Problem Statement**: Previous scaling law research inadequately addressed hyperparameter optimization across different compute budgets, creating uncertainty about whether models achieved optimal performance.

**Methodology**: The team conducted extensive grid searches across compute budgets ranging from 1e17 to 2e19 FLOPs, evaluating batch size and learning rate combinations. They identified a "wide band" of near-optimal parameters (within 0.25% of minimum generalization error).

**Key Findings**:

The optimal batch size and learning rate follow power law relationships with compute budget C:

```
η_opt = 0.3118 · C^(-0.1250)
B_opt = 0.2920 · C^(0.3271)
```

**Interpretation**: As compute budget increases:
- Batch size gradually increases (exponent 0.3271)
- Learning rate gradually decreases (exponent -0.1250)

This aligns with intuitive empirical practices but provides the first quantitative formulation for the LLaMA-style architecture family.

**Limitation Acknowledged**: The formulae don't account for factors beyond compute budget C. The authors note that models with identical compute but different model/data allocations exhibit slightly different optimal parameter spaces, suggesting more complex underlying dynamics.

#### 1.2 Model Scale Representation Innovation

**Critical Insight**: Previous work used either non-embedding parameters N₁ or complete parameters N₂ to represent model scale, both yielding significant approximation errors.

**Novel Approach**: Introduction of **non-embedding FLOPs/token (M)** as the model scale representation:

```
M = 72·n_layer·d_model² + 12·n_layer·d_model·l_seq
```

Compared to:
```
6N₁ = 72·n_layer·d_model²  (ignores attention overhead)
6N₂ = 72·n_layer·d_model² + 6·n_vocab·d_model  (includes vocabulary)
```

**Quantitative Impact**: Table 3 demonstrates that for small models (8 layers, 512 width), 6N₁ underestimates compute by 57% while 6N₂ overestimates by 32%. For large models (80 layers, 8192 width), discrepancies reduce to 8% and 6% respectively.

**Result**: Using M enables more accurate compute budget formulation (C = M·D instead of C ≈ 6N·D) and better performance predictions, as evidenced by Figure 5 showing accurate forecasting of 67B model performance from small-scale experiments.

#### 1.3 Optimal Model/Data Scaling Strategy

Using IsoFLOP profiling across 8 compute budgets (1e17 to 3e20 FLOPs), the team fitted:

```
M_opt = 0.1715 · C^0.5243
D_opt = 5.8316 · C^0.4757
```

**Comparison with Prior Work**:

| Study | Model Scaling Exponent (a) | Data Scaling Exponent (b) | Dataset |
|-------|---------------------------|--------------------------|---------|
| Kaplan et al. (OpenAI) | 0.73 | 0.27 | OpenWebText2 |
| Hoffmann et al. (Chinchilla) | 0.49 | 0.51 | MassiveText |
| DeepSeek (Early Data) | 0.450 | 0.550 | Internal v1 |
| **DeepSeek (Current Data)** | **0.524** | **0.476** | Internal v2 |
| DeepSeek (OpenWebText2) | 0.578 | 0.422 | OpenWebText2 |

**Data Quality Hypothesis**: The exponent a increases with data quality, suggesting that higher-quality data supports larger models for the same data scale. This potentially explains divergent conclusions in prior scaling law research and provides an indirect data quality assessment method.

---

### 2. Model Architecture and Training

#### 2.1 Architecture Design Decisions

**Base Architecture**: Follows LLaMA's proven design with modifications:
- Pre-Norm structure with RMSNorm
- SwiGLU activation (FFN intermediate dimension: 8/3·d_model)
- Rotary Position Embedding (RoPE)
- Grouped-Query Attention (GQA) for 67B model (8 KV heads for efficiency)

**Depth vs Width Trade-off**: 
- DeepSeek 7B: 30 layers (vs typical ~32)
- DeepSeek 67B: **95 layers** (vs LLaMA-2 70B's 80 layers)

**Rationale**: Prioritizing depth over FFN width while maintaining parameter count yields better performance and facilitates pipeline partitioning for distributed training.

#### 2.2 Multi-Step Learning Rate Scheduler

**Innovation**: Replaces cosine decay with a three-stage multi-step scheduler:
1. 80% of tokens: Maximum learning rate
2. 10% of tokens: 31.6% of maximum
3. 10% of tokens: 10% of maximum

**Advantages**:
- Performance parity with cosine scheduler (Figure 1a)
- **Enables continual training** by reusing early checkpoints
- Flexibility for dynamic scaling decisions

**Design Choice**: The 80/10/10 split balances reuse ratios against potential performance gains from alternative distributions (Figure 1b).

#### 2.3 Data Pipeline

**Scale**: 2 trillion tokens (bilingual Chinese/English corpus)

**Processing Stages**:

1. **Aggressive Deduplication**: Cross-dump deduplication of Common Crawl
   - Single dump: 22.2% deduplication rate
   - 91 dumps: **89.8% deduplication rate** (4× more effective)

2. **Filtering**: Multi-level quality assessment
   - Linguistic evaluation (individual document level)
   - Semantic evaluation (corpus-level coherence)

3. **Remixing**: Rebalancing underrepresented domains

**Tokenizer**: BBPE (Byte-level BPE) with 100,000 conventional tokens + 15 special tokens, padded to 102,400 for computational efficiency. Number splitting follows LLaMA to improve numerical reasoning.

---

### 3. Alignment Pipeline

#### 3.1 Supervised Fine-Tuning (SFT)

**Data Composition**:
- 1.5M instruction instances (1.2M helpful + 300K safety)
- Helpful data distribution: 31.2% language tasks, 46.6% math, 22.2% code
- Coverage: English and Chinese

**Training Strategy**:
- 7B model: 4 epochs with two-stage approach
  - Stage 1: All data (48.2% HumanEval, 63.9% GSM8K, but 2.0% repetition)
  - Stage 2: Conversational data only (maintains scores, reduces repetition to 1.4%)
- 67B model: 2 epochs only (early overfitting observed, <1% repetition achieved in single stage)

**Observation on Repetition**: Math SFT data's reasoning patterns cause weaker models to generate repetitive responses. Staged fine-tuning or DPO mitigates this without sacrificing benchmark performance.

#### 3.2 Direct Preference Optimization (DPO)

**Implementation**:
- Learning rate: 5e-6
- Batch size: 512
- 1 epoch with warmup and cosine decay

**Effect**: DPO enhances open-ended generation quality (MT-Bench: 8.35 → 8.76) with minimal impact on standard benchmarks, demonstrating effectiveness for conversational alignment without reward model training overhead.

---

### 4. Evaluation Results

#### 4.1 Base Model Performance

**English Benchmarks** (DeepSeek 67B vs LLaMA-2 70B):
- Comparable on language understanding: HellaSwag (84.0 vs 84.0), PIQA (83.6 vs 82.0)
- Superior on reasoning: BBH (68.7 vs 62.9), MMLU (71.3 vs 69.0)
- **Significantly better on code/math**:
  - HumanEval: **42.7% vs 28.7%** (+49% relative improvement)
  - MBPP: **57.4% vs 45.6%** (+26%)
  - MATH: **18.7% vs 13.5%** (+39%)
  - GSM8K: **63.4% vs 58.4%** (+9%)

**Chinese Benchmarks**:
- CHID (idiom understanding): **92.1% vs 55.5%** (requires Chinese training data)
- C-Eval: **66.1% vs 51.4%**
- CMMLU: **70.8% vs 53.1%**

**Key Insight**: The performance gap between DeepSeek 67B and LLaMA-2 70B exceeds that between respective 7B models, indicating language interference affects smaller models more severely. However, mathematical reasoning transfers across languages (LLaMA-2's CMath performance).

#### 4.2 Chat Model Performance

**AlignBench (Chinese)**: DeepSeek 67B Chat scores 6.43, surpassing ChatGPT (6.08) and ranking third behind GPT-4 variants. DPO version achieves 6.69 with improvements across all categories.

**MT-Bench (English)**: DPO version scores 8.76, second only to GPT-4 (9.26) and ahead of GPT-3.5 (8.39).

**Held-Out Evaluations** (demonstrating true generalization):
- LeetCode Weekly Contests: 17.5% (vs 12.7% for Qwen 72B)
- Hungarian Math Exam: 58/100 (vs 52 for Qwen 72B)
- IFEval (instruction following): 55.5% (vs 50.8% for Qwen 72B)

#### 4.3 Safety Evaluation

**Internal Safety Test**: 2,400 manually curated questions across 5 major categories
- Overall safety rate: >95% across all categories
- Highest: "Other Safety Issues" (767/800 = 95.9%)
- Lowest: "Trade Secrets & IP" (281/300 = 93.7%)

**Do-Not-Answer Benchmark**: Score of **97.8**, exceeding both ChatGPT (97.7) and GPT-4 (96.5)

---

### 5. Key Discussion Findings

#### 5.1 Multi-Choice Question Data

**Experiment**: Adding 20M Chinese MC questions during fine-tuning

**Results**:
- MMLU: 49.4% → 60.9% (+11.5 points)
- C-Eval: 47.0% → 71.3% (+24.3 points)
- CMMLU: 49.7% → 73.8% (+24.1 points)
- TriviaQA: 57.9% → 57.9% (no change)
- ChineseQA: 75.0% → 74.4% (slight decrease)

**Conclusion**: MC data improves MC benchmark performance but doesn't enhance true intelligence or generative QA capability. **Decision: Exclude MC data to avoid benchmark overfitting.**

#### 5.2 System Prompt Effects

**Observation**: Adding system prompts (modified from LLaMA-2's template) affects models differently:
- 7B model: MT-Bench 7.15 → 7.11 (-0.04)
- 67B model: MT-Bench 8.35 → 8.58 (+0.23)

**Interpretation**: Larger models better comprehend and follow system-level instructions, while smaller models may suffer from train-test distribution mismatch.

#### 5.3 Instruction Data in Pre-Training

**Experiment**: Adding 5M instruction instances (primarily MC) in final 10% of pre-training

**Finding**: Base model benchmark improvements equal those from adding the same data during SFT. **Decision: Omit instruction data from pre-training** to preserve pre-training data budget for more general knowledge.

---

### 6. Infrastructure and Engineering

**Training Framework**: HAI-LLM with:
- 4D parallelism: Data, Tensor, Sequence, Pipeline (1F1B schedule)
- ZeRO-1 optimizer state partitioning
- FlashAttention for memory efficiency
- Operator fusion (LayerNorm, GEMM, Adam, in-place cross-entropy)
- Mixed precision: bf16 computation, fp32 gradient accumulation

**Fault Tolerance**: Asynchronous checkpointing every 5 minutes ensures ≤5 min training loss on hardware failures.

**Evaluation Infrastructure**: vLLM for generative tasks, continuous batching for non-generative tasks.

---

## Key Takeaways

1. **Scaling Laws Are Data-Dependent**: The optimal model/data allocation strategy varies significantly with data quality (exponent a ranges from 0.45 to 0.578). Higher quality data justifies larger models.

2. **Hyperparameter Formulae Enable Predictable Scaling**: Power law relationships for batch size and learning rate (B_opt ∝ C^0.33, η_opt ∝ C^-0.125) provide empirical guidance for future scaling.

3. **Non-Embedding FLOPs/Token (M) Is Superior**: Using M instead of parameter count N reduces prediction error by up to 50% for small models, enabling accurate 1000× extrapolation.

4. **Depth Over Width for Large Models**: Prioritizing layers (95 for 67B) over FFN width yields better performance while maintaining parameter count.

5. **Two-Stage SFT Mitigates Repetition**: For smaller models struggling with math/code SFT data, separating capability acquisition (stage 1) from conversational refinement (stage 2) reduces repetition without sacrificing performance.

6. **Benchmark Overfitting Is Real**: Multi-choice data dramatically improves MC benchmarks without enhancing true intelligence. Held-out evaluations reveal the performance gap between model sizes more accurately than standard benchmarks.

7. **Open-Source Parity with Proprietary Models**: Rigorous methodology enables 67B model to match/exceed GPT-3.5 in open-ended evaluations while surpassing LLaMA-2 70B on technical benchmarks.

---

## Conclusion

DeepSeek LLM demonstrates that systematic application of refined scaling laws, coupled with high-quality data curation and thoughtful architectural choices, enables open-source models to achieve competitive performance with proprietary systems. The project's "longtermist" philosophy—investing in fundamental research to guide future scaling—contrasts with practices focused solely on fixed-size model optimization.

**Future Directions** (from paper):
- Upcoming releases: Code intelligence and Mixture-of-Experts (MoE) technical reports
- Next version improvements: Enhanced reasoning, Chinese knowledge, math, and code capabilities through larger, refined datasets
- Alignment research: Reinforcement learning to boost complex reasoning

**Limitations**:
- Knowledge cutoff (January 2025 for this analysis context)
- Potential hallucination in long-form generation
- Non-exhaustive Chinese data in v1 may impact certain cultural topics
- Limited multilingual capability beyond Chinese/English

The paper's methodological rigor, transparency about design decisions, and comprehensive ablation studies set a high standard for open-source LLM research, providing the community with actionable insights for future model development.

Reference: https://arxiv.org/pdf/2401.02954
