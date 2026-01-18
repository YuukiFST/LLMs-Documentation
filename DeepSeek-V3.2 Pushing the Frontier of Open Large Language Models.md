# DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models

## Executive Summary

DeepSeek-V3.2 introduces three primary technical contributions to advance open-source large language models: (1) DeepSeek Sparse Attention (DSA), an efficient attention mechanism reducing computational complexity from O(L²) to O(Lk) while maintaining performance in long-context scenarios; (2) a scalable reinforcement learning framework allocating post-training compute exceeding 10% of pre-training cost, achieving performance comparable to GPT-5; and (3) a large-scale agentic task synthesis pipeline generating 1,800+ environments and 85,000+ prompts for scalable post-training. The standard variant matches GPT-5 performance, while DeepSeek-V3.2-Speciale surpasses GPT-5 and achieves parity with Gemini-3.0-Pro, attaining gold-medal performance in IMO 2025, IOI 2025, ICPC World Final 2025, and CMO 2025.

Performance metrics demonstrate substantial improvements: 93.1% on AIME 2025, 2386 Codeforces rating, 73.1% SWE-Verified resolution rate, and 46.4% Terminal Bench 2.0 accuracy. DeepSeek-V3.2-Speciale achieves 96.0% on AIME 2025, 2701 Codeforces rating, and 99.2% on HMMT Feb 2025. The framework addresses three critical deficiencies limiting open-source models: architectural inefficiency in long sequences, insufficient post-training compute, and weak generalization in agentic scenarios.

---

## 1. Technical Architecture

### 1.1 Motivation and Problem Statement

Three critical deficiencies constrain open-source model capabilities in complex tasks. First, reliance on vanilla attention mechanisms severely limits efficiency for long sequences, creating obstacles for scalable deployment and post-training. Second, insufficient computational investment during post-training restricts performance on challenging tasks. Third, open-source models demonstrate marked deficiencies in generalization and instruction-following compared to proprietary systems in agentic deployment scenarios.

### 1.2 Architecture Components

DeepSeek-V3.2 employs identical architecture to DeepSeek-V3.2-Exp. The sole modification from DeepSeek-V3.1-Terminus involves introducing DeepSeek Sparse Attention through continued training.

**DSA Prototype Components:**

The lightning indexer computes index scores between query token h_t ∈ R^d and preceding token h_s ∈ R^d:

```
I_{t,s} = Σ_{j=1}^{H_I} w_{t,j}^I · ReLU(q_{t,j}^I · k_s^I)
```

where H_I denotes indexer head count; q_{t,j}^I ∈ R^{d_I} and w_{t,j}^I ∈ R derive from query token h_t; k_s^I ∈ R^{d_I} derives from preceding token h_s. ReLU activation optimizes throughput. The indexer implements FP8 precision for computational efficiency.

The fine-grained token selection mechanism retrieves key-value entries {c_s} corresponding to top-k index scores. Attention output u_t applies attention between query token h_t and sparsely selected entries:

```
u_t = Attn(h_t, {c_s | I_{t,s} ∈ Top-k(I_{t,:})})
```

**MLA Instantiation:**

DSA instantiates on Multi-head Latent Attention (MLA) using MQA mode for continued training compatibility. Each latent vector (MLA key-value entry) shares across all query heads of the query token at kernel level for computational efficiency.

### 1.3 Training or Pre-Training Protocol

**Continued Pre-Training Stages:**

Starting from DeepSeek-V3.1-Terminus base checkpoint (128K context length), continued pre-training proceeds through two stages with training data distribution aligned to 128K long-context extension data.

*Dense Warm-up Stage:*
- Duration: 1000 steps (2.1B tokens total)
- Batch composition: 16 sequences × 128K tokens per step
- Learning rate: 10^-3
- Optimization target: Lightning indexer only (all other parameters frozen)
- Objective: KL-divergence loss aligning indexer outputs with main attention distribution

```
L_I = Σ_t DKL(p_{t,:} || Softmax(I_{t,:}))
```

where p_{t,:} represents L1-normalized aggregated main attention scores.

*Sparse Training Stage:*
- Duration: 15000 steps (943.7B tokens total)
- Batch composition: 480 sequences × 128K tokens per step
- Learning rate: 7.3 × 10^-6
- Selected tokens: 2048 key-value tokens per query token
- Optimization: Separate optimization paths for indexer (L_I only) and main model (language modeling loss only)
- Modified objective considering only selected token set S_t:

```
L_I = Σ_t DKL(p_{t,S_t} || Softmax(I_{t,S_t}))
```

Indexer input detached from computational graph for independent optimization.

### 1.4 Performance Impact

**Computational Complexity:**

DSA reduces core attention complexity from O(L²) to O(Lk) where k (≪ L) represents selected token count. Lightning indexer maintains O(L²) complexity but requires substantially less computation than MLA in DeepSeek-V3.1-Terminus.

**Inference Costs:**

Benchmarking on deployed H800 GPU service (2 USD/GPU-hour rental) demonstrates significant end-to-end speedup in long-context scenarios. For short-sequence prefilling, masked MHA mode simulates DSA for higher efficiency under short-context conditions.

**Performance Parity:**

Evaluation across standard benchmarks, human preference (ChatbotArena), and long-context tasks (AA-LCR, Fiction.liveBench) demonstrates DeepSeek-V3.2-Exp achieves parity with DeepSeek-V3.1-Terminus despite sparse attention mechanism. AA-LCR evaluation shows 4-point improvement over DeepSeek-V3.1-Terminus in reasoning mode.

---

## 2. Post-Training or Optimization Methods

Post-training employs sparse attention identical to sparse continued pre-training stage. The pipeline maintains consistency with DeepSeek-V3.2-Exp, comprising specialist distillation and mixed RL training.

**Specialist Distillation:**

Domain-specific specialists fine-tune from identical DeepSeek-V3.2 base checkpoint across six domains: mathematics, programming, general logical reasoning, general agentic tasks, agentic coding, and agentic search. All domains support thinking and non-thinking modes. Each specialist undergoes large-scale RL training. Separate models generate training data for long chain-of-thought reasoning (thinking mode) versus direct response generation (non-thinking mode). Models trained on distilled data achieve performance marginally below domain specialists, with gaps eliminated through subsequent RL training.

**Mixed RL Training:**

Group Relative Policy Optimization (GRPO) serves as the RL algorithm, merging reasoning, agent, and human alignment training into unified RL stage. This approach balances performance across domains while circumventing catastrophic forgetting in multi-stage training.

Reward structure:
- Reasoning/agent tasks: Rule-based outcome reward, length penalty, language consistency reward
- General tasks: Generative reward model with prompt-specific evaluation rubrics

**Model Variants:**

*DeepSeek-V3.2:* Integrates reasoning, agent, and human alignment data from specialists through thousands of continued RL training steps.

*DeepSeek-V3.2-Speciale:* Trained exclusively on reasoning data with reduced length penalty during RL. Incorporates dataset and reward methodology from DeepSeekMath-V2 for enhanced mathematical proof capabilities.

**GRPO Scaling Innovations:**

*Unbiased KL Estimate:*
Corrects K3 estimator using importance-sampling ratio between current policy π_θ and old policy π_old:

```
DKL(π_θ(o_{i,t}) || π_ref(o_{i,t})) = 
  (π_θ(o_{i,t}|q,o_{i,<t}) / π_old(o_{i,t}|q,o_{i,<t})) × 
  (π_ref(o_{i,t}|q,o_{i,<t}) / π_θ(o_{i,t}|q,o_{i,<t}) - 
   log(π_ref(o_{i,t}|q,o_{i,<t}) / π_θ(o_{i,t}|q,o_{i,<t})) - 1)
```

Eliminates systematic estimation errors and unbounded gradient weights when π_θ ≪ π_ref, stabilizing training dynamics.

*Off-Policy Sequence Masking:*
Binary mask M applied to GRPO loss for sequences with negative advantages and significant policy divergence:

```
M_{i,t} = 0 if Â_{i,t} < 0 and (1/|o_i|)Σ_t log(π_old(o_{i,t}|q,o_{i,<t})/π_θ(o_{i,t}|q,o_{i,<t})) > δ
M_{i,t} = 1 otherwise
```

where δ controls policy divergence threshold. Improves tolerance for off-policy updates from batched rollout data and training-inference framework inconsistencies.

*Keep Routing:*
Preserves expert routing paths from inference framework during training for MoE models, ensuring identical expert parameters optimize across frameworks. Critical for RL training stability of MoE architectures.

*Keep Sampling Mask:*
Preserves top-p/top-k truncation masks during sampling from π_old and applies to π_θ during training, ensuring identical action subspaces. Maintains language consistency during RL training while avoiding extremely low-probability token optimization.

**Thinking in Tool-Use:**

*Context Management:*
Historical reasoning content discards only upon new user message introduction. Tool-related messages (tool outputs) preserve reasoning content throughout interaction. Tool call history and results remain preserved when reasoning traces removed.

*Cold-Start Mechanism:*
Leverages model instruction-following capability to incorporate tool execution within reasoning process. Distinct system prompts associated with different task types enable seamless integration without specialized training data initially.

*Large-Scale Agentic Task Synthesis:*

The synthesis pipeline generates diverse, executable environments:

**Search Agent (50,275 tasks):**
Multi-agent pipeline samples informative long-tail entities from web corpora. Question-construction agent explores entities using search tools with configurable depth/breadth parameters. Multiple answer-generation agents with heterogeneous configurations produce diverse candidates. Verification agent with search capabilities validates answers through multiple passes, retaining only verifiably correct ground-truth with incorrect candidates. Generative reward model scores responses across quality dimensions.

**Code Agent (24,667 tasks):**
Mines millions of issue-PR pairs from GitHub, filtered using heuristic rules and LLM-based judgments. Automated environment-setup agent builds executable environments handling package installation, dependency resolution, and test execution. JUnit format ensures consistent parsing across languages. Environment validated only when gold patch produces non-zero false-to-positive (F2P) test cases and zero pass-to-fail (P2F) test cases. Spans Python, Java, JavaScript, TypeScript, C, C++, Go, PHP.

**Code Interpreter Agent (5,908 tasks):**
Utilizes Jupyter Notebook for complex reasoning tasks across mathematics, logic, and data science requiring code execution capabilities.

**General Agent (1,827 environments, 4,417 tasks):**
Automatic environment-synthesis agent creates task-oriented environments that are hard to solve but easy to verify. Workflow: (1) Generate/retrieve relevant data from Internet and store in sandbox database; (2) Synthesize task-specific tools as functions; (3) Propose simple task with solution and verification functions; (4) Iteratively increase difficulty, updating solution and verification; (5) Augment toolset if insufficient. Retains only instances with non-zero pass@100 after RL on DeepSeek-V3.2.

---

## 3. Agentic or System-Level Design (if applicable)

The agentic framework addresses generalization and instruction-following deficiencies through systematic integration of reasoning capabilities into tool-use scenarios.

**Architectural Principles:**

Historical reasoning persistence throughout multi-turn tool interactions reduces token inefficiency from redundant re-reasoning. Context management strategy:
- Discard reasoning content only upon new user message arrival
- Preserve reasoning content when appending tool-related messages
- Maintain tool call and result history when removing reasoning traces

**Synthesis Pipeline Structure:**

The pipeline generates training data spanning real-world tools (web search APIs, coding tools, Jupyter Notebooks) and synthesized environments. Prompts derive from Internet extraction or synthetic generation rather than actual user interactions.

Task distribution: 24,667 code agent tasks (real environment, extracted prompts), 50,275 search agent tasks (real environment, synthesized prompts), 4,417 general agent tasks (synthesized environment and prompts), 5,908 code interpreter tasks (real environment, extracted prompts).

**Generalization Validation:**

Ablation studies demonstrate RL on synthetic general agent data produces substantial improvements on Tau2Bench, MCP-Mark, and MCP-Universe benchmarks. Restricting RL to code and search scenarios yields no improvement on these benchmarks, validating synthetic data generalization capacity.

Evaluation on 50 randomly sampled synthesized tasks shows: DeepSeek-V3.2-Exp 12% accuracy, Claude-4.5-Sonnet 34%, Gemini-3.0-Pro 51%, GPT-5-Thinking 62%, demonstrating sufficient task difficulty for frontier models.

**Context Management at Test-Time:**

Three strategies extend token budgets when usage exceeds 80% of 128K context window:
- Summary: Summarizes overflowed trajectory and re-initiates rollout
- Discard-75%: Removes first 75% of tool call history
- Discard-all: Resets context by discarding all previous tool call history

BrowseComp evaluation demonstrates context management enables test-time compute scaling. Discard-all achieves 67.6% accuracy (versus 51.4% baseline) with 364 average steps, comparable to parallel scaling (Parallel-fewest-step) while using significantly fewer steps.

---

## 4. Benchmark Performance and Ablations

### Standard Reasoning and Coding Performance

| Benchmark | Claude-4.5-Sonnet | GPT-5-High | Gemini-3.0-Pro | Kimi-K2-Thinking | DeepSeek-V3.2-Thinking |
|-----------|-------------------|------------|----------------|------------------|------------------------|
| MMLU-Pro (EM) | 88.2 | 87.5 | 90.1 | 84.6 | 85.0 |
| GPQA Diamond (Pass@1) | 83.4 | 85.7 | 91.9 | 84.5 | 82.4 |
| HLE (Pass@1) | 13.7 | 26.3 | 37.7 | 23.9 | 25.1 |
| LiveCodeBench (Pass@1-COT) | 64.0 | 84.5 | 90.7 | 82.6 | 83.3 |
| Codeforces (Rating) | 1480 | 2537 | 2708 | - | 2386 |
| AIME 2025 (Pass@1) | 87.0 | 94.6 | 95.0 | 94.5 | 93.1 |
| HMMT Feb 2025 (Pass@1) | 79.2 | 88.3 | 97.5 | 89.4 | 92.5 |
| HMMT Nov 2025 (Pass@1) | 81.7 | 89.2 | 93.3 | 89.2 | 90.2 |
| IMOAnswerBench (Pass@1) | - | 76.0 | 83.3 | 78.6 | 78.3 |

### Agentic Capabilities Performance

| Benchmark | Claude-4.5-Sonnet | GPT-5-High | Gemini-3.0-Pro | Kimi-K2-Thinking | DeepSeek-V3.2-Thinking |
|-----------|-------------------|------------|----------------|------------------|------------------------|
| Terminal Bench 2.0 (Acc) | 42.8 | 35.2 | 54.2 | 35.7 | 46.4 |
| SWE Verified (Resolved) | 77.2 | 74.9 | 76.2 | 71.3 | 73.1 |
| SWE Multilingual (Resolved) | 68.0 | 55.3 | - | 61.1 | 70.2 |
| BrowseComp (Pass@1) | 24.1 | 54.9 | - | 60.2* | 51.4/67.6* |
| BrowseCompZh (Pass@1) | 42.4 | 63.0 | - | 62.3 | 65.0 |
| HLE with Search (Pass@1) | 32.0 | 35.2 | 45.8 | 44.9 | 40.8 |
| τ²-Bench (Pass@1) | 84.7 | 80.2 | 85.4 | 74.3 | 80.3 |
| MCP-Universe (Success Rate) | 46.5 | 47.9 | 50.7 | 35.6 | 45.9 |
| MCP-Mark (Pass@1) | 33.3 | 50.9 | 43.1 | 20.4 | 38.0 |
| Tool-Decathlon (Pass@1) | 38.6 | 29.0 | 36.4 | 17.6 | 35.2 |

*Note: BrowseComp results with context management technique noted with asterisk.*

### DeepSeek-V3.2-Speciale Performance

Performance metrics with output token counts (in thousands):

| Benchmark | GPT-5-High | Gemini-3.0-Pro | Kimi-K2 | DeepSeek-V3.2 | DeepSeek-V3.2-Speciale |
|-----------|------------|----------------|---------|---------------|------------------------|
| AIME 2025 | 94.6 (13k) | 95.0 (15k) | 94.5 (24k) | 93.1 (16k) | 96.0 (23k) |
| HMMT Feb 2025 | 88.3 (16k) | 97.5 (16k) | 89.4 (31k) | 92.5 (19k) | 99.2 (27k) |
| HMMT Nov 2025 | 89.2 (20k) | 93.3 (15k) | 89.2 (29k) | 90.2 (18k) | 94.4 (25k) |
| IMOAnswerBench | 76.0 (31k) | 83.3 (18k) | 78.6 (37k) | 78.3 (27k) | 84.5 (45k) |
| LiveCodeBench | 84.5 (13k) | 90.7 (13k) | 82.6 (29k) | 83.3 (16k) | 88.7 (27k) |
| Codeforces | 2537 (29k) | 2708 (22k) | - | 2386 (42k) | 2701 (77k) |
| GPQA Diamond | 85.7 (8k) | 91.9 (8k) | 84.5 (12k) | 82.4 (7k) | 85.7 (16k) |
| HLE | 26.3 (15k) | 37.7 (15k) | 23.9 (24k) | 25.1 (21k) | 30.6 (35k) |

### Elite Competition Performance

DeepSeek-V3.2-Speciale achieves gold-medal thresholds across multiple competitions:

**IMO 2025:** 35/42 points (P1: 7, P2: 7, P3: 7, P4: 7, P5: 7, P6: 0) - Gold Medal

**CMO 2025:** 102/126 points (P1: 18, P2: 18, P3: 9, P4: 21, P5: 18, P6: 18) - Gold Medal

**IOI 2025:** 492/600 points (P1: 100, P2: 82, P3: 72, P4: 100, P5: 55, P6: 83) - Gold Medal (10th place)

**ICPC WF 2025:** 10/12 problems solved (A: 3 submissions, C: 1, D: 1, E: 2, F: 2, H: 1, I: 1, J: 1, K: 1, L: 1) - Gold Medal (2nd place)

### Ablation Studies

**Synthetic Task Difficulty Validation:**

Evaluation on 50 randomly sampled general synthesized tasks:

| Model | Pass@1 | Pass@2 | Pass@4 |
|-------|--------|--------|--------|
| DeepSeek-V3.2-Exp | 12% | 18% | 26% |
| Claude-4.5-Sonnet | 34% | 47% | 62% |
| Gemini-3.0-Pro | 51% | 65% | 74% |
| GPT-5-Thinking | 62% | 75% | 82% |

**Generalization from Synthetic Tasks:**

RL training exclusively on synthetic general agent data (non-thinking mode) demonstrates substantial improvements over DeepSeek-V3.2-SFT on Tau2Bench, MCP-Mark, and MCP-Universe benchmarks. DeepSeek-V3.2-Exp (trained only on code/search environments) shows no improvement on these benchmarks, validating synthetic data generalization capacity.

**Context Management Efficiency:**

BrowseComp evaluation across test-time compute expansion strategies (real steps measured):

- Baseline: 51.4% accuracy
- Summary: 60.2% accuracy (364 avg steps, lower efficiency)
- Discard-75%: Intermediate performance
- Discard-all: 67.6% accuracy (comparable to parallel scaling, significantly fewer steps)
- Parallel-fewest-step: Comparable accuracy with higher step count

**Non-Thinking vs Thinking Mode:**

| Benchmark | Non-Thinking | Thinking |
|-----------|--------------|----------|
| Terminal Bench 2.0 (Acc) | 37.1 | 46.4 |
| SWE Verified (Resolved) | 72.1 | 73.1 |
| SWE Multilingual (Resolved) | 68.9 | 70.2 |
| τ²-bench (Pass@1) | 77.2 | 80.3 |
| MCP-Universe (Success Rate) | 38.6 | 45.9 |
| MCP-Mark (Pass@1) | 26.5 | 38.0 |
| Tool-Decathlon (Pass@1) | 25.6 | 35.2 |

---

## 5. Key Technical Takeaways

- DeepSeek Sparse Attention reduces core attention complexity from O(L²) to O(Lk) through lightning indexer (O(L²) but minimal computation) and fine-grained token selection (2048 tokens per query)
- Unbiased KL estimation eliminates systematic gradient errors when π_θ ≪ π_ref, critical for stable RL convergence at scale
- Off-policy sequence masking selectively filters negative-advantage sequences exceeding policy divergence threshold, improving tolerance for batched rollout data
- Keep Routing and Keep Sampling Mask operations ensure training-inference consistency for MoE architectures and truncated sampling strategies
- Post-training computational budget exceeding 10% of pre-training cost enables performance parity with GPT-5 on reasoning benchmarks
- Specialist distillation across six domains (mathematics, programming, general reasoning, general agentic, agentic coding, agentic search) with thinking/non-thinking modes enables effective knowledge transfer
- Large-scale agentic task synthesis (1,827 environments, 85,000+ prompts) demonstrates strong generalization to out-of-domain benchmarks (Tau2Bench, MCP-Mark, MCP-Universe)
- Context management strategies (Summary, Discard-75%, Discard-all) extend effective token budget beyond 128K limit, with Discard-all achieving optimal efficiency-scalability balance
- DeepSeek-V3.2-Speciale achieves gold-medal performance across IMO 2025, IOI 2025, ICPC WF 2025, CMO 2025 without targeted competition-specific training
- Token efficiency remains substantially inferior to Gemini-3.0-Pro (23k vs 15k tokens for AIME 2025; 77k vs 22k for Codeforces), identifying critical optimization target

---

## 6. Conclusion

DeepSeek-V3.2 demonstrates that open-source large language models can achieve performance parity with frontier proprietary systems through architectural efficiency improvements, scaled post-training compute allocation, and systematic agentic capability development. The introduction of DeepSeek Sparse Attention addresses long-context computational bottlenecks while maintaining performance across standard and long-context evaluations. The scalable GRPO framework with unbiased KL estimation, off-policy sequence masking, and consistency-preserving operations enables stable convergence with post-training budgets exceeding 10% of pre-training cost.

The large-scale agentic task synthesis pipeline validates the hypothesis that diverse synthetic environments can drive generalizable reasoning capabilities in tool-use scenarios. Performance improvements on out-of-domain benchmarks (Tau2Bench, MCP-Mark, MCP-Universe) demonstrate effective transfer learning from 1,827 synthesized environments to real-world agentic tasks. Context management strategies successfully extend test-time compute capacity beyond architectural limits, with Discard-all achieving comparable performance to parallel scaling while maintaining superior efficiency.

DeepSeek-V3.2-Speciale establishes a milestone for open-source models, achieving gold-medal performance across elite mathematical and programming competitions (IMO 2025, IOI 2025, ICPC WF 2025, CMO 2025) and performance parity with Gemini-3.0-Pro on reasoning benchmarks. This demonstrates that relaxed length constraints during RL training unlock substantial capability improvements, though at the cost of reduced token efficiency.

Three limitations constrain current performance relative to frontier closed-source models: (1) narrower world knowledge breadth due to fewer total training FLOPs, addressable through scaled pre-training compute; (2) inferior token efficiency requiring longer generation trajectories to match output quality, necessitating optimization of reasoning chain intelligence density; (3) gap in solving complex tasks compared to frontier models, motivating continued refinement of foundation models and post-training methodology.

The framework's technical contributions—architectural efficiency through sparse attention, stable RL scaling through unbiased estimation and consistency operations, and generalizable agentic capabilities through systematic synthesis—provide a foundation for continued advancement of open-source large language models toward frontier capabilities.

---

## References

- Paper: arXiv:2512.02556v1 [cs.CL] 2 Dec 2025
- Code: https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/tree/main/inference
- Training: Continued pre-training from DeepSeek-V3.1-Terminus base checkpoint (128K context); dense warm-up 1000 steps (2.1B tokens); sparse training 15000 steps (943.7B tokens)
- Evaluation: Temperature 1.0; context window 128K tokens; math tasks use template "{question}\nPlease reason step by step, and put your final answer within \boxed{}."; MCP benchmarks use function calling format with tool outputs in 'tool' role
- Post-training: Computational budget exceeding 10% of pre-training cost; specialist distillation across six domains; GRPO algorithm with unbiased KL estimate, off-policy sequence masking, Keep Routing, Keep Sampling Mask
- Agentic synthesis: 24,667 code agent tasks (real environment, extracted prompts); 50,275 search agent tasks (real environment, synthesized prompts); 4,417 general agent tasks (synthesized environment and prompts); 5,908 code interpreter tasks (real environment, extracted prompts)
- Competition evaluation: IMO/CMO use DeepSeekMath-V2 generate-verify-refine loop; IOI samples 500 candidates, filters invalid submissions, selects 50 longest thinking traces; ICPC samples 32 candidates with identical filtering; maximum generation length 128K tokens, no tools or internet access
