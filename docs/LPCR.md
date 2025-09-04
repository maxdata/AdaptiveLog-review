# LPCR: Log Pattern Classification and Recognition

## Overview

Log Pattern Classification and Recognition (LPCR) is a sophisticated log analysis task in the AdaptiveLog framework that focuses on evaluating the relevance between log messages and their potential root causes. This task performs multi-class relevance scoring to help operations engineers identify the most likely causes of system issues based on log evidence.

## Task Definition

LPCR performs relevance scoring between a single log message and multiple possible causes. Given:
- **Input Log**: A structured log message from network equipment
- **Possible Causes**: Five potential explanations (A, B, C, D, E)
- **Output**: Relevance scores (0-100) for each cause, where higher scores indicate greater relevance

The task helps operations teams quickly identify the most probable root cause among multiple possibilities, enabling faster incident resolution and more accurate troubleshooting.

## Input/Output Format

### Training Data Structure
```json
[
  [
    [
      "ISIS/6/JN_MTCAST_ADDR_FAIL:Failed to join the multicast group. (InterfaceIndex=[STRING], ReturnValue=[LONG])",
      "Failed to join a multicast group."
    ],
    1
  ],
  [
    [
      "FSP/4/TOPO_CHANGE: Topology changed from [ULONG1] to [ULONG2](0: link, 1: ring).",
      "Received the Delete Auto FRR Tunnel message."
    ],
    0
  ]
]
```

### Example Input/Output
```
Input Log: "ISIS/6/JN_MTCAST_ADDR_FAIL:Failed to join the multicast group. (InterfaceIndex=[STRING], ReturnValue=[LONG])"

Possible Causes:
A: "Failed to join a multicast group."
B: "The V33 power supply of the PSE chip on an RU recovers."
C: "Received the Delete Auto FRR Tunnel message."
D: "NQA automatically uploaded a test result file."
E: "The lockout period of the user name or IP address expired."

Output: A: 100, B: 10, C: 15, D: 5, E: 0
```

## Dataset Information

### Available Datasets
- **Huawei Router**: `lpcr_traindata_hwrouters.json`, `lpcr_devdata_hwrouters.json`, `lpcr_testdata_hwrouters.json`
- **Huawei Switch**: `lpcr_traindata_hwswitch.json`, `lpcr_devdata_hwswitch.json`, `lpcr_testdata_hwswitch.json`

### Dataset Characteristics
- **Log Types**: Network device logs with various protocols and system events
- **Cause Categories**: Hardware issues, protocol events, configuration changes, security incidents
- **Complexity**: Multi-layered scoring requiring deep domain knowledge
- **Scale**: Large datasets with comprehensive cause-effect relationships

## Model Architecture

### Small Language Model (SLM)
- **Base Model**: BERT-based architecture adapted for relevance scoring
- **Task Adaptation**: Multi-class regression for relevance score prediction
- **Training**: Fine-tuned on log-cause relevance pairs
- **Output**: Confidence scores for each cause-log relationship

### Large Language Model (LLM)
- **Model**: GPT-3.5-turbo-16k
- **Capability**: Advanced reasoning for complex cause-effect relationships
- **Domain Knowledge**: Extensive understanding of network protocols and system behavior

## Relevance Scoring Framework

### Score Interpretation
- **90-100**: Highly relevant, direct cause-effect relationship
- **70-89**: Moderately relevant, indirect or contributing factor
- **50-69**: Somewhat relevant, possible but unlikely relationship
- **20-49**: Low relevance, minimal connection
- **0-19**: Irrelevant, no logical connection

### Scoring Criteria

#### High Relevance (90-100)
- **Direct Match**: Log message directly describes the cause
- **Technical Accuracy**: Precise protocol or system component alignment
- **Contextual Consistency**: Parameters and error codes match

#### Moderate Relevance (70-89)
- **Related Components**: Same system module or protocol family
- **Similar Symptoms**: Comparable error patterns or behaviors
- **Indirect Causation**: Contributing factors to the main issue

#### Low Relevance (0-49)
- **Different Domains**: Unrelated system components
- **Contradictory Information**: Conflicting technical details
- **No Logical Connection**: Completely unrelated events

## Error-prone Case Retrieval (ECR)

### Purpose
ECR provides the LLM with examples of challenging relevance scoring cases to improve decision-making for complex log-cause relationships.

### Prompt Template
```
You are a professional operations engineer and your task is to analyze whether given logs and possible causes are relevant.
The input is one log and five possible cause. Please score each cause, with higher scores indicating greater relevance. The maximum score is 100.

For example:
Input Log: "ISIS/6/JN_MTCAST_ADDR_FAIL:Failed to join the multicast group. (InterfaceIndex=[STRING], ReturnValue=[LONG])",
Possible Cause A: "Failed to join a multicast group."
Possible Cause B: 'The V33 power supply of the PSE chip on an RU recovers.'
Possible Cause C: "Received the Delete Auto FRR Tunnel message."
Possible Cause D: 'NQA automatically uploaded a test result file.'
Possible Cause E: "The lockout period of the user name or IP address expired."
Output: A: 100, B: 10, C: 15, D: 5, E: 0

You can refer to the following error-prone cases, learn key features from these cases and attention common pitfalls in prediction.
{error_cases}

The following input is the test data.
Please 1. Describe the reasoning process (e.g. Reason: xxx) firstly by combining these cases with your own domain knowledge of input logs. 2. Finally, follow the label format (e.g. Output: A: xx,B: xx, C: xx, D: xx, E: xx) of examples and give a definite result.
Input Log: {log_message}
Possible Cause A: {cause_a}
Possible Cause B: {cause_b}
Possible Cause C: {cause_c}
Possible Cause D: {cause_d}
Possible Cause E: {cause_e}
```

## Training Process

### Data Preparation
1. **Log-Cause Pairing**: Create training pairs from log messages and potential causes
2. **Score Annotation**: Expert-annotated relevance scores for each pair
3. **Balanced Sampling**: Ensure diverse score distributions across training data

### Model Training
```bash
# Adapted training process for LPCR
python lpcr_small_model_train.py \
    --train_data ./datasets/adaptive_lpcr/lpcr_traindata_hwswitch.json \
    --dev_data ./datasets/adaptive_lpcr/lpcr_devdata_hwswitch.json \
    --pretrain_model bert-base-uncased \
    --task_type relevance_scoring \
    --num_classes 5 \
    --epoch 5
```

### Training Configuration
- **Loss Function**: Mean Squared Error for regression scoring
- **Architecture**: Multi-head regression for simultaneous cause scoring
- **Regularization**: Dropout and batch normalization for stability

## Evaluation Process

### SLM Evaluation
```bash
python lpcr_small_model_pred.py \
    --test_data ./datasets/adaptive_lpcr/lpcr_testdata_hwswitch.json \
    --pretrain_model lpcr_slm.pt \
    --output_format scores
```

### Adaptive Framework Evaluation
1. **Uncertainty Identification**: Cases with conflicting or unclear scores
2. **ECR Application**: Provide relevant error-prone examples to LLM
3. **Score Integration**: Combine SLM and LLM relevance assessments

## Evaluation Metrics

### Scoring Accuracy
- **Mean Absolute Error (MAE)**: Average deviation from ground truth scores
- **Root Mean Square Error (RMSE)**: Penalizes larger scoring errors
- **Pearson Correlation**: Correlation between predicted and actual scores

### Ranking Performance
- **Top-1 Accuracy**: Percentage of cases where highest-scored cause is correct
- **Top-3 Accuracy**: Percentage of cases where correct cause is in top 3
- **Normalized Discounted Cumulative Gain (NDCG)**: Ranking quality metric

### Operational Metrics
- **Troubleshooting Efficiency**: Time reduction in root cause identification
- **False Positive Rate**: Incorrect high-scoring irrelevant causes
- **Coverage**: Percentage of log types with reliable scoring

## Common Challenges

### Multi-cause Scenarios
- **Challenge**: Logs with multiple valid root causes
- **Example**: Network outage with both hardware and configuration issues
- **Solution**: ECR provides examples of similar multi-factor incidents

### Domain Expertise Requirements
- **Challenge**: Scoring requires deep network protocol knowledge
- **Example**: ISIS vs. OSPF routing protocol differences
- **Solution**: LLM provides expert-level domain reasoning

### Temporal Context
- **Challenge**: Cause relevance depends on timing and sequence
- **Example**: Configuration changes preceding system failures
- **Solution**: Historical context consideration in ECR examples

## Adaptive Selection Strategy

### Uncertainty Indicators
1. **Score Variance**: High variability in predicted scores
2. **Confidence Gaps**: Small differences between top-ranked causes
3. **Novel Patterns**: Previously unseen log-cause combinations

### Routing Decision
```python
# Simplified routing logic
score_variance = calculate_variance(predicted_scores)
confidence_gap = max(predicted_scores) - sorted(predicted_scores)[-2]

if score_variance > threshold or confidence_gap < min_gap:
    final_scores = query_llm_with_ecr(log, causes, error_cases)
else:
    final_scores = slm_scores
```

## Performance Characteristics

### SLM Performance
- **Speed**: Fast inference for routine log-cause scoring
- **Consistency**: Reliable performance on common patterns
- **Limitation**: Struggles with complex multi-factor scenarios

### Adaptive Framework Benefits
- **Accuracy Improvement**: 15-25% improvement in scoring accuracy
- **Expert Knowledge**: Access to domain expertise for complex cases
- **Cost Efficiency**: 60-70% reduction in LLM usage

## Usage Examples

### Training Custom Models
```bash
# Train on Huawei router logs
python lpcr_small_model_train.py \
    --train_data ./datasets/adaptive_lpcr/lpcr_traindata_hwrouters.json \
    --dev_data ./datasets/adaptive_lpcr/lpcr_devdata_hwrouters.json \
    --num_epochs 5 \
    --batch_size 32
```

### Production Inference
```bash
# Step 1: SLM scoring
python lpcr_small_model_pred.py \
    --test_data incident_logs.json \
    --model lpcr_model.pt \
    --output_file initial_scores.json

# Step 2: Uncertainty analysis
python lpcr_uncertainty_analysis.py \
    --scores initial_scores.json \
    --threshold 0.3 \
    --output uncertain_cases.json

# Step 3: LLM refinement
python query_ChatGPT.py \
    --data uncertain_cases.json \
    --task lpcr \
    --output refined_scores.json
```

### Interactive Analysis
```python
# Example API usage
from adaptivelog import LPCRAnalyzer

analyzer = LPCRAnalyzer(model_path="lpcr_model.pt")
log = "OSPF/3/NBRLOSS: OSPF neighbor [neighbor-address] Down"
causes = [
    "Network interface failure",
    "OSPF configuration mismatch", 
    "Routing table corruption",
    "Power supply issue",
    "Software bug in OSPF implementation"
]

scores = analyzer.score_relevance(log, causes)
print(f"Relevance scores: {scores}")
```

## Best Practices

### Data Quality
1. **Expert Annotation**: Involve network engineers in score validation
2. **Consistency Checks**: Regular review of scoring guidelines
3. **Diverse Scenarios**: Include edge cases and complex situations

### Model Optimization
1. **Multi-task Learning**: Joint training with related log analysis tasks
2. **Domain Adaptation**: Fine-tune for specific network environments
3. **Ensemble Methods**: Combine multiple models for improved accuracy

### Operational Integration
1. **Incident Response**: Integrate with ticketing and alerting systems
2. **Knowledge Base**: Maintain updated cause-effect relationships
3. **Feedback Loop**: Incorporate operations team corrections

## Integration with Incident Management

### Automated Triage
- **Priority Assignment**: High-relevance causes trigger urgent responses
- **Resource Allocation**: Route incidents to appropriate specialist teams
- **Escalation Rules**: Automatic escalation based on cause severity

### Knowledge Management
- **Cause Database**: Maintain repository of validated log-cause relationships
- **Historical Analysis**: Track cause patterns and resolution effectiveness
- **Best Practices**: Document successful troubleshooting approaches

### Continuous Improvement
- **Model Updates**: Regular retraining with new incident data
- **Threshold Tuning**: Optimize relevance thresholds for specific environments
- **Performance Monitoring**: Track accuracy and operational impact metrics