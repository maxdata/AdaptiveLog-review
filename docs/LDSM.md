# LDSM: Log Diagnosis and Severity Mapping

## Overview

Log Diagnosis and Severity Mapping (LDSM) is one of the core log analysis tasks in the AdaptiveLog framework. This task focuses on determining whether a given log message is properly mapped to its corresponding natural language description and assessing the correctness of this mapping relationship.

## Task Definition

LDSM performs binary classification to analyze the relationship between:
- **Input**: A structured log message (with parameters and format)
- **Description**: A natural language explanation of what the log means
- **Output**: Binary label (0 or 1) indicating whether the description correctly matches the log

The task evaluates whether the natural language description accurately represents the meaning, context, and severity implications of the structured log message.

## Input/Output Format

### Training Data Structure
```json
[
  [
    [
      "PIM/1/INTELECTDR:OID [oid] Interface is elected to be DR. (IfIndex=[integer], IfIPverion=[integer], IfAddrType=[integer], Addr=[binary], IfName=[string], InstanceID=[integer], InstanceName=[string])",
      "This trap is generated when a switch was elected as the DR used to forward data in the shared network segment."
    ],
    1
  ],
  [
    [
      "WLAN/4/AP_DETECT_SOFTGRE_DOWN:OID [oid] AP detect softgre tunnel down notify.(APMAC=[OPAQUE], DstIP=[IPADDR], APName=[STRING], APID=[INTEGER])",
      "It is the trap indicating the successful member link negotiation and is a matching trap of hwLacpNegotiateFailed."
    ],
    0
  ]
]
```

### Format Explanation
- **First element**: List containing [log_message, description]
- **Second element**: Binary label (1 = correct mapping, 0 = incorrect mapping)

## Dataset Information

### Available Datasets
- **Huawei Switch**: `ldsm_hwswitch_train.json`, `ldsm_hwswitch_dev.json`, `ldsm_hwswitch_test.json`
- **Huawei Router**: `ldsm_hwrouters_train.json`, `ldsm_hwrouters_dev.json`, `ldsm_hwrouters_test.json`
- **Cisco Switch**: `ldsm_csswitch_train.json`, `ldsm_csswitch_dev.json`, `ldsm_csswitch_test.json`
- **Cisco Router**: `ldsm_csrouters_train.json`, `ldsm_csrouters_dev.json`, `ldsm_csrouters_test.json`

### Dataset Characteristics
- **Device Types**: Network equipment logs from Huawei and Cisco
- **Log Formats**: Standard network device log formats with severity levels, modules, and parameter placeholders
- **Challenge**: Distinguishing between semantically similar but incorrect descriptions

## Model Architecture

### Small Language Model (SLM)
- **Base Model**: BERT-base-uncased
- **Architecture**: Fine-tuned transformer with softmax classification head
- **Training**: Supervised fine-tuning on labeled log-description pairs
- **Output**: Binary classification with confidence scores

### Large Language Model (LLM)
- **Model**: GPT-3.5-turbo-16k
- **Role**: Handles uncertain cases identified by SLM
- **Enhancement**: Uses Error-prone Case Retrieval (ECR) for better reasoning

## Adaptive Selection Strategy

### Uncertainty Estimation
1. **Monte Carlo Dropout**: Multiple forward passes with dropout enabled
2. **Variance Calculation**: Measure prediction variance across multiple runs
3. **Threshold-based Routing**: Cases with high uncertainty are sent to LLM

### Selection Process
```python
# Simplified workflow
if slm_uncertainty > threshold:
    result = query_llm_with_ecr(log_pair, error_cases)
else:
    result = slm_prediction
```

## Error-prone Case Retrieval (ECR)

### Purpose
ECR provides the LLM with similar error-prone cases as context to improve reasoning for difficult log-description pairs.

### Prompt Template
```
You are a professional operations engineer and your task is to analyze whether given logs and natural language descriptions are relevant.
You can refer to the following error-prone cases, learn key features from these cases and attention common pitfalls in prediction.
Please 1. Describe the reasoning process firstly by referring to the reasoning process of relevant error-prone cases. 
2. Follow the label format of examples and give a definite result.
{error_cases}

The following input is the test data.
Please 1. Describe the reasoning process (e.g. Reason: xxx) firstly by referring to the reasons of relevant error-prone cases. 2. Finally, follow the label format (e.g. Label: xxx) and give a definite result.
Input: [{log_message}, {description}]
```

## Training Process

### SLM Fine-tuning
```bash
python ldsm_small_model_train.py \
    --train_data ./datasets/adaptive_ldsm/ldsm_hwswitch_train.json \
    --dev_data ./datasets/adaptive_ldsm/ldsm_hwswitch_dev.json \
    --pretrain_model bert-base-uncased \
    --epoch 3 \
    --batch_size 16 \
    --outfolder ldsm_slm.pt
```

### Training Configuration
- **Loss Function**: SoftmaxLoss for binary classification
- **Optimizer**: AdamW with warmup (10% of training steps)
- **Evaluation**: Regular validation on development set
- **Early Stopping**: Based on validation accuracy

## Evaluation Process

### SLM Evaluation
```bash
python ldsm_small_model_pred.py \
    --test_data ./datasets/adaptive_ldsm/ldsm_hwswitch_test.json \
    --pretrain_model ldsm_slm.pt
```

### Uncertainty Estimation
```bash
python ldsm_uncertain_pred.py \
    --test_data ./datasets/adaptive_ldsm/ldsm_hwswitch_test.json \
    --dev_data ./datasets/adaptive_ldsm/ldsm_hwswitch_dev.json \
    --model_path ldsm_slm.pt \
    --out_simple_path output/ldsm_hwswitch_simplesmaples.json \
    --out_hard_path output/ldsm_hwswitch_hardsmaples.json
```

### LLM Querying
```bash
python query_ChatGPT.py \
    --data ./datasets/adaptive_ldsm/ldsm_hwswitch_test.json \
    --model gpt-3.5-turbo-16k-0613
```

## Evaluation Metrics

### Primary Metrics
- **Accuracy**: Overall classification accuracy
- **F1 Score**: Weighted F1 score for balanced evaluation
- **Precision/Recall**: Per-class performance analysis

### Adaptive Framework Metrics
- **Cost Efficiency**: Reduction in LLM API calls
- **Performance Gain**: Improvement over SLM-only baseline
- **Routing Accuracy**: Correctness of uncertainty-based routing decisions

## Common Challenges

### Semantic Similarity
- **Challenge**: Distinguishing between semantically similar but technically incorrect descriptions
- **Example**: A PIM (Protocol Independent Multicast) log incorrectly described as LACP (Link Aggregation Control Protocol) functionality

### Parameter Interpretation
- **Challenge**: Understanding parameter significance in context
- **Solution**: ECR provides similar cases for parameter pattern recognition

### Domain Knowledge
- **Challenge**: Requires understanding of network protocols and device operations
- **Solution**: LLM provides domain expertise for uncertain cases

## Performance Characteristics

### SLM Performance
- **Speed**: Fast inference (~50-200ms per sample)
- **Accuracy**: Good performance on clear cases
- **Limitation**: Struggles with edge cases and domain-specific nuances

### Adaptive Framework Benefits
- **Cost Reduction**: 60-80% reduction in LLM API calls
- **Accuracy Improvement**: 5-15% improvement over SLM-only
- **Scalability**: Maintains performance with increased data volume

## Usage Examples

### Basic Training
```bash
# Train SLM on Huawei switch logs
python ldsm_small_model_train.py \
    --train_data ./datasets/adaptive_ldsm/ldsm_hwswitch_train.json \
    --dev_data ./datasets/adaptive_ldsm/ldsm_hwswitch_dev.json
```

### Adaptive Inference
```bash
# Step 1: Evaluate SLM and identify uncertain cases
python ldsm_uncertain_pred.py \
    --test_data ./datasets/adaptive_ldsm/ldsm_hwswitch_test.json \
    --model_path ldsm_slm.pt

# Step 2: Process uncertain cases with LLM
python query_ChatGPT.py \
    --data output/ldsm_hwswitch_hardsmaples.json
```

## Best Practices

### Data Preparation
1. **Balance Dataset**: Ensure balanced positive/negative examples
2. **Quality Control**: Verify log-description pair accuracy
3. **Domain Coverage**: Include diverse network device types and scenarios

### Model Training
1. **Hyperparameter Tuning**: Optimize learning rate and batch size
2. **Regularization**: Use dropout to prevent overfitting
3. **Validation**: Regular evaluation on held-out development set

### Deployment
1. **Threshold Tuning**: Optimize uncertainty threshold for cost-accuracy trade-off
2. **Monitoring**: Track routing decisions and performance metrics
3. **Feedback Loop**: Continuously improve with production data