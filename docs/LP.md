# LP: Log Parsing (Level Parsing)

## Overview

Log Parsing (LP), also referred to as Level Parsing, is a critical log analysis task in the AdaptiveLog framework that focuses on determining the severity level of log messages. This task automatically classifies logs into different severity categories to help operations engineers prioritize their response to system events.

## Task Definition

LP performs severity level classification for network device logs by analyzing the log content and determining the appropriate severity level. The task specifically focuses on distinguishing between different severity levels, particularly:

- **Error Level**: Indicates serious problems that require immediate attention
- **Info Level**: Records informative messages about normal system operations

The goal is to automatically assign the correct severity level to logs where the severity has been masked or is unknown.

## Input/Output Format

### Training Data Structure
```json
[
  [
    "ISIS/[MASK]/LV1_T1TMR_STAT_SETRR:In ISIS [process-id], level-1 T1 timer started on circuit [circuit-name] during ISIS GR, setting RR(Counter=[USHORT]).",
    0
  ],
  [
    "RIPNG/[MASK]/RL_NOTIFY_DEL_OVER: RIPng finished route deletion from RM DataBase. (InstanceId=[ULONG])",
    0
  ],
  [
    "Entitytrap/[MASK]/POWERINVALID(t):OID [oid] Power supply is unavailable for some reason.(Index=[INTEGER], EntityPhysicalIndex=[INTEGER], PhysicalName=\"[OCTET]\", EntityTrapFaultID=[INTEGER])",
    1
  ]
]
```

### Format Explanation
- **First element**: Log message with severity level masked as `[MASK]`
- **Second element**: Severity label (0 = Info, 1 = Error)

### Log Format Structure
Network device logs follow the standard format:
```
MODULE/SEVERITY/MESSAGE_ID:Log description with parameters
```
Where:
- **MODULE**: The system module that generated the log
- **SEVERITY**: Numerical severity level (masked in training)
- **MESSAGE_ID**: Unique identifier for the log type
- **Parameters**: Placeholder values in brackets (e.g., [INTEGER], [STRING])

## Dataset Information

### Available Datasets
- **Huawei Switch**: `lp_hwswitch_train.json`, `lp_hwswitch_dev.json`, `lp_hwswitch_test.json`
- **Huawei Router**: `lp_hwrouters_train.json`, `lp_hwrouters_dev.json`, `lp_hwrouters_test.json`
- **Cisco Switch**: `lp_csswitch_train.json`, `lp_csswitch_dev.json`, `lp_csswitch_test.json`
- **Cisco Router**: `lp_csrouters_train.json`, `lp_csrouters_dev.json`, `lp_csrouters_test.json`

### Dataset Characteristics
- **Network Protocols**: ISIS, OSPF, BGP, PIM, VRRP, LACP, and more
- **Device Types**: Enterprise switches and routers
- **Severity Distribution**: Balanced mix of Error and Info level logs
- **Parameter Variety**: Rich parameter types (IP addresses, interface names, protocol IDs)

## Model Architecture

### Small Language Model (SLM)
- **Base Model**: BERT-base-uncased
- **Task Adaptation**: Binary classification head for severity prediction
- **Training**: Fine-tuned on labeled log severity data
- **Features**: Contextual understanding of log semantics and network domain knowledge

### Large Language Model (LLM)
- **Model**: GPT-3.5-turbo-16k
- **Role**: Handles complex severity classification cases
- **Advantage**: Leverages extensive domain knowledge for nuanced severity decisions

## Severity Classification Logic

### Error Level Indicators
- **System Failures**: Power issues, hardware faults, critical errors
- **Security Events**: Authentication failures, attacks, unauthorized access
- **Protocol Failures**: Neighbor loss, configuration conflicts, timeout errors
- **Resource Exhaustion**: Memory shortage, bandwidth limits exceeded

### Info Level Indicators
- **Normal Operations**: Successful configurations, routine notifications
- **Status Updates**: Interface state changes, timer events, routine maintenance
- **Statistics**: Performance metrics, counters, periodic reports
- **Informational**: Debug messages, trace information

## Error-prone Case Retrieval (ECR)

### Purpose
ECR helps the LLM make better severity decisions by providing examples of similar logs where severity classification was challenging or error-prone.

### Prompt Template
```
You are a professional operations engineer in network device log analysis. Given a h3c log, your task is to analyse the severity of a log based on the similar examples.
Note: Log is output in the format of module name/severity level/summary/log description.
The input is a log whose severity level is masked with [MASK]. The severity level including Error and Info.
"Error" level is used to indicate serious problems, while the "Info" level is for recording informative messages about the normal operation of the system.

You can refer to the following error-prone cases, learn key features from these cases and attention common pitfalls in prediction.
{error_cases}

The following input is the test data. You can identify relevant error-prone cases where the reasoning process is similar to the test data as reference.
Please 1. Describe the reasoning process (e.g. Reason: xxx) firstly by combining these cases with your own domain knowledge of input logs. 2. Finally, follow the label format (e.g. Label: xxx) of examples and give a definite result.
Input: {log_message}
```

## Training Process

### Data Preprocessing
1. **Severity Masking**: Replace actual severity levels with `[MASK]` token
2. **Parameter Normalization**: Standardize parameter placeholders
3. **Module Standardization**: Consistent module name formatting

### Model Training
```bash
# Training command (adapted from LDSM training)
python lp_small_model_train.py \
    --train_data ./datasets/adaptive_lp/lp_hwswitch_train.json \
    --dev_data ./datasets/adaptive_lp/lp_hwswitch_dev.json \
    --pretrain_model bert-base-uncased \
    --epoch 3 \
    --batch_size 16
```

### Training Configuration
- **Loss Function**: Binary cross-entropy for severity classification
- **Optimization**: AdamW optimizer with learning rate scheduling
- **Regularization**: Dropout and weight decay to prevent overfitting

## Evaluation Process

### SLM Evaluation
```bash
python lp_small_model_pred.py \
    --test_data ./datasets/adaptive_lp/lp_hwswitch_test.json \
    --pretrain_model lp_slm.pt
```

### Adaptive Framework Evaluation
1. **Uncertainty Estimation**: Identify logs with unclear severity
2. **LLM Querying**: Process uncertain cases with error-prone examples
3. **Result Integration**: Combine SLM and LLM predictions

## Evaluation Metrics

### Classification Metrics
- **Accuracy**: Overall severity classification accuracy
- **Precision**: Accuracy of Error and Info predictions
- **Recall**: Coverage of Error and Info cases
- **F1 Score**: Balanced measure of precision and recall

### Domain-Specific Metrics
- **Critical Miss Rate**: Percentage of Error logs classified as Info (high impact)
- **False Alarm Rate**: Percentage of Info logs classified as Error (operational overhead)
- **Protocol-wise Accuracy**: Performance breakdown by network protocol

## Common Challenges

### Contextual Ambiguity
- **Challenge**: Similar log messages with different severity implications
- **Example**: Interface down events (could be Error or Info depending on context)
- **Solution**: ECR provides contextual examples for better decision-making

### Protocol-Specific Nuances
- **Challenge**: Different protocols have different severity conventions
- **Example**: BGP neighbor changes vs. ISIS neighbor changes
- **Solution**: Protocol-aware error-prone case selection

### Parameter Sensitivity
- **Challenge**: Parameter values affecting severity interpretation
- **Example**: Threshold-based alarms with different criticality levels
- **Solution**: Parameter-aware reasoning with LLM

## Adaptive Selection Strategy

### Uncertainty Indicators
1. **Prediction Confidence**: Low softmax probability scores
2. **Semantic Ambiguity**: Logs with mixed severity signals
3. **Novel Patterns**: Previously unseen log message patterns

### Routing Logic
```python
# Simplified routing decision
if prediction_confidence < threshold or is_novel_pattern(log):
    severity = query_llm_with_ecr(log, relevant_error_cases)
else:
    severity = slm_prediction
```

## Performance Characteristics

### SLM Performance
- **Speed**: Sub-second inference for most logs
- **Accuracy**: High performance on common log patterns
- **Limitation**: Struggles with domain-specific edge cases

### Adaptive Framework Benefits
- **Accuracy Improvement**: 10-20% improvement on challenging cases
- **Cost Efficiency**: 70% reduction in LLM usage compared to LLM-only approach
- **Reliability**: Consistent performance across different network device types

## Usage Examples

### Basic Severity Classification
```bash
# Train model on Cisco router logs
python lp_small_model_train.py \
    --train_data ./datasets/adaptive_lp/lp_csrouters_train.json \
    --dev_data ./datasets/adaptive_lp/lp_csrouters_dev.json
```

### Production Deployment
```bash
# Step 1: Run SLM inference
python lp_small_model_pred.py \
    --test_data production_logs.json \
    --pretrain_model lp_model.pt

# Step 2: Process uncertain cases
python lp_uncertain_analysis.py \
    --input production_logs.json \
    --model lp_model.pt \
    --threshold 0.7

# Step 3: LLM processing for uncertain cases
python query_ChatGPT.py \
    --data uncertain_logs.json \
    --task level_parsing
```

## Best Practices

### Data Quality
1. **Balanced Datasets**: Equal representation of Error and Info logs
2. **Representative Sampling**: Cover all major network protocols and scenarios
3. **Expert Validation**: Human expert review of edge cases

### Model Optimization
1. **Threshold Tuning**: Optimize uncertainty threshold for specific deployments
2. **Protocol-Specific Models**: Consider separate models for different network domains
3. **Continuous Learning**: Regular retraining with new log patterns

### Operational Deployment
1. **Monitoring**: Track classification accuracy and routing decisions
2. **Feedback Loop**: Incorporate operations team feedback for model improvement
3. **Escalation**: Clear procedures for handling model uncertainty

## Integration with Network Operations

### Alerting Systems
- **Critical Path**: Error logs trigger immediate alerts
- **Information Logs**: Stored for trend analysis and troubleshooting
- **Filtering**: Reduce alert fatigue through accurate severity classification

### Troubleshooting Workflows
- **Priority Queuing**: Error logs processed first by operations teams
- **Historical Analysis**: Info logs provide context for error investigation
- **Pattern Recognition**: Identify recurring issues across severity levels

### Compliance and Reporting
- **Audit Trails**: Maintain records of severity classification decisions
- **SLA Monitoring**: Track error frequency and resolution times
- **Capacity Planning**: Use info logs for infrastructure planning