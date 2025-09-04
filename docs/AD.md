# AD: Anomaly Detection

## Overview

Anomaly Detection (AD) is a critical log analysis task in the AdaptiveLog framework that focuses on identifying abnormal patterns and behaviors in system log sequences. This task performs binary classification to determine whether a sequence of log messages contains anomalous events that may indicate system failures, security breaches, or performance issues.

## Task Definition

AD performs binary anomaly detection on log sequences by analyzing:
- **Input**: A sequence of chronologically ordered log messages
- **Context**: Temporal relationships and patterns within the sequence
- **Output**: Binary classification (True = anomaly detected, False = normal behavior)

The task identifies deviations from normal system behavior that may require immediate attention from operations teams.

## Input/Output Format

### Training Data Structure
```json
[
  [0, 0, "Output: False"],
  [1, 0, "Output: False"],
  [2, 0, "Output: False"],
  [3, 0, "Output: False"],
  [4, 0, "Output: False"],
  [5, 1, "Output: True"],
  [6, 0, "Output: False"]
]
```

### Format Explanation
- **First element**: Sequence ID or log entry index
- **Second element**: Ground truth label (0 = normal, 1 = anomaly)
- **Third element**: Expected output format for LLM

### Log Sequence Structure
Log sequences typically contain:
- **Temporal Ordering**: Chronologically arranged log messages
- **Contextual Relationships**: Dependencies between consecutive log events
- **Pattern Recognition**: Normal vs. abnormal event sequences

## Dataset Information

### Available Datasets
- **BGL (Blue Gene/L)**: `ad_bgl_chatgpt.json` - Supercomputer system logs
- **Thunderbird (TB)**: `ad_tb_chatgpt.json` - High-performance computing cluster logs

### Dataset Characteristics
- **BGL Dataset**: 
  - Source: IBM Blue Gene/L supercomputer
  - Scale: Large-scale parallel computing environment
  - Anomalies: Hardware failures, software errors, performance degradation
  
- **Thunderbird Dataset**:
  - Source: Sandia National Labs cluster
  - Scale: High-performance computing environment
  - Anomalies: System crashes, network issues, resource exhaustion

### Anomaly Types
- **Hardware Failures**: Component malfunctions, power issues, connectivity problems
- **Software Errors**: Application crashes, memory leaks, deadlocks
- **Performance Issues**: Resource bottlenecks, throughput degradation
- **Security Incidents**: Unauthorized access, suspicious activities

## Model Architecture

### Small Language Model (SLM)
- **Base Model**: BERT-based sequence classification
- **Architecture**: Transformer encoder with binary classification head
- **Input Handling**: Sequence-level encoding with attention mechanisms
- **Training**: Fine-tuned on labeled normal/anomalous sequences

### Large Language Model (LLM)
- **Model**: GPT-3.5-turbo-16k
- **Capability**: Advanced pattern recognition and contextual reasoning
- **Domain Knowledge**: Understanding of system behaviors and failure patterns

## Anomaly Detection Framework

### Normal Behavior Patterns
- **Routine Operations**: Regular system maintenance, scheduled tasks
- **Expected Sequences**: Standard startup/shutdown procedures
- **Periodic Events**: Scheduled backups, health checks, monitoring reports
- **Operational Flows**: Normal user activities, data processing workflows

### Anomaly Indicators
- **Unexpected Sequences**: Unusual event ordering or timing
- **Error Cascades**: Propagating failures across system components
- **Resource Violations**: Memory/disk/CPU threshold breaches
- **Security Violations**: Authentication failures, unauthorized access attempts

### Temporal Pattern Analysis
- **Sequence Dependencies**: Events that must occur in specific orders
- **Timing Constraints**: Events with expected time intervals
- **Frequency Analysis**: Unusual event occurrence rates
- **Correlation Detection**: Related events across different system components

## Error-prone Case Retrieval (ECR)

### Purpose
ECR provides the LLM with examples of challenging anomaly detection cases to improve classification accuracy for ambiguous log sequences.

### Prompt Template
```
You are a professional Operations Engineer and your task is to analyze whether an anomaly exists in the given log sequence, and output True if it exists and False if it does not.
You can learn key features from these cases and attention common pitfalls in prediction.
{error_cases}

The following input is the test data.
Please 1. Describe the reasoning process (e.g. Reason: xxx) firstly by combining these cases with your own domain knowledge of input logs. 2. Finally, follow the label format (e.g. Label: xxx) of examples and give a definite result.
Input Log Sequence: {log_sequence}
```

## Training Process

### Sequence Preparation
1. **Temporal Alignment**: Ensure chronological ordering of log messages
2. **Sequence Segmentation**: Create fixed-length or sliding window sequences
3. **Label Assignment**: Assign anomaly labels to entire sequences
4. **Data Augmentation**: Generate synthetic anomalous sequences

### Model Training
```bash
# Adapted training process for AD
python ad_small_model_train.py \
    --train_data ./datasets/anomaly_detection/ad_bgl_train.json \
    --dev_data ./datasets/anomaly_detection/ad_bgl_dev.json \
    --pretrain_model bert-base-uncased \
    --sequence_length 50 \
    --epoch 10 \
    --batch_size 8
```

### Training Configuration
- **Loss Function**: Binary cross-entropy for anomaly classification
- **Sequence Handling**: Padding/truncation for variable-length sequences
- **Class Balancing**: Weighted loss to handle imbalanced anomaly data

## Evaluation Process

### SLM Evaluation
```bash
python ad_small_model_pred.py \
    --test_data ./datasets/anomaly_detection/ad_bgl_test.json \
    --pretrain_model ad_slm.pt \
    --sequence_length 50
```

### Adaptive Framework Evaluation
1. **Uncertainty Assessment**: Identify sequences with unclear anomaly status
2. **ECR Application**: Provide relevant anomalous examples to LLM
3. **Final Classification**: Combine SLM and LLM anomaly assessments

## Evaluation Metrics

### Binary Classification Metrics
- **Accuracy**: Overall anomaly detection accuracy
- **Precision**: Accuracy of anomaly predictions (reduces false alarms)
- **Recall**: Coverage of actual anomalies (reduces missed incidents)
- **F1 Score**: Balanced measure of precision and recall
- **AUC-ROC**: Area under the receiver operating characteristic curve

### Operational Metrics
- **False Positive Rate**: Normal sequences incorrectly flagged as anomalous
- **False Negative Rate**: Anomalous sequences missed by the detector
- **Mean Time to Detection (MTTD)**: Average time to identify anomalies
- **Alert Fatigue**: Impact of false positives on operations teams

### Specialized AD Metrics
- **Precision at K**: Precision considering top-K most confident predictions
- **Early Detection Rate**: Percentage of anomalies detected early in sequence
- **Anomaly Localization**: Accuracy in identifying anomalous events within sequences

## Common Challenges

### Imbalanced Data
- **Challenge**: Anomalies are rare events in normal system operations
- **Impact**: Models biased toward predicting normal behavior
- **Solution**: Class weighting, synthetic anomaly generation, cost-sensitive learning

### Temporal Dependencies
- **Challenge**: Anomalies may span multiple time windows
- **Example**: Gradual system degradation over extended periods
- **Solution**: Multi-scale sequence analysis, sliding window approaches

### Context Sensitivity
- **Challenge**: Same event sequence may be normal or anomalous depending on context
- **Example**: High CPU usage during business hours vs. off-hours
- **Solution**: Context-aware features, temporal pattern modeling

### Novel Anomalies
- **Challenge**: Previously unseen anomaly patterns
- **Example**: New attack patterns, unprecedented failure modes
- **Solution**: LLM reasoning for novel pattern recognition

## Adaptive Selection Strategy

### Uncertainty Indicators
1. **Prediction Confidence**: Low softmax probability for either class
2. **Sequence Complexity**: Unusual patterns not seen in training
3. **Temporal Inconsistencies**: Events that violate expected temporal relationships

### Routing Logic
```python
# Simplified routing decision
prediction_confidence = max(anomaly_probability, normal_probability)
sequence_novelty = calculate_novelty_score(sequence)

if prediction_confidence < confidence_threshold or sequence_novelty > novelty_threshold:
    anomaly_status = query_llm_with_ecr(sequence, error_cases)
else:
    anomaly_status = slm_prediction
```

## Performance Characteristics

### SLM Performance
- **Speed**: Real-time processing for continuous monitoring
- **Baseline Accuracy**: Good performance on known anomaly patterns
- **Limitation**: Struggles with novel or complex anomaly patterns

### Adaptive Framework Benefits
- **Improved Recall**: Better detection of subtle and novel anomalies
- **Reduced False Positives**: Expert reasoning reduces false alarms
- **Cost Efficiency**: 70% reduction in LLM usage compared to LLM-only

## Usage Examples

### Basic Anomaly Detection
```bash
# Train on BGL dataset
python ad_small_model_train.py \
    --train_data ./datasets/anomaly_detection/ad_bgl_train.json \
    --dev_data ./datasets/anomaly_detection/ad_bgl_dev.json \
    --sequence_length 100
```

### Production Monitoring
```bash
# Step 1: Real-time anomaly detection
python ad_small_model_pred.py \
    --input_stream system_logs.stream \
    --model ad_model.pt \
    --real_time \
    --output anomaly_alerts.json

# Step 2: Uncertain case analysis
python ad_uncertainty_analysis.py \
    --predictions anomaly_alerts.json \
    --threshold 0.7 \
    --output uncertain_sequences.json

# Step 3: LLM validation
python query_ChatGPT.py \
    --data uncertain_sequences.json \
    --task anomaly_detection \
    --output validated_anomalies.json
```

### Interactive Analysis
```python
# Example API usage
from adaptivelog import AnomalyDetector

detector = AnomalyDetector(model_path="ad_model.pt")
log_sequence = [
    "System startup initiated",
    "Loading configuration files",
    "Network interfaces initialized", 
    "Critical error: Memory allocation failed",
    "System attempting recovery",
    "Recovery failed - initiating shutdown"
]

is_anomaly = detector.detect_anomaly(log_sequence)
print(f"Anomaly detected: {is_anomaly}")
```

## Best Practices

### Data Preparation
1. **Temporal Integrity**: Maintain chronological ordering of log sequences
2. **Sequence Boundaries**: Define logical sequence boundaries (sessions, processes)
3. **Label Quality**: Ensure accurate anomaly labeling with domain experts

### Model Training
1. **Sequence Length**: Optimize sequence length for anomaly detection window
2. **Class Balancing**: Handle imbalanced normal/anomalous data appropriately
3. **Validation Strategy**: Use temporal splits to avoid data leakage

### Operational Deployment
1. **Threshold Tuning**: Optimize detection thresholds for operational requirements
2. **Alert Management**: Implement intelligent alerting to reduce false positives
3. **Feedback Loop**: Incorporate operations team feedback for continuous improvement

## Integration with Operations

### Real-time Monitoring
- **Stream Processing**: Continuous analysis of incoming log streams
- **Sliding Windows**: Moving window analysis for temporal anomalies
- **Alert Generation**: Immediate notifications for detected anomalies

### Incident Response
- **Priority Classification**: Severity-based prioritization of anomaly alerts
- **Context Provision**: Additional information for incident responders
- **Escalation Rules**: Automatic escalation based on anomaly characteristics

### Historical Analysis
- **Trend Detection**: Long-term anomaly pattern analysis
- **Root Cause Analysis**: Correlation with system changes and incidents
- **Performance Baseline**: Update normal behavior baselines over time

## Advanced Features

### Multi-modal Anomaly Detection
- **Log + Metrics**: Combine log analysis with system metrics
- **Cross-system Correlation**: Detect anomalies across multiple systems
- **Behavioral Profiling**: Learn normal behavior patterns for different contexts

### Explainable Anomaly Detection
- **Anomaly Attribution**: Identify specific events causing anomaly classification
- **Feature Importance**: Highlight key sequence characteristics
- **Visual Analysis**: Timeline visualization of anomalous sequences

### Adaptive Learning
- **Online Learning**: Continuous model updates with new data
- **Concept Drift**: Adaptation to changing system behaviors
- **Feedback Integration**: Learning from operations team corrections

## Domain-Specific Applications

### High-Performance Computing
- **Job Failure Prediction**: Early detection of computation failures
- **Resource Contention**: Identification of resource allocation issues
- **Performance Degradation**: Detection of decreasing system efficiency

### Enterprise Networks
- **Security Monitoring**: Intrusion and attack pattern detection
- **Performance Issues**: Network congestion and routing problems
- **Configuration Changes**: Unauthorized or problematic configuration modifications

### Cloud Infrastructure
- **Service Failures**: Microservice and container anomalies
- **Scaling Issues**: Auto-scaling and resource allocation problems
- **Multi-tenant Isolation**: Cross-tenant interference detection