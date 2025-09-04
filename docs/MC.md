# MC: Module Classification

## Overview

Module Classification (MC) is a fundamental log analysis task in the AdaptiveLog framework that focuses on identifying the system module responsible for generating a specific log message. This task is crucial for network operations as it helps engineers quickly understand which system component is experiencing issues and route incidents to the appropriate specialist teams.

## Task Definition

MC performs multi-class classification to determine the source module of log messages where the module name has been masked. Given:
- **Input**: A log message with the module name replaced by `[MASK]`
- **Module List**: Predefined set of possible system modules
- **Output**: The correct module name from the available options

The task requires understanding of log message patterns, parameter types, and system architecture to accurately identify the originating module.

## Input/Output Format

### Training Data Structure
```json
[
  [
    "[MASK]/6/CREATE_HLP_TIMER:OSPF [process-id] helper waits for timer to be created for interface [interface-name].",
    "OSPF"
  ],
  [
    "[MASK]/3/IN_SHORT_PKT_E:The IP packet received is too short. (ProcessId=[USHORT], PacketType=[USHORT], SourceAddress=[IPADDR], DestinationAddress=[IPADDR])",
    "OSPF"
  ],
  [
    "[MASK]/4/GENERATE_CKP_NO_SPACE_BASE: The base checkpoint was not generated because the storage space was not enough.",
    "CONFIGURATION"
  ]
]
```

### Format Explanation
- **First element**: Log message with module masked as `[MASK]`
- **Second element**: Correct module name (ground truth label)

### Log Structure
Network device logs follow the format:
```
MODULE/SEVERITY/MESSAGE_ID:Description with parameters
```
Where the MODULE field is masked for classification.

## Dataset Information

### Available Datasets
- **Huawei Switch**: `mc_hwswitch_train.json`, `mc_hwswitch_dev.json`, `mc_hwswitch_test.json`
- **Huawei Router**: `mc_hwrouters_train.json`, `mc_hwrouters_dev.json`, `mc_hwrouters_test.json`
- **Cisco Switch**: `mc_csswitch_train.json`, `mc_csswitch_dev.json`, `mc_csswitch_test.json`
- **Cisco Router**: `mc_csrouters_train.json`, `mc_csrouters_dev.json`, `mc_csrouters_test.json`

### Module Categories

#### Network Protocols
- **OSPF**: Open Shortest Path First routing protocol
- **ISIS**: Intermediate System to Intermediate System routing
- **BGP**: Border Gateway Protocol for inter-domain routing
- **PIM**: Protocol Independent Multicast
- **VRRP**: Virtual Router Redundancy Protocol
- **LACP**: Link Aggregation Control Protocol

#### System Components
- **CONFIGURATION**: System configuration management
- **DEVM**: Device management functions
- **WLAN**: Wireless LAN management
- **RM**: Route management
- **SHELL**: Command line interface and user sessions
- **ALML**: Alarm management

#### Infrastructure
- **MBR**: Member/board management
- **BASETRAP**: Base system traps and notifications
- **INFO**: Information center and logging
- **PM**: Performance monitoring
- **AAA**: Authentication, Authorization, and Accounting

## Model Architecture

### Small Language Model (SLM)
- **Base Model**: BERT-base-uncased
- **Architecture**: Multi-class classification with softmax output layer
- **Vocabulary**: Extended with network protocol and system terminology
- **Training**: Fine-tuned on masked module classification task

### Large Language Model (LLM)
- **Model**: GPT-3.5-turbo-16k
- **Advantage**: Extensive knowledge of network protocols and system architectures
- **Context**: Leverages detailed understanding of log message semantics

## Classification Strategy

### Feature Extraction
1. **Message Content**: Log description and error patterns
2. **Parameter Types**: Interface names, IP addresses, protocol IDs
3. **Severity Levels**: Different modules have characteristic severity distributions
4. **Message Patterns**: Module-specific terminology and phrasing

### Module Identification Patterns

#### OSPF Module Indicators
- **Keywords**: "neighbor", "LSA", "area", "router-id"
- **Parameters**: Process IDs, router addresses, interface indices
- **Common Messages**: Neighbor state changes, LSA updates, area transitions

#### ISIS Module Indicators
- **Keywords**: "adjacency", "circuit", "level", "LSP"
- **Parameters**: System IDs, circuit names, level specifications
- **Common Messages**: Adjacency formation, LSP flooding, authentication

#### WLAN Module Indicators
- **Keywords**: "AP", "station", "SSID", "radio"
- **Parameters**: MAC addresses, AP IDs, radio indices
- **Common Messages**: AP connectivity, client associations, interference

#### System Module Indicators
- **CONFIGURATION**: Checkpoint, backup, rollback operations
- **DEVM**: Hardware status, board management, entity traps
- **SHELL**: User login, command execution, session management

## Error-prone Case Retrieval (ECR)

### Purpose
ECR helps the LLM identify modules for ambiguous log messages by providing examples of similar cases where module classification was challenging.

### Prompt Template
```
You are a professional operation and maintenance engineer with a wealth of knowledge, and your task is to analyze the module to which a given huawei switch log belongs. Requirement: You need to utilize your extensive knowledge to deeply analyze the contents of the logs and choose one answer to output from the following module list.
All Modules: {module_list}
The input is a log whose module name is masked with [MASK], and the output is to find the module to which the log belongs from the Module list above.

You can refer to the following error-prone cases, note key features based on these cases and common pitfalls in prediction.
{error_cases}

The following input is the test data.
Please 1. Describe the reasoning process (e.g. Reason: xxx) firstly by referring to the reasons of relevant error-prone cases. 2. Finally, follow the label format (e.g. Label: xxx) of examples and give a definite result.
Input: {log_message}
```

## Training Process

### Data Preprocessing
1. **Module Masking**: Replace module names with `[MASK]` token
2. **Label Encoding**: Convert module names to numerical labels
3. **Vocabulary Extension**: Add domain-specific terms to tokenizer

### Model Training
```bash
# Adapted training process for MC
python mc_small_model_train.py \
    --train_data ./datasets/adaptive_mc/mc_hwswitch_train.json \
    --dev_data ./datasets/adaptive_mc/mc_hwswitch_dev.json \
    --pretrain_model bert-base-uncased \
    --num_classes 50 \
    --epoch 5 \
    --batch_size 16
```

### Training Configuration
- **Loss Function**: Cross-entropy loss for multi-class classification
- **Optimization**: AdamW with learning rate scheduling
- **Regularization**: Dropout and weight decay for generalization

## Evaluation Process

### SLM Evaluation
```bash
python mc_small_model_pred.py \
    --test_data ./datasets/adaptive_mc/mc_hwswitch_test.json \
    --pretrain_model mc_slm.pt \
    --module_list modules.txt
```

### Adaptive Framework Evaluation
1. **Confidence Assessment**: Identify low-confidence predictions
2. **ECR Application**: Provide relevant examples to LLM
3. **Final Classification**: Integrate SLM and LLM predictions

## Evaluation Metrics

### Classification Performance
- **Accuracy**: Overall module classification accuracy
- **Top-k Accuracy**: Accuracy considering top-k predictions
- **Per-Module Precision/Recall**: Performance breakdown by module
- **Confusion Matrix**: Detailed classification error analysis

### Operational Metrics
- **Routing Accuracy**: Correct incident routing to specialist teams
- **Response Time**: Reduction in incident triage time
- **False Routing Rate**: Incorrect team assignments due to misclassification

## Common Challenges

### Overlapping Functionality
- **Challenge**: Multiple modules may handle similar operations
- **Example**: Both OSPF and ISIS handle routing, may have similar log patterns
- **Solution**: ECR provides distinguishing examples between similar modules

### Parameter Ambiguity
- **Challenge**: Generic parameters used across multiple modules
- **Example**: Interface indices appear in network and system logs
- **Solution**: Context-aware analysis using surrounding message content

### Vendor Variations
- **Challenge**: Different vendors use different module naming conventions
- **Example**: Huawei vs. Cisco module hierarchies and naming
- **Solution**: Vendor-specific model training and adaptation

### Novel Module Types
- **Challenge**: New software features introduce previously unseen modules
- **Example**: SDN controllers, cloud integration modules
- **Solution**: LLM reasoning for handling novel module patterns

## Adaptive Selection Strategy

### Uncertainty Indicators
1. **Low Confidence Scores**: Softmax probabilities below threshold
2. **Close Competing Classes**: Multiple modules with similar scores
3. **Novel Message Patterns**: Previously unseen log message structures

### Routing Logic
```python
# Simplified routing decision
prediction_confidence = max(softmax_scores)
second_best = sorted(softmax_scores)[-2]
confidence_gap = prediction_confidence - second_best

if prediction_confidence < confidence_threshold or confidence_gap < min_gap:
    module = query_llm_with_ecr(log, module_list, error_cases)
else:
    module = slm_prediction
```

## Performance Characteristics

### SLM Performance
- **Speed**: Sub-second classification for most logs
- **Accuracy**: High performance on common module patterns
- **Scalability**: Handles large volumes of log messages efficiently

### Adaptive Framework Benefits
- **Accuracy Improvement**: 12-18% improvement on challenging cases
- **Coverage**: Better handling of edge cases and novel patterns
- **Cost Efficiency**: 65% reduction in LLM usage compared to LLM-only

## Usage Examples

### Basic Module Classification
```bash
# Train on Cisco router logs
python mc_small_model_train.py \
    --train_data ./datasets/adaptive_mc/mc_csrouters_train.json \
    --dev_data ./datasets/adaptive_mc/mc_csrouters_dev.json \
    --num_classes 45
```

### Production Deployment
```bash
# Step 1: Initial classification
python mc_small_model_pred.py \
    --test_data production_logs.json \
    --model mc_model.pt \
    --output initial_classifications.json

# Step 2: Uncertainty analysis
python mc_uncertainty_analysis.py \
    --predictions initial_classifications.json \
    --threshold 0.8 \
    --output uncertain_logs.json

# Step 3: LLM refinement
python query_ChatGPT.py \
    --data uncertain_logs.json \
    --task module_classification \
    --module_list all_modules.txt
```

### Real-time Classification API
```python
# Example API usage
from adaptivelog import ModuleClassifier

classifier = ModuleClassifier(model_path="mc_model.pt")
log = "[MASK]/4/NBR_STATE_CHANGE: Neighbor state changed to Full"
module = classifier.classify(log)
print(f"Predicted module: {module}")
```

## Best Practices

### Data Quality
1. **Balanced Distribution**: Ensure adequate representation of all modules
2. **Quality Control**: Validate module assignments with domain experts
3. **Coverage**: Include diverse log types and scenarios for each module

### Model Optimization
1. **Class Weighting**: Handle imbalanced module distributions
2. **Hierarchical Classification**: Consider module relationships and hierarchies
3. **Cross-validation**: Robust evaluation across different device types

### Operational Integration
1. **Team Routing**: Automatically route incidents to appropriate teams
2. **Priority Assignment**: Module-based priority and escalation rules
3. **Knowledge Base**: Maintain module-specific troubleshooting guides

## Integration with Network Operations

### Incident Management
- **Automated Routing**: Direct incidents to specialist teams based on module
- **Escalation Rules**: Module-specific escalation procedures
- **SLA Management**: Different response times for different modules

### Troubleshooting Workflows
- **Expert Assignment**: Route to engineers with module expertise
- **Context Provision**: Provide module-specific historical data
- **Tool Selection**: Recommend appropriate diagnostic tools per module

### Monitoring and Alerting
- **Module-based Dashboards**: Separate monitoring views per module
- **Threshold Management**: Module-specific alert thresholds
- **Trend Analysis**: Module-wise performance and issue tracking

## Advanced Features

### Hierarchical Classification
- **Module Families**: Group related modules (e.g., routing protocols)
- **Two-stage Classification**: Coarse-to-fine module identification
- **Confidence Propagation**: Use family-level confidence for module decisions

### Multi-device Learning
- **Cross-platform Models**: Training across different device types
- **Transfer Learning**: Adapt models between vendor platforms
- **Domain Adaptation**: Fine-tune for specific network environments

### Temporal Context
- **Sequence Analysis**: Consider log sequences for better module identification
- **Historical Context**: Use previous classifications for consistency
- **Pattern Recognition**: Identify module-specific log patterns over time