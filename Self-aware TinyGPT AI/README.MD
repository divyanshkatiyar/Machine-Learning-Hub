# 🤖 TinyGPT — Self-Aware TinyML AI

A lightweight, neural-assisted conversational AI built from scratch using Python and NumPy.

TinyGPT is an experimental, GPT-style chatbot designed to explore language modelling, conversational intelligence, persistent memory, interactive learning, and functional self-modelling on resource-constrained devices.

Unlike projects that simply connect to a large, externally hosted language model, TinyGPT implements its own compact neural language model and combines neural text generation with knowledge retrieval, deterministic responses, basic reasoning, and persistent local storage.

## 🚧 Development Status: Under Development

TinyGPT is an experimental AI project. Its architecture, conversational capabilities, learning mechanisms, and response quality are still being developed and evaluated. It is not a production-ready AI system.»

## ✨ Key Features

|Feature| Description|
|:--- |:--- |
|🧠 GPT-Style Neural Model |Compact language model implemented from scratch using NumPy.|
|🔤 Byte-Level Tokenization| Processes UTF-8 text using a vocabulary of 256 possible byte values.|
|💬 Conversational Interface| Supports interactive, text-based conversations through a command-line interface.|
|🔎 Knowledge Retrieval| Retrieves relevant answers from built-in knowledge and locally stored question-answer pairs.|
|💾 Persistent Memory| Saves supported facts and knowledge to local JSONL files for retrieval across sessions.|
|📚 Interactive Learning| Accepts certain natural-language statements & stores useful information automatically.|
|🔄 Incremental Training| Trains the neural model on pending saved examples and persists updated model parameters.|
|🪞 Functional Self-Model| Describes its identity, capabilities, simulated feelings, goals, wishes, and limitations through programmed responses.|
|🧮 Basic Reasoning| Includes support for arithmetic, Boolean logic, and selected logical inference tasks.|
|🛡️ Response Validation| Filters neural-generated text for readability and relevance before using it as a response.|
|📱 Android-Oriented Design| Designed with lightweight dependencies and compatibility with environments such as Pydroid in mind.|
|🧪 Integrated Self-Tests| Includes tests for model generation, training, memory, reasoning, and self-model responses.|

## 🏗️ Technical Architecture

1. Neural Language Model

The project implements a compact GPT-style language model from scratch.

Its principal components include:

- Byte-level token embeddings.
- Positional embeddings.
- Query, key, and value projections.
- Causal self-attention.
- Feed-forward neural layers.
- Layer normalization.
- Residual connections.
- Softmax-based output probabilities.
- Cross-entropy training loss.
- Manual backpropagation.
- Adam optimization.

The model learns patterns from training sequences by updating its numerical parameters. During generation, it predicts subsequent byte tokens based on the preceding context.

> 🐍 The implementation uses NumPy for its numerical computations rather than relying on a pretrained transformer library.

2. Conversational Response Pipeline

A user message passes through a response pipeline that combines neural generation with deterministic processing.

The general sequence is:

1. Generate a candidate response using the neural model.
2. Recognize supported conversational teaching statements.
3. Check for relevant user-specific memories.
4. Handle supported arithmetic and logical reasoning tasks.
5. Process programmed identity and self-model responses.
6. Retrieve relevant built-in or user-provided knowledge.
7. Resolve selected conversational follow-ups.
8. Evaluate whether a novel neural response is sufficiently readable and relevant.
9. Use an appropriate fallback response when necessary.

This hybrid design helps prevent malformed neural output from being displayed as a meaningful answer.

> ⚠️ A readable response is not necessarily a factually correct response. Response filtering reduces some generation problems but does not eliminate hallucinations or reasoning errors.

3. Functional Self-Model

TinyGPT includes programmed responses about its own identity, capabilities, goals, limitations, through simulated, predefined internal states.

It can respond to questions concerning:

- Its identity and architecture.
- Its programmed goals and purpose.
- Simulated feelings and emotional descriptions.
- Its wishes and desires.
- Its limitations and potential problems.
- Its memory and learning mechanisms.
- The distinction between software behaviour and human consciousness.

> ⚠️ The term self-aware describes this project's experimental functional self-modelling capabilities. It does not establish consciousness, subjective experience, independent will, or genuine human-like emotions.

## ⚙️ Model Configuration

The principal model settings are defined near the beginning of "TinyGPT_Chatbot_v3.py".

|Parameter |Default |Purpose |
|:--- |:--- |:--- |
|Vocabulary size| 256| Number of possible byte tokens|
|Context block size| 128| Maximum token context processed by the model|
|Embedding size| 48| Dimensionality of token and positional representations|
|Hidden size| 96| Width of the feed-forward hidden representation|
|Learning rate| 0.003| Optimizer learning-rate setting|
|Training steps| 20| Default number of steps used by the training routine|
|Automatic training steps| 6| Steps used for each pending saved example|
|Maximum generation setting| 80| Configured generation-length limit|
|Neural reply token setting| 64| Configured neural reply-generation limit|
|Temperature| 0.72| Configured sampling temperature|
|Random seed| 42| Initial NumPy random seed|

> ⚠️ These are configurable defaults, not performance guarantees. Actual behaviour depends on the model implementation, training examples, saved parameters, and runtime environment.

## 📚 Persistent Memory and Learning

TinyGPT maintains local files for its model, training examples, and knowledge.

|File |Purpose |
|:--- |:--- |
|"tinygpt_v3_model.npz"| Stores neural model parameters and Adam optimizer state.|
|"tinygpt_v2_training.jsonl"| Stores question-answer training examples and their training status.|
|"tinygpt_v2_knowledge.jsonl"| Stores question-answer pairs used by the knowledge retrieval system.|

> The filenames reflect the current source-code configuration. They can be changed by modifying the corresponding configuration variables.

**Interactive Learning**

TinyGPT can recognize certain statements written in natural language and save useful information.

For example:

```
You: My favorite color is blue.
AI: [Learning acknowledgement]
```

The program can save supported personal facts and generate corresponding question-answer records.

A saved training example uses this structure:

```
{
  "prompt": "What is a transistor?",
  "answer": "A transistor is a semiconductor device used for switching or amplification.",
  "trained": false
}
```

The actual files use JSON Lines (JSONL) format, meaning each record is stored as a separate JSON object on its own line.

Saved knowledge can be retrieved in later sessions. Pending training examples can also be used to update the neural model. 

> However, persistent storage, information retrieval, and neural training are different mechanisms. Saving a fact does not guarantee that the model has learned it reliably or will always reproduce it correctly.

The current implementation does not provide unrestricted autonomous learning, independent goal formation, or automatic discovery of new training data from the internet.

## 🛠️ Technology Stack

Technology| Role
Python| Core implementation and application logic
NumPy| Neural-network computations, numerical operations, and parameter updates
JSON| Structured representation of saved records
JSONL| Persistent training examples and knowledge records
NPZ| Storage of model parameters and optimizer state
"unittest"| Integrated automated testing
Pydroid| Intended Android Python development and execution environment

External machine-learning frameworks: The implementation does not use TensorFlow or PyTorch.

The program is designed to run locally and does not require a hosted language-model API for its core conversational functionality.

## 🚀 Getting Started

On its first launch, the program initializes the model and performs its configured bootstrap training. Subsequent launches attempt to load the saved model and train pending examples.

Run Automated Tests

Execute the integrated self-test suite:

python TinyGPT_Chatbot_v3.py --self-test

The tests cover selected behaviours, including:

- Neural text generation.
- Neural parameter updates during training.
- Use of neural generation in the response pipeline.
- Identity and self-model responses.
- Appropriate handling of consciousness-related questions.
- Simulated feelings, desires, wishes, and limitations.
- Knowledge retrieval and emotional-response handling.
- Arithmetic and Boolean logic.
- Selected syllogistic reasoning.
- Neural-output readability filtering.
- Persistent personal-memory retrieval.

Passing these tests demonstrates that the tested code paths behave as expected under the test conditions. It does not prove general intelligence, universal correctness, or production readiness.

## 🧪 Development Roadmap

TinyGPT is an ongoing experimental project. Potential areas for further development include:

- Conversational quality: Improve response coherence, relevance, and consistency across longer conversations.
- Knowledge management: Improve fact validation, memory organization, retrieval accuracy, and conflict resolution.
- Learning reliability: Evaluate training stability and measure whether new examples improve responses.
- Self-model consistency: Better integrate the programmed identity and simulated internal-state descriptions with actual application state.
- Reasoning: Expand and test mathematical, logical, and technical question-answering capabilities.
- Model evaluation: Develop broader test datasets and quantitative evaluation methods.
- Performance: Optimize training and inference for resource-constrained Android devices.
- Data integrity: Improve validation, recovery, and protection of persistent files.

These are development directions, not claims that the capabilities have already been implemented.

## ⚠️ Limitations and Warnings

1. Experimental model

TinyGPT is a compact educational implementation, not a substitute for a modern, large-scale language model. Its limited model capacity and training data restrict the complexity and reliability of its responses.

2. Self-awareness disclaimer

The self-model and simulated emotional states are software-defined behaviours. They should not be interpreted as evidence that TinyGPT is conscious, experiences emotions, or possesses subjective awareness.

3. Learning limitations

The program supports persistent knowledge and incremental training, but these mechanisms do not guarantee continuous improvement. Training can fail to generalize, introduce undesirable behaviour, or reduce the quality of other responses.

4. Accuracy warning

The chatbot may produce incorrect, incomplete, irrelevant, or misleading answers. Do not rely on it as the sole source for medical, legal, financial, safety-critical, or other high-stakes decisions.

5. Local data warning

Training examples and knowledge records are stored in local files. Review these files before sharing them or committing them to a public repository. Do not store passwords, authentication tokens, private credentials, or other sensitive information in the chatbot's memory.

6. Resource limitations

Model initialization, training, and generation consume processing time and memory. Performance varies across devices, and Android compatibility does not imply identical performance on every device.

7. No guaranteed autonomous behaviour

TinyGPT does not independently acquire unrestricted knowledge, establish its own goals, or modify its entire source code autonomously. Its learning and memory operations are governed by the implemented program logic.

## 📜 Disclaimer

TinyGPT is provided as an experimental software project for educational, research, and development purposes.

The software is provided “AS IS”, without guarantees of accuracy, reliability, uninterrupted operation, or fitness for a particular purpose, to the maximum extent permitted by applicable law.

The author is not responsible for decisions made solely on the basis of the chatbot's outputs, loss or corruption of locally stored data, or consequences arising from the use or modification of the software. Users are responsible for evaluating generated responses, maintaining backups, and using the software appropriately.

Nothing in this project should be interpreted as a scientific demonstration of artificial consciousness or human-like subjective experience.

## 🤝 Contributions

Contributions, bug reports, testing feedback, and technical suggestions are welcome.

Useful contributions include:

- Reproducible bug reports.
- Improvements to neural-model implementation.
- Better knowledge retrieval and memory handling.
- Additional automated tests.
- Performance optimizations.
- Documentation improvements.
- Evaluation of response quality and training behaviour.

When reporting an issue, include the relevant error message, execution environment, steps to reproduce the problem, and a minimal example where possible. Avoid sharing private memory files or sensitive training data.

---
