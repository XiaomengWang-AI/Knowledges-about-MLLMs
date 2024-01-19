# Paper-List
This repository is about security on multimodal large-scale language models (MLLMs). 
Firstly, it contains basic characteristics of the mainstream models, and these MLLMs are classified into several types simply. 
Then, it introduces threats on different MLLMs, mainly about adversarial attack. 
Additionally, it gives other introduction about other threats, like backdoor attack, privacy leaking.
## Multimodal large-scale Language Models (MLLMs)
Firstly, we have to clarify some key terms, which are usually used chaotically in papers. 

**Computer vision (CV) tasks:** image classification, object detection, image captioning, 

**Nautral language processing (NLP) tasks:** speech recognition, text prediction, machine translation.

**Language model (LM)** is a statistical model that predicts the probability of a sequence of words, e.g., RNN, LSTM, which is used in various NLP tasks.

**Large-scale language models (LLMs)** are a subset of language models that are distinguished by their size (both in terms of the model architecture and the training data). 
They are typically built using deep learning techniques and have a large number of parameters, which are often based on transformer architectures, like GPT (Generative Pre-trained Transformer), BERT (Bidirectional Encoder Representations from Transformers). 
Beyond standard NLP tasks, they are capable of more complex tasks like writing essays, generating creative content, summarizing text, answering questions, and even generating code, e.g., GPT-3 by OpenAI, BERT by Google, T5 (Text-To-Text Transfer Transformer).

**Vision-laguage pre-training (VLP) model**

**Vision-language model** (also called visual-language model) (VLMs) are popular nowadays. 
VLMs denote models that can process (input or output) different modalities (mainly are image and text). 
Traditionally, models for tasks, like image caption, sentiment analysis, visual question and answering, are included into VLMs. 
There are various design in model structure. In 2021, ClIP is firstly proposed, which is unimodal and used for zero-shot classification. 
Image embeddings and text embeddings separately comes out of CLIP, in other words, image and text are simultaneously encoded by respective encoder. 
With the rate of progress being that it is, BLIP, BLIP-2, InstructBLIP, Flamingo are proposed that are multimodal, whose features learned by making use of multiple modes. 
These models are mainly for generation task, e.g., image generation, image caption, VQA.

**Multimodal large language models (MLLMs)** are also called large multimodal models(LMMs). They can process different modalities (e.g., audio, video, not only image and text), e.g., GPT4.0.
The high-level overview of how they work are as follows: 
1. An encoder for each data modality to generate the embeddings for data of that modality. Most image encoders are based on the architecture of Vision Transformer (ViT).
2. A way to align embeddings of different modalities into the same multimodal embedding space.
3. [Generative models only] A language model to generate text responses. Since inputs can contain both text and visuals, new techniques need to be developed to allow the language model to condition its responses on not just text, but also visuals.

**Other confusing concepts**
1. GPT and ChatGPT 
**GPT** refers to the underlying language model technology, including GPT-2, GPT-3 and GPT-4. 
They are pre-trained on vast amounts of text data and are capable of generating human-like text based on the input they receive. 
GPT models can be used for a wide range of natural language processing tasks, such as text generation, translation, summarization, and more.
**ChatGPT** is a specific application of the GPT models, tailored for conversational use. 
While it is built upon the same GPT architecture, ChatGPT is fine-tuned with a focus on generating coherent and contextually relevant dialogue. 
This makes it particularly suitable for applications like chatbots or conversational agents. 
It is designed to maintain a conversation with users, providing responses that are more aligned with a conversational context.

2. Prompt and instruction


## Security on MLLMs
### Adversarial attack on VLMs

### Other threats on VLMs


## Other Concepts
1. **RAG (Retrieval Augmented Generation)**
2. **TTS (text-to-speech)**

## References
1. Exploring Multimodal Large Language Models: A Step Forward in AI. https://medium.com/@cout.shubham/exploring-multimodal-large-language-models-a-step-forward-in-ai-626918c6a3ec
2. Extension on BLIP model series: https://raghul-719.medium.com/neural-networks-intuitions-17-blip-series-blip-blip-2-and-instruct-blip-papers-explanation-2378bc860d53
