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

**Vision-language model** (also called visual-language model) (VLMs) are popular nowadays. 
VLMs denote models that can process (input or output) different modalities (mainly are image and text). 
Traditionally, models for tasks, like image caption, sentiment analysis, visual question and answering, are included into VLMs. 
There are various design in model structure. In 2021, ClIP is firstly proposed, which is unimodal and used for zero-shot classification. 
Image embeddings and text embeddings separately comes out of CLIP, in other words, image and text are simultaneously encoded by respective encoder. 
With the rate of progress being that it is, BLIP, BLIP-2, InstructBLIP, Flamingo are proposed that are multimodal, whose features learned by making use of multiple modes. 
These models are mainly for generation task, e.g., image generation, image caption, VQA.

**Multimodal large-scale language models (MLLMs)** can process different modalities (e.g., audio, video, not only image and text), e.g., GPT4.0.


## Security on MLLMs
### Adversarial attack on VLMs

### Other threats on VLMs
