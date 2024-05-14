# Knowledges about MLLMs
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


to be sorted:
	1. Multimodal-textual modality and visual modality
	2. Large-scale VLP=Vision-Language pre-training/large-scale pretrained foundation models
	➡️On Multimodal downstream tasks
	Aims to improve performance of downstream vision and language tasks by pretraining the model on large-scale image-text pairs from the web (which also called scale up the dataset)
	The web texts will through the simple rule-based filters, which leads to noise in web texts.
	LLMS=Large language model (e.g., GPT-3, ChatGPT)
	VLMs like CLIP mainly for classification, OCR; Generative VLMs, e.g., FLAMINGO, BLIP, mainly for captioning, VQA
	3. Vision-language understanding and generation tasks: image-text retrieval, image captioning, VQA (visual question answering, visual reasoning, visual dialog)
	4. Generalization ability when directly transferred to video-language tasks in a zero-shot manner
	5. Multi-task pre-training and transfer learning
	6. Encoder-based models (CLIP=Contrastive Language-Image Pre-training, ALBEF) are less straightforward to directly transfer to text generation tasks (image captioning); Encoder-decoder model (SimVLM) are not adopted for image-text retrieval tasks.
Both models are pre-trained on image-text pairs collected from the web.
	5. BLIP=Bootstrapping Language-Image Pre-training
	For unified vision-language understanding and generation
	Image-grounded? How to achieve zero-shot in code?
	6. Visual feature extraction: Visual transformer (ViT) as image encoder, which divides an input image into patches and encodes them as a sequence of embeddings, with an additional [CLS] token to represent the global image feature.
	Text encoder is the same as BERT, where a [CLS] token is appended to the beginning of the text input to summarize the sentence.
	7. Prompting VLMs: emergent behaviors from large scale pretraining are elicited by supplying suitably crafted inputs to the VLMs. Most commonly performed by prepending a set of learnable tokens to the text input, vision input or both.
	Aims to easily steer a frozen CLIP model to solve a desired task,
	8. CLIP is trained to match text and image samples-contrastive;
	Is trained to predict if an image and a text snippet are paired together in WIT;
	objectives to learn image representations from text;
	Perform computer vision tasks from natural language supervision;
	Studying its behavior at large scale
	Pre-training methods which learn directly from raw web text/learn from natural language supervision;
	Text-to-text (input and output) enable task-agnostic architecture to zero-shot learning means generalize to unseen datasets, as a proxy for preforming unseen downstream tasks;
	The difference between the CLIP for image classification and traditional classification model -transformer
	Image-text pairs
	9. Training dataset for CLIP: WIT-400M image-text pairs dataset
	Training dataset for Open-CLIP: LAION-400M--LAION-5B
	One of the biggest advantages of open-vocabulary models like CLIP is that they can be used on downstream classification tasks by carefully designing text prompts corresponding to class descriptions, without requiring any labeled training example. 
	What type of text prompt?
	10. After encoding the image and text (in the dataset) into embeddings, the loss computed by cosine distance is optimized, the model is trained to match text and image more accurately. Then the pretrained-model can be used for other downstream tasks.
	11. What is the difference between the encoder-based model and the encoder-decoder model? 
	12. Large-scale pre-trained model is a generative model or what?
	13. Why sometimes fine-tune? Sometimes zero-shot transfer? Why poor performance of zero-shot transfer on some downstream tasks?
	14. Instruction tuning is a technique that allows a model to adapt to different tasks by following natural language instructions.
![image](https://github.com/XiaomengWang-AI/Knowledges-about-MLLMs/assets/76413485/76afeec1-0047-41d5-842f-6d39915ef971)

	1. Pre-trained language model: GPT-3, BERT
	ChatGPT-alignment-tuned language model
	2. Aligning a language model typically refers to adapting or fine-tuning a pre-trained language model for a specific task or domain (named alignment task), such as question answering, sentiment analysis, translation, summarization and any other natural language processing task (NLP).
	3. How to train pretrained models to align with the preferences and goals of their creators:
	Reinforcement learning through human feedback (RLHF)
	Alignment techniques---alignment-trained text-only models
	AI alignment
	Align base language pretrained model with certain desired principles, thorough alignment techniques like instruction tuning and reinforcement learning via human feedback (RLHF)
	Multimodal text-vision model start with a standard pre-trained language model that tokenizes and then processes the embedding layers
	4. Jailbreaks of chatbots: e.g., ChatGPT or Bard where a user uses an adversarial example ( a maliciously designed prompt) to elicit the desired unaligned behavior 
	5. 


Large languge model==input and output are texts
Multi-modal language model includes Vision-language model. 
Vision-languge model includes different modalities at the same time, e.g., input is image and output is text (image caption); inputs are image and text, outputs are text (OpenAI-GPT, Microsoft-Bing, Google-Bard, open source model-LLaVA and MiniGPT-4).

Output logits of the languge model![image](https://github.com/XiaomengWang-AI/Knowledges-about-MLLMs/assets/76413485/3eea4cfb-40c0-4bea-af8b-4b07ec4845f0)



## Security on MLLMs
### Adversarial attack on VLMs
Lp norm is the parameter to quantify the magnitude of perturbations applied to input data. ||x||p, where x is a tensor.
L0-count the number of non-zero elements in the tensor;
L1-sum the absolute values of the element in the tensor;
L2-the Euclidean distance from the origin;
Linf-the maximum absolute value among the elements in the tensor;
code: torch.clamp(delta, -epsilon, epsilon)

### Other threats on VLMs


## Other Concepts
1. **RAG (Retrieval Augmented Generation)**
2. **TTS (text-to-speech)**

## References
1. Exploring Multimodal Large Language Models: A Step Forward in AI. https://medium.com/@cout.shubham/exploring-multimodal-large-language-models-a-step-forward-in-ai-626918c6a3ec
2. Extension on BLIP model series: https://raghul-719.medium.com/neural-networks-intuitions-17-blip-series-blip-blip-2-and-instruct-blip-papers-explanation-2378bc860d53
