---
layout: post
title: "Run GLM-4.6V on AMD MI300X GPU with vLLM"
date: 2025-12-09
categories: [LLM, AMD, MI300X, vLLM]
---

GLM-4.6V is the latest multimodal model from Z.AI, designed to bridge the gap between visual perception and executable action. In this post, we'll explore what makes GLM-4.6V special and how you can run it on AMD's powerful MI300X GPUs using vLLM.

## 1. Overview about GLM-4.6V

GLM-4.6V is a 106B parameter foundation model that achieves State-of-the-Art (SoTA) performance in visual understanding, comparable to other leading models like GPT-4V. It introduces several groundbreaking capabilities:

*   **Native Multimodal Function Calling:** Unlike previous models that required converting visual inputs to text descriptions, GLM-4.6V can directly process images, screenshots, and documents as tool inputs. It can also generate visual outputs like charts and rendered pages, integrating them into its reasoning chain.
*   **Interleaved Image-Text Content Generation:** The model can synthesize coherent content that mixes text and images, ideal for generating rich reports or articles.
*   **Multimodal Document Understanding:** With a context window of up to 128k tokens, it can process and understand long documents, charts, and complex layouts without OCR pre-processing.
*   **Frontend Replication & Visual Editing:** It can reconstruct HTML/CSS from screenshots and support natural language-driven edits.

For those with more constrained resources, a lightweight version, **GLM-4.6V-Flash (9B)**, is also available for local deployment.

## 2. How to run on AMD MI300X GPU

Running GLM-4.6V on AMD MI300X is straightforward thanks to vLLM support. Ensure you have a working ROCm environment set up for your MI300X.

### Prerequisites & Installation


Try it by launching the vLLM container: 

```bash
docker run -it \
 --privileged \
 --network=host \
 --group-add=video \
 --ipc=host \
 --cap-add=SYS_PTRACE \
 --security-opt seccomp=unconfined \
 --device /dev/kfd \
 --device /dev/dri \
 --name vllm-omni \
 rocm/vllm-dev:nightly
```

You need to install `transformers` with version >= 0.5.0 

```bash
https://github.com/huggingface/transformers.git
pip install '.[torch]'
```

### Running Inference

Launch vLLM server:

```bash
vllm serve zai-org/GLM-4.6V \
     --tensor-parallel-size 4 \
     --tool-call-parser glm45 \
     --reasoning-parser glm45 \
     --enable-auto-tool-choice \
     --enable-expert-parallel \
     --allowed-local-media-path / \
     --mm-encoder-tp-mode data \
     --mm_processor_cache_type shm
```
You can also use --tensor-parallel-size 2 and 8 to run on 2 or 8 MI300X GPU. 
The same command can be used to run zai-org/GLM-4.6V-FP8 on 1, 2, 4, 8 MI300X GPU.

Here are two quick examples of demonstrating the capabilities of GLM-4.6V.

Example 1: Image Description

![Visual Grounding Example](https://cloudcovert-1305175928.cos.ap-guangzhou.myqcloud.com/%E5%9B%BE%E7%89%87grounding.PNG)



```bash
curl -X POST \
    http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "zai-org/GLM-4.6V",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://cloudcovert-1305175928.cos.ap-guangzhou.myqcloud.com/%E5%9B%BE%E7%89%87grounding.PNG"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Where is the second bottle of beer from the right on the table?  Provide coordinates in [[xmin,ymin,xmax,ymax]] format"
                    }
                ]
            }
        ],
        "thinking": {
            "type":"enabled"
        }
    }'
```

The output: 

{"id":"chatcmpl-afb2ac2dce2bd986","object":"chat.completion","created":1765416718,"model":"/models/GLM-4.6V","choices":[{"index":0,"message":{"role":"assistant","content":"\nThe coordinates of the second bottle of beer from the right on the table are <|begin_of_box|>[[94,598,177,991]]<|end_of_box|>.","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning":"The image shows an outdoor table setting with various items on it, including bottles of beer. The question asks for the coordinates of the second bottle of beer from the right on the table. By visually inspecting the table, we identify the bottles of beer and count from the right - hand side to find the second one. Then, we determine the bounding box coordinates of that specific bottle.","reasoning_content":"The image shows an outdoor table setting with various items on it, including bottles of beer. The question asks for the coordinates of the second bottle of beer from the right on the table. By visually inspecting the table, we identify the bottles of beer and count from the right - hand side to find the second one. Then, we determine the bounding box coordinates of that specific bottle."},"logprobs":null,"finish_reason":"stop","stop_reason":151336,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":696,"total_tokens":807,"completion_tokens":111,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}

You can see it can successfully identify the second bottle of beer from the right on the table and provide the coordinates [94,598,177,991]. It also shows the reasoning process in the "reasoning_content" field.


## 3. Summary and Call to Action

GLM-4.6V represents a significant leap forward in open multimodal AI, bringing native visual tool use and long-context understanding to the forefront. When paired with the high-bandwidth memory and compute power of AMD MI300X GPUs, it becomes a formidable tool for enterprise-grade multimodal applications.

We encourage you to try running GLM-4.6V on your AMD infrastructure today! Check out the [official documentation](https://docs.z.ai/guides/vlm/glm-4.6v) and the [Hugging Face model card](https://huggingface.co/zai-org/GLM-4.6V) for more deep dives.
