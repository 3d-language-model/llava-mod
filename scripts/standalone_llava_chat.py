import argparse
import torch
from transformers import logging as hf_logging

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO

# Set logging verbosity for the transformers package to only log errors
hf_logging.set_verbosity_error()


class LLaVaChat(object):

    def __init__(self, args):
        # Model
        disable_torch_init()

        self.model_name = get_model_name_from_path(args.model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            args.model_path, args.model_base, self.model_name
        )

        self.conv_mode = None
        if 'llama-2' in self.model_name.lower():
            self.conv_mode = "llava_llama_2"
        elif "v1" in self.model_name.lower():
            self.conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            self.conv_mode = "mpt"
        else:
            self.conv_mode = "llava_v0"

        self.reset()

    def reset(self):
        # Initialize a conversation from template (default conv_mode is "multimodal")
        # (conv_mode determines the conversation template to use from llava.conversation module)
        self.conv = conv_templates[self.conv_mode].copy()

        # Cache for image features
        self.image_features = None
    
    def __call__(self, query, image_features=None):

        qs = query
        if image_features is not None:
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        else:
            qs = qs + '\n'
        
        if self.image_features is None:
            self.image_features = image_features
        
        self.conv.append_message(self.conv.roles[0], qs)
        self.conv.append_message(self.conv.roles[1], None)

        input_ids = None
        # Get the prompt
        prompt = self.conv.get_prompt()
        # Tokenize this prompt
        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).cuda()

        stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=None,
                image_features=self.image_features,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        self.conv.append_message(self.conv.roles[1], outputs)
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        return outputs
    
    def load_image(self, image_file):
        if image_file.startswith('http') or image_file.startswith('https'):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')
        return image
    
    def encode_image(self, image_tensor_half_cuda):
        return self.model.encode_images(image_tensor_half_cuda)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    args = parser.parse_args()

    # eval_model(args)

    chat = LLaVaChat(args)
    image = chat.load_image(args.image_file)
    image_tensor = chat.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
    image_features = chat.encode_image(image_tensor.half().cuda())
    print_separator = "====================================="
    query = "List the set of objects in this image."
    outputs = chat(query=query, image_features=image_features)
    print(query)
    print(print_separator)
    print(outputs)
    print(print_separator)

    query = "List potential uses for each of these objects."
    print(query)
    print(print_separator)
    outputs = chat(query=query, image_features=None)
    print(outputs)
    print(print_separator)

