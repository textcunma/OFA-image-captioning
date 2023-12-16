# Ref: https://huggingface.co/OFA-Sys/ofa-large

from PIL import Image
from torchvision import transforms
import torch
import argparse
from transformers import OFATokenizer, OFAModel
from OFA.transformers.src.transformers.models.ofa.generate import sequence_generator

def image_caption(args):
    image_path = args.image_path
    ckpt_dir = args.ckpt_dir

    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    resolution = 480
    patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
            transforms.ToTensor(), 
            transforms.Normalize(mean=mean, std=std)
        ])

    tokenizer = OFATokenizer.from_pretrained(ckpt_dir)

    txt = " what does the image describe?"
    inputs = tokenizer([txt], return_tensors="pt").input_ids
    img = Image.open(image_path)
    patch_img = patch_resize_transform(img).unsqueeze(0)

    # using the generator of fairseq version
    model = OFAModel.from_pretrained(ckpt_dir, use_cache=True)
    generator = sequence_generator.SequenceGenerator(
                        tokenizer=tokenizer,
                        beam_size=5,
                        max_len_b=16, 
                        min_len=0,
                        no_repeat_ngram_size=3,
                    )
    data = {}
    data["net_input"] = {"input_ids": inputs, 'patch_images': patch_img, 'patch_masks':torch.tensor([True])}
    gen_output = generator.generate([model], data)
    
    gen = [gen_output[i][0]["tokens"] for i in range(len(gen_output))]

    return tokenizer.batch_decode(gen, skip_special_tokens=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--image_path', type=str, default="test.png")
    parser.add_argument('--ckpt_dir', type=str, default="./OFA-huge")
    args = parser.parse_args()

    # 実行
    result_txt = image_caption(args)
    print(result_txt)