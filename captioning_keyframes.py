# Ref: https://huggingface.co/OFA-Sys/ofa-large

from PIL import Image
from torchvision import transforms
import torch
import argparse
import os
import cv2
from transformers import OFATokenizer, OFAModel
from OFA.transformers.src.transformers.models.ofa.generate import sequence_generator

mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
resolution = 480
patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
        transforms.ToTensor(), 
        transforms.Normalize(mean=mean, std=std)
    ])

tokenizer = OFATokenizer.from_pretrained("OFA-huge")
txt = " what does the image describe?"
inputs = tokenizer([txt], return_tensors="pt").input_ids
model = OFAModel.from_pretrained("OFA-huge", use_cache=True)

def image_caption(img):
    patch_img = patch_resize_transform(img).unsqueeze(0)

    # using the generator of fairseq version
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



def calculate_histogram(frame):
    # フレームのヒストグラムを計算
    histogram = cv2.calcHist([frame], [0], None, [256], [0, 256])
    # ヒストグラムを正規化
    histogram = cv2.normalize(histogram, histogram)
    return histogram


def histogram_difference(hist1, hist2):
    # ヒストグラム差分を計算
    diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return diff

def create_folder(video_path, video_name):
    # ビデオファイル名と同じ名前のフォルダを作成
    folder_path = os.path.join(os.path.dirname(video_path), video_name)

    # 既に存在する場合は作成しない
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def keyframe_caption(args):
    # フォルダ作成
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    create_folder(args.video_path, video_name)

    cap = cv2.VideoCapture(args.video_path)
    threshold = args.th

    if not cap.isOpened():
        print("Error: Unable to open video.")
        return
    
    # 最初のフレームを読み込み
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Unable to read the first frame.")
        return
    
    # 最初のフレームのヒストグラムを計算
    prev_hist = calculate_histogram(prev_frame)
    
    keyframes_caption = []
    frame_index = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # ヒストグラムを計算
        current_hist = calculate_histogram(frame)
        
        # ヒストグラム差分を計算
        diff = histogram_difference(prev_hist, current_hist)
        
        # ヒストグラム差分が閾値より大きい場合はキーフレームとして追加
        if diff < threshold:
            prev_hist = current_hist

            pillow_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pillow_image.save("./"+video_name+"/"+str(frame_index)+".png")

            result_txt = image_caption(pillow_image)
            keyframes_caption.append(result_txt)
          
        frame_index += 1
    
    cap.release()

    for i in range(len(keyframes_caption)):
        print(keyframes_caption[i])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--video_path', type=str, default="test.mp4")
    parser.add_argument('--th', type=float, default=0.95)
    parser.add_argument('--ckpt_dir', type=str, default="./OFA-huge")
    args = parser.parse_args()

    # 実行
    result_txt = keyframe_caption(args)
    print(result_txt)