import os
import torch
import torchaudio
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import cv2
import subprocess
import folder_paths
import folder_paths


# 定义一个将tensor转换为PIL图像的函数。
def tensor2PIL(tensor, thr=0.5):
    if tensor is None:
        return None
    tensor = tensor.squeeze()
    if tensor.dtype == torch.float32:
        tensor = tensor * 255
    if tensor.ndim == 3:
        tensor = tensor.permute(1, 2, 0)
    if tensor.shape[2] == 1:
        tensor = tensor.squeeze(2)
    tensor = tensor.cpu().numpy().astype(np.uint8)
    if tensor.ndim == 2:
        return Image.fromarray(tensor, mode='L')
    elif tensor.ndim == 3:
        if tensor.shape[2] == 3:
            return Image.fromarray(tensor, mode='RGB')
        elif tensor.shape[2] == 4:
            return Image.fromarray(tensor, mode='RGBA')
    return None

# 定义一个将PIL图像转换为tensor的函数
def PIL2tensor(pil_image):
    if pil_image is None:
        return None
    if pil_image.mode == 'RGB':
        arr = np.array(pil_image)
        arr = arr[:, :, ::-1]
        arr = arr.astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr)
        tensor = tensor.permute(2, 0, 1)
    elif pil_image.mode == 'RGBA':
        arr = np.array(pil_image)
        r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]
        arr = np.stack((r, g, b, a), axis=0)
        arr = arr.astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr)
    elif pil_image.mode == 'L':
        arr = np.array(pil_image)
        arr = arr.astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr)
        tensor = tensor.unsqueeze(0)
    else:
        pil_image = pil_image.convert('RGB')
        arr = np.array(pil_image)
        arr = arr[:, :, ::-1]
        arr = arr.astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr)
        tensor = tensor.permute(2, 0, 1)
    return tensor

# 生成艺术文字图像的函数
def generate_art_text_image(text, font_size, font_color, bg_color, width, height, align='center'):
    # 创建一个空白图像
    image = Image.new('RGBA', (width, height), bg_color)
    draw = ImageDraw.Draw(image)
    
    # 尝试加载中文字体
    try:
        font = ImageFont.truetype('simhei.ttf', font_size)
    except:
        # 如果找不到中文字体，使用默认字体
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    # 初始化默认值
    x, y = 0, 0
    
    # 计算文本的尺寸
    if font:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    else:
        text_width, text_height = 100, 50  # 假设的默认大小
    
    # 计算文本的位置以实现居中对齐
    if align == 'center':
        x = (width - text_width) // 2
        y = (height - text_height) // 2
    elif align == 'left':
        x = 10
        y = (height - text_height) // 2
    elif align == 'right':
        x = width - text_width - 10
        y = (height - text_height) // 2
    
    # 在图像上绘制文本
    if font:
        draw.text((x, y), text, font=font, fill=font_color)
    else:
        draw.text((x, y), text, fill=font_color)
    
    return image

class FTKSaveImage:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "images": ("IMAGE", {"default": None}),
            "output_dir": ("STRING", {"default": '', "multiline": False, "tooltip": "输出目录"}),
            "filename_prefix": ("STRING", {"default": "FTK"}),
            }}

    CATEGORY = "🏵️ FTK/I_O"
    RETURN_TYPES = ("FTKOUT",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save_images"
    OUTPUT_NODE = True

    def save_images(self, images, output_dir='', filename_prefix="FTK"):
        results = []
        paths=[]
        if images is None:
            return results
        
        # 确保输出目录存在        
        if output_dir=='':
            output_dir=folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        
        #图片名格式filename_prefix_00000.png 获取目录下png文件名匹配最后6位如果存在则从最后6位开始递增
        existing_files = [f for f in os.listdir(output_dir) if f.endswith('.png') and f.startswith(filename_prefix)]
        if existing_files:
            existing_numbers = []
            for f in existing_files:
                # 尝试提取数字部分
                try:
                    number_part = f.split('_')[-1].split('.')[0]
                    if number_part.isdigit():
                        existing_numbers.append(int(number_part))
                except:
                    continue
            
            if existing_numbers:
                next_number = max(existing_numbers) + 1
            else:
                next_number = 0
        else:
            next_number = 0


        # 处理每个图像
        for i, image in enumerate(images):
            # 确保图像是正确的格式
            if isinstance(image, torch.Tensor):
                # 将张量转换为PIL图像
                img_np = image.cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)
                
                # 处理图像维度
                if len(img_np.shape) == 4:
                    img_np = img_np[0]
                if img_np.shape[0] == 1 or img_np.shape[0] == 3 or img_np.shape[0] == 4:
                    img_np = np.transpose(img_np, (1, 2, 0))
                
                # 创建PIL图像
                if img_np.shape[2] == 1:
                    img_pil = Image.fromarray(img_np[:, :, 0], mode="L")
                elif img_np.shape[2] == 3:
                    img_pil = Image.fromarray(img_np, mode="RGB")
                elif img_np.shape[2] == 4:
                    img_pil = Image.fromarray(img_np, mode="RGBA")
                else:
                    img_pil = Image.fromarray(img_np)
            else:
                # 假设已经是PIL图像
                img_pil = image
            
            # 生成文件名 - 使用next_number并格式化为5位数字
            current_image_number = next_number + i
            filename = f"{filename_prefix}_{current_image_number:05d}.png"
            file_path = os.path.join(output_dir, filename)
            
            # 保存图像
            img_pil.save(file_path)
            
            # 添加到结果中
            results.append({
                "filename": filename,
                "subfolder": "",
                "type": "output"
            })
            paths.append(file_path)
        
        return {"result": (paths,),"ui":{"images": results}}

class FTKSaveVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames": ("IMAGE", {"default": None}),
                "output_dir": ("STRING", {"default": '', "multiline": False, "tooltip": "输出目录"}),
                "filename_prefix": ("STRING", {"default": "FTK"}),
                "fps": ("FLOAT", {"default": 25, "min": 1, "max": 60, "step": 0.1}),
            },
            "optional": {
                "audio": ("AUDIO", {"default": None, "tooltip": "可选的音频输入，提供时将合并到视频中"}),
            }
        }

    CATEGORY = "🏵️ FTK/I_O"
    RETURN_TYPES = ("FTKOUT",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save_video"
    OUTPUT_NODE = True

    def save_video(self, frames, output_dir='', filename_prefix="FTK", fps=24, audio=None):
        results = []
        paths = []
        if frames is None:
            return results
        
        # 确保输出目录存在
        if output_dir=='':
            output_dir=folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        
        #视频名格式filename_prefix_00000.mp4 获取目录下mp4文件名匹配最后6位如果存在则从最后6位开始递增
        existing_files = [f for f in os.listdir(output_dir) if f.endswith('.mp4') and f.startswith(filename_prefix)]
        if existing_files:
            existing_numbers = []
            for f in existing_files:
                # 尝试提取数字部分
                try:
                    number_part = f.split('_')[-1].split('.')[0]
                    if number_part.isdigit():
                        existing_numbers.append(int(number_part))
                except:
                    continue
            
            if existing_numbers:
                next_number = max(existing_numbers) + 1
            else:
                next_number = 0
        else:
            next_number = 0
        
        # 生成文件名 - 使用next_number并格式化为5位数字，确保使用.mp4扩展名
        filename = f"{filename_prefix}_{next_number:05d}.mp4"
        file_path = os.path.join(output_dir, filename)
        
        # 处理frames
        if isinstance(frames, torch.Tensor):
            # 确保frames是正确的格式
            if len(frames.shape) == 5:  # 批次维度
                frames = frames[0]
            
            # 转换为numpy数组
            frames_np = frames.cpu().numpy()
            
            # 处理维度和数据类型
            if frames_np.shape[-1] == 3:  # [frames, height, width, 3]
                # 确保值在0-1范围内
                if frames_np.max() <= 1.0:
                    frames_np = (frames_np * 255).astype(np.uint8)
                else:
                    frames_np = frames_np.astype(np.uint8)
                
                # 转换为BGR格式
                frames_np = frames_np[..., ::-1]  # RGB to BGR
                
                # 获取视频尺寸
                height, width = frames_np.shape[1], frames_np.shape[2]
            else:
                # 如果维度不正确，使用默认尺寸
                height, width = 512, 512
                frames_np = np.zeros((len(frames_np), height, width, 3), dtype=np.uint8)
        else:
            # 假设frames是一个列表
            if not frames:
                return results
            
            # 获取第一帧的尺寸
            first_frame = frames[0]
            if isinstance(first_frame, torch.Tensor):
                first_frame_np = first_frame.cpu().numpy()
                if first_frame_np.max() <= 1.0:
                    first_frame_np = (first_frame_np * 255).astype(np.uint8)
                height, width = first_frame_np.shape[0], first_frame_np.shape[1]
            else:
                height, width = first_frame.shape[0], first_frame.shape[1]
            
            # 转换所有帧
            frames_np = []
            for frame in frames:
                if isinstance(frame, torch.Tensor):
                    frame_np = frame.cpu().numpy()
                    if frame_np.max() <= 1.0:
                        frame_np = (frame_np * 255).astype(np.uint8)
                    frame_np = frame_np[..., ::-1]  # RGB to BGR
                else:
                    frame_np = frame[..., ::-1]  # RGB to BGR
                frames_np.append(frame_np)
            
            frames_np = np.array(frames_np)
        
        # 创建临时文件路径用于存储未压缩的视频帧，确保使用.mp4扩展名
        temp_raw_path = os.path.join(output_dir, f"{filename_prefix}_raw_{next_number:05d}.mp4")
        
        if audio is None:
            # 没有音频时，直接使用ffmpeg保存为H.264编码的视频
            # 使用ffmpeg命令行工具处理，确保使用H.264编码
            # 先将帧保存为临时文件，然后用ffmpeg转换
            
            # 使用OpenCV先保存为临时raw视频
            # 定义视频编码器 - 使用MJPG作为中间格式
            fourcc_temp = cv2.VideoWriter_fourcc(*'mp4v')
            temp_out = cv2.VideoWriter(temp_raw_path, fourcc_temp, fps, (width, height))
            
            # 写入每一帧到临时文件
            for frame in frames_np:
                temp_out.write(frame)
            
            # 释放临时VideoWriter
            temp_out.release()
            
            # 使用ffmpeg将临时视频转换为H.264编码的MP4
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', temp_raw_path,
                '-c:v', 'libx264',  # 使用H.264编码
                '-preset', 'medium',  # 编码速度和压缩率的平衡
                '-crf', '23',  # 恒定速率因子，越低质量越好
                '-c:a', 'aac',  # 音频编码为AAC
                '-strict', 'experimental',
                '-y',  # 覆盖已存在的文件
                file_path
            ]
            
            # 执行ffmpeg命令
            subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # 清理临时文件
            if os.path.exists(temp_raw_path):
                os.remove(temp_raw_path)
        else:
            # 有音频时的处理流程
            # 保存无声视频到临时文件，确保使用.mp4扩展名
            temp_video_path = os.path.join(output_dir, f"{filename_prefix}_temp_{next_number:05d}.mp4")
            
            # 使用MJPG作为中间格式保存临时视频
            fourcc_temp = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_path, fourcc_temp, fps, (width, height))
            
            # 写入每一帧
            for frame in frames_np:
                out.write(frame)
            
            # 释放VideoWriter
            out.release()
            
            # 保存音频到临时wav文件
            temp_audio_path = os.path.join(output_dir, f"{filename_prefix}_temp_{next_number:05d}.wav")
            # 处理音频数据
            for (batch_number, waveform) in enumerate(audio["waveform"].cpu()):
                torchaudio.save(temp_audio_path, waveform, audio["sample_rate"], format="wav")
                break  # 只处理第一个批次的音频
            
            # 使用ffmpeg合并视频和音频，并确保使用H.264编码
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', temp_video_path,
                '-i', temp_audio_path,
                '-c:v', 'libx264',  # 使用H.264编码视频
                '-preset', 'medium',
                '-crf', '23',
                '-c:a', 'aac',   # 音频编码为AAC
                '-strict', 'experimental',
                '-y',  # 覆盖已存在的文件
                file_path
            ]
            
            # 执行ffmpeg命令
            subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # 清理临时文件
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
        
        # 添加到结果中
         # 添加到结果中
        results.append({
            "filename": os.path.basename(file_path),
            "subfolder": "",
            "type": "output"
        })
        paths.append(file_path)
        return {"result": (paths,),"ui":{"gifs": results}}

class FTKSaveAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"audio": ("AUDIO", ),
                             "output_dir": ("STRING", {"default": '', "multiline": False, "tooltip": "输出目录"}),
                            "filename_prefix": ("STRING", {"default": "FTK"})},
                }

    RETURN_TYPES = ("FTKOUT",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save_audio"

    OUTPUT_NODE = True

    CATEGORY = "🏵️ FTK/I_O"

    def save_audio(self, audio, output_dir='', filename_prefix="FTK"):
        results = []
        paths = []
        if audio is None:
            return results
            
        # 确保输出目录存在
        if output_dir=='':
            output_dir=folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
            
        #音频名格式filename_prefix_00000.wav 获取目录下wav文件名匹配最后6位如果存在则从最后6位开始递增
        existing_files = [f for f in os.listdir(output_dir) if f.endswith('.wav') and f.startswith(filename_prefix)]
        if existing_files:
            existing_numbers = []
            for f in existing_files:
                # 尝试提取数字部分
                try:
                    number_part = f.split('_')[-1].split('.')[0]
                    if number_part.isdigit():
                        existing_numbers.append(int(number_part))
                except:
                    continue
            
            if existing_numbers:
                next_number = max(existing_numbers) + 1
            else:
                next_number = 0
        else:
            next_number = 0
            
        file_path = ""  # 初始化file_path变量
        for (batch_number, waveform) in enumerate(audio["waveform"].cpu()):
            # 生成文件名 - 使用next_number并格式化为5位数字
            current_number = next_number + batch_number
            file = f"{filename_prefix}_{current_number:05d}.wav"
            file_path = os.path.join(output_dir, file)
            torchaudio.save(file_path, waveform, audio["sample_rate"], format="wav")

            # 添加到结果中
             # 添加到结果中
            results.append({
                "filename": file,
                "subfolder": "",
                "type": "output"
            })
            paths.append(file_path)        
        return {"result": (paths,),"ui":{"audio": results}}

class FTKSaveText:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "text_content": ("STRING", {"forceInput": True}),
            "output_dir": ("STRING", {"default": '', "multiline": False, "tooltip": "输出目录"}),
            "filename_prefix": ("STRING", {"default": "FTK"})},
        }

    RETURN_TYPES = ("FTKOUT",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save_text"
    OUTPUT_NODE = True
    CATEGORY = "🏵️ FTK/I_O"

    def save_text(self, text_content, output_dir='', filename_prefix="FTK"):
        """保存文本内容到指定文件"""
        import os
        import folder_paths
        
        results = []
        
        # 确保输出目录存在
        if output_dir == '':
            output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        
        # 文件名格式: filename_prefix_00000.txt
        existing_files = [f for f in os.listdir(output_dir) if f.endswith('.txt') and f.startswith(filename_prefix)]
        max_number = -1
        
        if existing_files:
            for f in existing_files:
                try:
                    # 尝试提取数字部分
                    number_part = f.split('_')[-1].split('.')[0]
                    if number_part.isdigit():
                        max_number = max(max_number, int(number_part))
                except:
                    pass
        
        # 生成新文件名
        new_number = max_number + 1
        filename = f"{filename_prefix}_{new_number:05d}.txt"
        full_path = os.path.join(output_dir, filename)
        
        # 保存文本文件
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
        
        results.append(full_path)
        print(f"文本已保存到: {full_path}")
        
        return {"ui": {"text": (results,)}, "result": (results,)}

class FTK_OUTPUT:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "result": ("FTKOUT", {"forceInput": True}),
            }
    }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "out"
    CATEGORY = "🏵️ FTK/I_O"
    OUTPUT_NODE = True 
    def out(self, result):
        #收到的FTKOUT类型数据，解析为List(file_path)
        print("FTKOUT--in>>:",result)
        if isinstance(result, list):
            out = "|".join(result)
        print("FTKOUT--out>>:",out)
        return {"ui": {"FTKOUTPUT": (result,)}, "result": (result,)}

class FTK_INPUT:
    """
    整合节点，合并了文本、图片、视频和音频输入功能
    特点：
    1. 所有输入都是可选的
    2. 添加了完善的异常处理
    3. 按分类调整了输入参数的位置
    4. 优化了命名规范
    5. 支持两个视频输入
    6. 移除了scale_by参数
    """
    @classmethod
    def INPUT_TYPES(s):
        
        return {
            "required": {
                # 通用设置
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8, "tooltip": "图像和视频的宽度"}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8, "tooltip": "图像和视频的高度"}),
                "length": ("INT", {"default": 48, "min": 16, "max": 1200, "step": 1, "tooltip": "预期视频的长度"}),
                "output_dir": ("STRING", {"default": '', "multiline": False, "tooltip": "输出目录"}),
            },
            "optional": {
                # 文本输入部分
                "text_input_a": ("STRING", {"default": '', "multiline": True, "tooltip": "主要文本输入"}),
                "text_input_b": ("STRING", {"default": '', "multiline": True, "tooltip": "次要文本输入"}),
                "text_input_c": ("STRING", {"default": '', "multiline": True, "tooltip": "第三文本输入"}),
                "text_input_d": ("STRING", {"default": '', "multiline": True, "tooltip": "第四文本输入"}),
                
                # 图片输入部分
                "image_1_path": ("STRING", {"default": '', "multiline": False, "tooltip": "第一张图片路径"}),
                "image_2_path": ("STRING", {"default": '', "multiline": False, "tooltip": "第二张图片路径"}),
                "image_3_path": ("STRING", {"default": '', "multiline": False, "tooltip": "第三张图片路径"}),
                
                # 视频输入部分 - 视频1
                "video_1_file": ("STRING", {"default": "", "multiline": False, "tooltip": "第一个视频文件路径"}),
                "video_1_frame_rate": ("FLOAT", {"default": 24, "min": 1, "max": 60, "step": 0.1, "tooltip": "第一个视频帧率"}),
                "video_1_max_frames": ("INT", {"default": 64, "min": 1, "max": 10000, "step": 1, "tooltip": "第一个视频最大输出帧数"}),
                "video_1_start_frame": ("INT", {"default": 0, "min": 0, "step": 1, "tooltip": "第一个视频起始帧"}),
                
                # 视频输入部分 - 视频2
                "video_2_file": ("STRING", {"default": "", "multiline": False, "tooltip": "第二个视频文件路径"}),
                "video_2_frame_rate": ("FLOAT", {"default": 24, "min": 1, "max": 60, "step": 0.1, "tooltip": "第二个视频帧率"}),
                "video_2_max_frames": ("INT", {"default": 64, "min": 1, "max": 10000, "step": 1, "tooltip": "第二个视频最大输出帧数"}),
                "video_2_start_frame": ("INT", {"default": 0, "min": 0, "step": 1, "tooltip": "第二个视频起始帧"}),
                
                # 音频输入部分
                "audio_main_path": ("STRING", {"default": "", "multiline": False, "tooltip": "主音频路径"}),
                "audio_alternate_path": ("STRING", {"default": "", "multiline": False, "tooltip": "备用音频路径"}),
            }
        }

    CATEGORY = "🏵️ FTK/I_O"
    
    # 返回类型包括所有输入节点的返回类型
    RETURN_TYPES = (
        # 常用参数
        "INT", "INT", "INT", "STRING",
        # 文本返回
        "STRING", "STRING", "STRING", "STRING",
        # 图片返回
        "IMAGE", "MASK", "IMAGE", "MASK", "IMAGE", "MASK",
        # 视频返回
        "IMAGE",  # 视频1
        "IMAGE",  # 视频2
        # 音频返回
        "AUDIO", "AUDIO"
    )
    
    RETURN_NAMES = (
        # 常用参数
        "width", "height", "length", "output_dir",
        # 文本返回名称
        "text_a", "text_b", "text_c", "text_d",
        # 图片返回名称
        "image_1", "mask_1", "image_2", "mask_2", "image_3", "mask_3",
        # 视频返回名称 - 视频1
        "video_1_frames",
        # 视频返回名称 - 视频2
        "video_2_frames",
        # 音频返回名称
        "audio_1", "audio_2"
    )
    
    FUNCTION = "process_all_media"

    def load_single_image(self, image_path, width, height):
        """加载单张图片"""
        try:
            # 检查路径是否为空
            if not image_path or not image_path.strip():
                # 返回默认图片和遮罩
                default_image = torch.zeros((1, height, width, 3), dtype=torch.float32)
                default_mask = torch.zeros((height, width), dtype=torch.float32, device="cpu")
                return default_image, default_mask
            
            # 检查文件是否存在
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found: {image_path}")
                # 返回默认图片和遮罩
                default_image = torch.zeros((1, height, width, 3), dtype=torch.float32)
                default_mask = torch.zeros((height, width), dtype=torch.float32, device="cpu")
                return default_image, default_mask
            
            # 加载图片
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            width_img, height_img = image.size
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]

            # 处理遮罩
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
                if mask.shape != (height_img, width_img):
                    mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), 
                                                          size=(height_img, width_img), 
                                                          mode='bilinear', 
                                                          align_corners=False).squeeze()
            else:
                mask = torch.zeros((height_img, width_img), dtype=torch.float32, device="cpu")

            return image, mask
        except Exception as e:
            print(f"Warning: Failed to load image {image_path}: {e}")
            # 返回默认图片和遮罩
            default_image = torch.zeros((1, height, width, 3), dtype=torch.float32)
            default_mask = torch.zeros((height, width), dtype=torch.float32, device="cpu")
            return default_image, default_mask

    def load_video_file(self, video_path, frame_rate, max_frames, start_frame=0):
        """加载视频"""
        try:
            # 检查路径是否为空
            if not video_path or not video_path.strip() or video_path == "None":
                return (torch.zeros((1, 512, 512, 3)), 0, 512, 512)
            
            # 检查文件是否存在
            if not os.path.exists(video_path):
                print(f"Warning: Video file not found: {video_path}")
                return (torch.zeros((1, 512, 512, 3)), 0, 512, 512)
            
            cap = cv2.VideoCapture(video_path)
            
            # 检查视频是否成功打开
            if not cap.isOpened():
                print(f"Warning: Failed to open video file: {video_path}")
                cap.release()
                return (torch.zeros((1, 512, 512, 3)), 0, 512, 512)
            
            # 获取视频的原始帧率和总帧数
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 确保start_frame不超过总帧数
            start_frame = min(start_frame, total_frames - 1)
            
            # 设置起始帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # 计算帧率比例
            fps_ratio = original_fps / frame_rate if frame_rate > 0 else 1
            
            # 计算实际最大帧数
            remaining_frames = total_frames - start_frame
            max_frames = min(max_frames, remaining_frames)
            
            frames = []
            frame_idx = start_frame
            target_frame = 0
            
            # 读取视频帧
            while frame_idx < total_frames and len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 按照目标帧率采样
                if (frame_idx - start_frame) >= int(target_frame * fps_ratio):
                    # 转换BGR为RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # 归一化
                    frame = frame.astype(np.float32) / 255.0
                    frames.append(frame)
                    target_frame += 1
                
                frame_idx += 1
            
            cap.release()
            
            # 转换为张量
            if frames:
                frames_tensor = torch.from_numpy(np.stack(frames, axis=0))
            else:
                # 如果没有读取到帧，返回一个空的张量
                frames_tensor = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            
            return (frames_tensor, len(frames), width, height)
        except Exception as e:
            print(f"Warning: Failed to load video {video_path}: {e}")
            return (torch.zeros((1, 512, 512, 3)), 0, 512, 512)

    def load_audio_file(self, audio_path):
        """加载音频"""
        try:
            # 检查路径是否为空
            if not audio_path or not audio_path.strip():
                return None
            
            # 检查文件是否存在
            if not os.path.exists(audio_path):
                print(f"Warning: Audio file not found: {audio_path}")
                return None
            
            waveform, sample_rate = torchaudio.load(audio_path)
            audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
            # 检查是否是单声道
            if waveform.shape[0] == 1:
                # 复制单声道信号到双声道
                waveform = torch.cat([waveform, waveform], dim=0)
                audio["waveform"] = waveform.unsqueeze(0)
            return audio
        except Exception as e:
            print(f"Warning: Failed to load audio {audio_path}: {e}")
            return None

    def process_all_media(self, 
                         width=512, height=512, length=48, output_dir='',
                         text_input_a='', text_input_b='', text_input_c='', text_input_d='',
                         image_1_path='', image_2_path='', image_3_path='',
                         video_1_file='', video_1_frame_rate=24, video_1_max_frames=100, video_1_start_frame=0,
                         video_2_file='', video_2_frame_rate=24, video_2_max_frames=100, video_2_start_frame=0,
                         audio_main_path='', audio_alternate_path=''):
        """处理所有媒体输入"""
        try:
            # 文本处理
            processed_text_a = text_input_a if text_input_a else ''
            processed_text_b = text_input_b if text_input_b else ''
            processed_text_c = text_input_c if text_input_c else ''
            processed_text_d = text_input_d if text_input_d else ''
            
            # 图片处理
            image_1, mask_1 = self.load_single_image(image_1_path, width, height)
            image_2, mask_2 = self.load_single_image(image_2_path, width, height)
            image_3, mask_3 = self.load_single_image(image_3_path, width, height)
            
            # 视频处理 - 视频1
            video_1_frames, _, _, _ = self.load_video_file(
                video_1_file, 
                video_1_frame_rate, video_1_max_frames, video_1_start_frame
            )
            
            # 视频处理 - 视频2
            video_2_frames, _, _, _ = self.load_video_file(
                video_2_file, 
                video_2_frame_rate, video_2_max_frames, video_2_start_frame
            )
            
            # 音频处理
            audio_main = self.load_audio_file(audio_main_path)
            audio_alternate = self.load_audio_file(audio_alternate_path)
            
            
            # 返回所有结果 - 按照RETURN_NAMES的顺序
            return (
                # 常用参数
                width, height, length, output_dir,
                # 文本返回
                processed_text_a, processed_text_b, processed_text_c, processed_text_d,
                # 图片返回
                image_1, mask_1, image_2, mask_2, image_3, mask_3,
                # 视频返回
                video_1_frames,
                video_2_frames,
                # 音频返回
                audio_main, audio_alternate
            )
        except Exception as e:
            print(f"Error in process_all_media: {e}")
            # 返回默认值
            default_image = torch.zeros((1, height, width, 3), dtype=torch.float32)
            default_mask = torch.zeros((height, width), dtype=torch.float32, device="cpu")
            # 计算length（可以使用height作为默认值）
            length = height
            
            return (
                # 常用参数
                width, height, length,
                # 文本返回
                '', '', '', '',
                # 图片返回
                default_image, default_mask, default_image, default_mask, default_image, default_mask,
                # 视频返回
                torch.zeros((1, 512, 512, 3)),
                torch.zeros((1, 512, 512, 3)),
                # 音频返回
                None, None
            )


# 节点映射字典
NODE_CLASS_MAPPINGS = {
    "FTKSaveImage": FTKSaveImage,   
    "FTKSaveVideo": FTKSaveVideo,    
    "FTKSaveText": FTKSaveText,
    "FTK_OUTPUT": FTK_OUTPUT,    
    "FTKSaveAudio": FTKSaveAudio,
    "FTK_INPUT": FTK_INPUT,
}
