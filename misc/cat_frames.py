from PIL import Image
import os

def concatenate_images(image_folder, output_path, direction='horizontal'):
    # 获取文件夹中的所有图片文件
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('png', 'jpg', 'jpeg'))]
    images = [Image.open(os.path.join(image_folder, img)) for img in image_files]

    # 获取所有图片的宽度和高度
    widths, heights = zip(*(img.size for img in images))

    if direction == 'horizontal':
        # 计算拼接后的总宽度和最大高度
        total_width = sum(widths)
        max_height = max(heights)
        # 创建一个新的空白图像
        new_image = Image.new('RGB', (total_width, max_height))
        # 拼接图片
        x_offset = 0
        for img in images:
            new_image.paste(img, (x_offset, 0))
            x_offset += img.width
    else:
        # 计算拼接后的总高度和最大宽度
        total_height = sum(heights)
        max_width = max(widths)
        # 创建一个新的空白图像
        new_image = Image.new('RGB', (max_width, total_height))
        # 拼接图片
        y_offset = 0
        for img in images:
            new_image.paste(img, (0, y_offset))
            y_offset += img.height

    # 保存拼接后的图片
    new_image.save(output_path)
    print(f"Concatenated image saved to {output_path}")

# 示例用法
image_folder = '/Users/sunqifan/Desktop/egoschema_videos/output_frames/500_0b4529ac-5a4e-4d30-b6b6-c6504c509c0c'
output_path = os.path.join(image_folder, 'concat_image.jpg')
concatenate_images(image_folder, output_path, direction='horizontal')