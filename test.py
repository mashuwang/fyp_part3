import os
from options.test_options import TestOptions
from data.paired_dataset import PairedDataset  # 使用与训练相同的数据集
from models import create_model
from util.visualizer import save_images
from util import html
from torch.utils.data import DataLoader
from torchvision import transforms

if __name__ == '__main__':
    opt = TestOptions().parse()

    # 定义数据转换，与训练时一致
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # 使用 PairedDataset 创建数据集
    test_dataset = PairedDataset(opt)
    dataset = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_threads)

    # 创建模型
    model = create_model(opt)
    model.setup(opt)

    # 创建一个网站以保存结果
    web_dir = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{opt.epoch}')
    webpage = html.HTML(web_dir, f'Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {opt.epoch}')

    # 如果设置了 eval 模式，则将模型设置为评估模式
    if opt.eval:
        model.eval()

    # 遍历数据集进行测试
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # 仅对 opt.num_test 张图像应用模型
            break
        model.set_input(data)  # 从数据加载器中解包数据
        model.test()           # 运行推理
        visuals = model.get_current_visuals()  # 获取图像结果
        img_path = model.get_image_paths()     # 获取图像路径
        if i % 5 == 0:  # 每处理五张图像，保存一次结果
            print(f'processing ({i:04d})-th image... {img_path}')
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

    # 保存 HTML
    webpage.save()