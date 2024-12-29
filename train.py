import time
from options.train_options import TrainOptions
from data.paired_dataset import PairedDataset  # 导入自定义数据集
from models import create_model
from util.visualizer import Visualizer
from torch.utils.data import DataLoader
from torchvision import transforms

if __name__ == '__main__':
    opt = TrainOptions().parse()

    # 定义数据转换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # 使用 PairedDataset 创建数据集
    # 去掉 root_dir 参数，直接传递 opt
    train_dataset = PairedDataset(opt)
    dataset = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_threads)

    dataset_size = len(train_dataset)
    print(f'The number of training images = {dataset_size}')

    model = create_model(opt)
    model.setup(opt)

    visualizer = Visualizer(opt)
    total_iters = 0

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:
                print(f'Saving the latest model (epoch {epoch}, total_iters {total_iters})')
                model.save_networks('latest')
                model.save_networks(epoch)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print(f'Saving the model at the end of epoch {epoch}, iters {total_iters}')
            model.save_networks('latest')
            model.save_networks(epoch)

        print(f'End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} \t Time Taken: {time.time() - epoch_start_time} sec')
        model.update_learning_rate()