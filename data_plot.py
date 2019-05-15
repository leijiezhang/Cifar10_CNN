import matplotlib.pyplot as plt
import torch


def _plot_img_item(img_orig, classes, conf_img, tag_img):
    plt.figure(figsize=(12, 4))
    # subplot the original image
    plt.subplot(121)
    img_show = img_orig
    plt.imshow(img_show)

    # subplot the confidence level of the current image
    plt.subplot(122)
    img_conf_item = torch.reshape(conf_img, (len(conf_img), 1))
    img_tag_item = torch.reshape(tag_img, (len(tag_img), 1))
    img_conf_cmp = torch.cat((img_conf_item, img_tag_item), 1)
    cmap = plt.cm.jet

    img_conf_plt = plt.imshow(img_conf_cmp.numpy(), origin='lower', cmap=cmap)
    ax = plt.gca()
    my_y_ticks = torch.linspace(0, len(img_conf_cmp) - 1, len(img_conf_cmp)).numpy()
    my_x_ticks = torch.linspace(0, len(torch.t(img_conf_cmp)) - 1, len(torch.t(img_conf_cmp))).numpy()
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    ax.set_xticklabels(('con', 'cls'))

    ax.set_yticklabels(classes)
    # show colorbar
    cbar = plt.colorbar(img_conf_plt)
    cbar.set_label('conf level')

    plt.show()


def _plot_img_trend(output_arr_train, output_arr_val, result_dir, result_file, remark):
    plt.figure(figsize=(17, 5))
    # subplot the results in training session
    ax1 = plt.subplot(121)
    # ax1 = plt.subplots()
    ax2 = ax1.twinx()
    train_list_size = len(output_arr_train)
    idx = torch.linspace(1, train_list_size, train_list_size).numpy()
    indices = torch.tensor([2])
    lr_values = torch.index_select(output_arr_train, 1, indices).view(train_list_size).numpy()
    indices = torch.tensor([1])
    loss_values = torch.index_select(output_arr_train, 1, indices).view(train_list_size).numpy()
    indices = torch.tensor([0])
    acc_values = torch.index_select(output_arr_train, 1, indices).view(train_list_size).numpy()
    ax1.plot(idx, loss_values, 'b-', label='loss')
    ax1.plot(idx, acc_values, 'r-.', label='acc')
    ax2.plot(idx, lr_values, 'm-.', label='lr')
    plt.title('Train')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('acc/loss')
    ax2.set_ylabel('')
    ax2.legend(loc=9)
    ax1.legend(loc=1)

    # subplot the results in training session
    plt.subplot(122)
    test_list_size = len(output_arr_val)
    idx = torch.linspace(1, test_list_size, test_list_size).numpy()
    indices = torch.tensor([1])
    loss_values = torch.index_select(output_arr_val, 1, indices).view(test_list_size).numpy()
    indices = torch.tensor([0])
    acc_values = torch.index_select(output_arr_val, 1, indices).view(test_list_size).numpy()
    plt.plot(idx, loss_values, 'b-', label='loss')
    plt.plot(idx, acc_values, 'r-.', label='acc')
    plt.title('test')
    plt.xlabel('epoch')
    plt.ylabel('acc/loss')
    plt.legend(loc='best')
    title_arry = result_file.split('-')
    sup_title = f'net:{title_arry[0]}/acc:{title_arry[1]}/lr:{title_arry[2]}{remark}'
    plt.suptitle(sup_title)
    # result_plot_dir = f'{result_dir}/{result_file}.eps'
    # plt.savefig(result_plot_dir)
    result_plot_dir = f'{result_dir}/{result_file}.png'
    plt.savefig(result_plot_dir)
    plt.show()


