import torch
import tkinter as tk
from load_data import _load_data
import torchvision.transforms as transforms
from data_plot import _plot_img_item
import tkinter.messagebox
from data_plot import _plot_img_trend


img_pos = 0
data_dir = './data'
data_name = 'cifar10'
result_file = 'densenet121-0.9310-0.0500'
result_dir = './results'
remark = '/normal'

# load data
transform = transforms.Compose(
        [transforms.Resize(252),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
_, _, classes, _, testset = _load_data(data_dir, data_name, transform, transform, 200)

data_result_dir = f'{result_dir}/{result_file}.pt'

data = torch.load(data_result_dir, map_location='cpu')
best_epoch = data['best_epoch']
best_acc = data['best_acc']
output_arr_val = data['output_arr_val']
output_arr_train = data['output_arr_train']
error_img_msg = data['error_img_msg']
img_show_idx = error_img_msg['image_udx'].data.to('cpu')
img_conf = error_img_msg['conf'].data.to('cpu')
img_show_set = testset.data[img_show_idx, ::]
# img_tag_idx = torch.Tensor(testset.targets)[img_show_idx]
img_tag = torch.zeros(len(img_conf), len(torch.t(img_conf)))
j = 0
for i in img_show_idx:
    img_tag[j, testset.targets[i]] =1
    j += 1


def get_screen_size(window):
    return window.winfo_screenwidth(), window.winfo_screenheight()


def get_window_size(window):
    return window.winfo_reqwidth(), window.winfo_reqheight()


def center_window(root, width, height):
    screenwidth = root.winfo_screenwidth()
    screenheight = root.winfo_screenheight()
    # size = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
    size = '%dx%d+%d+%d' % (width, height, 130, (screenheight - height) / 2)
    print(size)
    root.geometry(size)


top = tk.Tk()
top.title('Results Monitor')
center_window(top, 350, 150)


def nextCallBack():
    global img_pos
    print('picture position entering:', img_pos, '/', len(img_conf))
    tmp_img_pos = img_pos
    if img_pos < len(img_conf)-1:
        img_pos = tmp_img_pos + 1
        img_orig = img_show_set[img_pos, ::]
        conf_img = img_conf[img_pos, :]
        tag_img = img_tag[img_pos, :]
        _plot_img_item(img_orig, classes, conf_img, tag_img)
    else:
        tkinter.messagebox.showinfo("=============")
    print('picture position in the end:', img_pos, '/', len(img_conf))


def preCallBack():
    global img_pos
    print('picture position entering:', img_pos, '/', len(img_conf))
    tmp_img_pos = img_pos
    if img_pos > 0:
        img_pos -= img_pos
        img_pos = tmp_img_pos - 1
        img_orig = img_show_set[img_pos, ::]
        conf_img = img_conf[img_pos, :]
        tag_img = img_tag[img_pos, :]
        _plot_img_item(img_orig, classes, conf_img, tag_img)
    else:
        tkinter.messagebox.showinfo("=============")
    print('picture position in the end:', img_pos, '/', len(img_conf))


def jumpCallBack():
    global img_pos
    print('picture position entering:', img_pos, '/', len(img_conf))
    num_jump_str = E_jump.get()
    num_jump = int(num_jump_str)-1

    if num_jump > 0 and num_jump < len(img_conf):
        img_pos = num_jump
        img_orig = img_show_set[num_jump, ::]
        conf_img = img_conf[num_jump, :]
        tag_img = img_tag[num_jump, :]
        _plot_img_item(img_orig, classes, conf_img, tag_img)
    else:
        tkinter.messagebox.showinfo("=============")
    print('picture position in the end:', img_pos, '/', len(img_conf))


def trendCallBack():
    # plot the trend
    _plot_img_trend(output_arr_train, output_arr_val, result_dir, result_file, remark)


L_jump = tk.Label(top, width=10, height=3, text='Jump to：')
L_jump.grid(row=0, sticky='w')
E_jump = tk.Entry(top, width=15)
E_jump.grid(row=0, column=1, sticky='e')

B_jump = tk.Button(top, width=10, height=1, text="Jump", command=jumpCallBack)
B_jump.grid(row=0, column=2, sticky='e')

B_next = tk.Button(top, width=10, height=3, text="Next", command=nextCallBack)
B_next.grid(row=1, column=2, sticky='e')

B_pre = tk.Button(top, width=10, height=3, text="Pre", command=preCallBack)
B_pre.grid(row=1, sticky='e')

B_trend = tk.Button(top, width=10, height=3, text="Trend", command=trendCallBack)
B_trend.grid(row=1, column=1, sticky='e')

top.mainloop()

